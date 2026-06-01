//! Browser-native WebAssembly bindings for verisolv.
//!
//! The existing `rust_core` crate is a PyO3 extension module. This crate keeps
//! the browser artifact separate while reusing the pure Rust RK4 kernel by path.
//! ODE right-hand sides are parsed once into a small expression tree so the
//! integration loop can run inside WebAssembly without a per-derivative JS
//! callback.

#[path = "../../rust_core/src/ode/rk4.rs"]
mod rk4;

use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct OdeSystem {
    state_names: Vec<String>,
    param_names: Vec<String>,
    expressions: Vec<Expr>,
}

#[wasm_bindgen]
impl OdeSystem {
    #[wasm_bindgen(constructor)]
    pub fn new(
        state_names: &str,
        param_names: &str,
        ode_expressions: &str,
    ) -> Result<OdeSystem, JsValue> {
        let states = split_names(state_names);
        let params = split_names(param_names);
        if states.is_empty() {
            return Err(js_err("at least one state variable is required"));
        }

        let expressions: Result<Vec<_>, _> = ode_expressions
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(parse_expression)
            .collect();
        let expressions = expressions.map_err(|e| js_err(&e))?;

        if expressions.len() != states.len() {
            return Err(js_err(&format!(
                "expected {} ODE expression(s), got {}",
                states.len(),
                expressions.len()
            )));
        }

        Ok(OdeSystem {
            state_names: states,
            param_names: params,
            expressions,
        })
    }

    pub fn dimension(&self) -> usize {
        self.state_names.len()
    }

    pub fn rk4_step(
        &self,
        t: f64,
        y: Vec<f64>,
        params: Vec<f64>,
        h: f64,
    ) -> Result<Vec<f64>, JsValue> {
        self.validate_inputs(&y, &params)?;
        if !h.is_finite() || h == 0.0 {
            return Err(js_err("step size must be a finite non-zero number"));
        }

        let mut rhs = |time: f64, state: &[f64]| self.eval_rhs(time, state, &params);
        let (_times, ys) = rk4::rk4_core(&mut rhs, t, t + h, &y, 1);
        Ok((0..self.state_names.len()).map(|i| ys[(i, 1)]).collect())
    }

    pub fn derivatives(
        &self,
        t: f64,
        y: Vec<f64>,
        params: Vec<f64>,
    ) -> Result<Vec<f64>, JsValue> {
        self.validate_inputs(&y, &params)?;
        let values = self.eval_rhs(t, &y, &params);
        if values.iter().all(|value| value.is_finite()) {
            Ok(values)
        } else {
            Err(js_err("ODE expression produced a non-finite derivative"))
        }
    }
}

impl OdeSystem {
    fn validate_inputs(&self, y: &[f64], params: &[f64]) -> Result<(), JsValue> {
        if y.len() != self.state_names.len() {
            return Err(js_err(&format!(
                "state vector length {} does not match dimension {}",
                y.len(),
                self.state_names.len()
            )));
        }
        if params.len() != self.param_names.len() {
            return Err(js_err(&format!(
                "parameter vector length {} does not match parameter count {}",
                params.len(),
                self.param_names.len()
            )));
        }
        if !y.iter().all(|value| value.is_finite()) {
            return Err(js_err("state vector contains a non-finite value"));
        }
        if !params.iter().all(|value| value.is_finite()) {
            return Err(js_err("parameter vector contains a non-finite value"));
        }
        Ok(())
    }

    fn eval_rhs(&self, t: f64, y: &[f64], params: &[f64]) -> Vec<f64> {
        let mut scope = Scope::new(t);
        for (name, value) in self.state_names.iter().zip(y.iter()) {
            scope.insert(name, *value);
        }
        for (name, value) in self.param_names.iter().zip(params.iter()) {
            scope.insert(name, *value);
        }
        self.expressions
            .iter()
            .map(|expr| eval_expr(expr, &scope))
            .collect()
    }
}

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

fn js_err(message: &str) -> JsValue {
    JsValue::from_str(message)
}

fn split_names(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

#[derive(Clone, Debug)]
enum Expr {
    Number(f64),
    Name(String),
    Unary {
        op: UnaryOp,
        value: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Call {
        name: String,
        args: Vec<Expr>,
    },
}

#[derive(Clone, Copy, Debug)]
enum UnaryOp {
    Neg,
}

#[derive(Clone, Copy, Debug)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Number(f64),
    Name(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    LParen,
    RParen,
    Comma,
    Eof,
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Parser {
        Parser { tokens, pos: 0 }
    }

    fn parse(mut self) -> Result<Expr, String> {
        let expr = self.parse_add()?;
        if !matches!(self.peek(), Token::Eof) {
            return Err(format!("unexpected token {:?}", self.peek()));
        }
        Ok(expr)
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let token = self.peek().clone();
        if !matches!(token, Token::Eof) {
            self.pos += 1;
        }
        token
    }

    fn parse_add(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                Token::Plus => BinaryOp::Add,
                Token::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.advance();
            expr = Expr::Binary {
                op,
                left: Box::new(expr),
                right: Box::new(self.parse_mul()?),
            };
        }
        Ok(expr)
    }

    fn parse_mul(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_pow()?;
        loop {
            let op = match self.peek() {
                Token::Star => BinaryOp::Mul,
                Token::Slash => BinaryOp::Div,
                Token::Percent => BinaryOp::Rem,
                _ => break,
            };
            self.advance();
            expr = Expr::Binary {
                op,
                left: Box::new(expr),
                right: Box::new(self.parse_pow()?),
            };
        }
        Ok(expr)
    }

    fn parse_pow(&mut self) -> Result<Expr, String> {
        let left = self.parse_unary()?;
        if matches!(self.peek(), Token::Caret) {
            self.advance();
            return Ok(Expr::Binary {
                op: BinaryOp::Pow,
                left: Box::new(left),
                right: Box::new(self.parse_pow()?),
            });
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Token::Plus => {
                self.advance();
                self.parse_unary()
            }
            Token::Minus => {
                self.advance();
                Ok(Expr::Unary {
                    op: UnaryOp::Neg,
                    value: Box::new(self.parse_unary()?),
                })
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.advance() {
            Token::Number(value) => Ok(Expr::Number(value)),
            Token::Name(name) => {
                if !matches!(self.peek(), Token::LParen) {
                    return Ok(Expr::Name(name));
                }
                self.advance();
                let mut args = Vec::new();
                if !matches!(self.peek(), Token::RParen) {
                    loop {
                        args.push(self.parse_add()?);
                        if !matches!(self.peek(), Token::Comma) {
                            break;
                        }
                        self.advance();
                    }
                }
                match self.advance() {
                    Token::RParen => Ok(Expr::Call { name, args }),
                    other => Err(format!("expected closing parenthesis, found {other:?}")),
                }
            }
            Token::LParen => {
                let expr = self.parse_add()?;
                match self.advance() {
                    Token::RParen => Ok(expr),
                    other => Err(format!("expected closing parenthesis, found {other:?}")),
                }
            }
            other => Err(format!("unexpected token {other:?}")),
        }
    }
}

fn parse_expression(source: &str) -> Result<Expr, String> {
    Parser::new(tokenize(source)?).parse()
}

fn tokenize(source: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = source.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if ch.is_ascii_digit() || ch == '.' {
            let start = i;
            i += 1;
            while i < chars.len() {
                let current = chars[i];
                let prev = chars[i - 1];
                if current.is_ascii_digit() || current == '.' || current == 'e' || current == 'E' {
                    i += 1;
                    continue;
                }
                if (current == '+' || current == '-') && (prev == 'e' || prev == 'E') {
                    i += 1;
                    continue;
                }
                break;
            }
            let raw: String = chars[start..i].iter().collect();
            let value = raw
                .parse::<f64>()
                .map_err(|_| format!("invalid number `{raw}`"))?;
            tokens.push(Token::Number(value));
            continue;
        }
        if ch == '_' || ch.is_ascii_alphabetic() {
            let start = i;
            i += 1;
            while i < chars.len() && (chars[i] == '_' || chars[i].is_ascii_alphanumeric()) {
                i += 1;
            }
            tokens.push(Token::Name(chars[start..i].iter().collect()));
            continue;
        }
        tokens.push(match ch {
            '+' => Token::Plus,
            '-' => Token::Minus,
            '*' => Token::Star,
            '/' => Token::Slash,
            '%' => Token::Percent,
            '^' => Token::Caret,
            '(' => Token::LParen,
            ')' => Token::RParen,
            ',' => Token::Comma,
            _ => return Err(format!("unsupported character `{ch}`")),
        });
        i += 1;
    }
    tokens.push(Token::Eof);
    Ok(tokens)
}

struct Scope {
    values: HashMap<String, f64>,
}

impl Scope {
    fn new(t: f64) -> Scope {
        let mut values = HashMap::new();
        values.insert("t".to_string(), t);
        values.insert("pi".to_string(), std::f64::consts::PI);
        values.insert("e".to_string(), std::f64::consts::E);
        values.insert("tau".to_string(), std::f64::consts::TAU);
        Scope { values }
    }

    fn insert(&mut self, name: &str, value: f64) {
        self.values.insert(name.to_string(), value);
    }

    fn get(&self, name: &str) -> f64 {
        *self.values.get(name).unwrap_or(&f64::NAN)
    }
}

fn eval_expr(expr: &Expr, scope: &Scope) -> f64 {
    match expr {
        Expr::Number(value) => *value,
        Expr::Name(name) => scope.get(name),
        Expr::Unary { op, value } => match op {
            UnaryOp::Neg => -eval_expr(value, scope),
        },
        Expr::Binary { op, left, right } => {
            let a = eval_expr(left, scope);
            let b = eval_expr(right, scope);
            match op {
                BinaryOp::Add => a + b,
                BinaryOp::Sub => a - b,
                BinaryOp::Mul => a * b,
                BinaryOp::Div => a / b,
                BinaryOp::Rem => a % b,
                BinaryOp::Pow => a.powf(b),
            }
        }
        Expr::Call { name, args } => {
            let values: Vec<f64> = args.iter().map(|arg| eval_expr(arg, scope)).collect();
            match name.as_str() {
                "abs" if values.len() == 1 => values[0].abs(),
                "sqrt" if values.len() == 1 => values[0].sqrt(),
                "sin" if values.len() == 1 => values[0].sin(),
                "cos" if values.len() == 1 => values[0].cos(),
                "tan" if values.len() == 1 => values[0].tan(),
                "exp" if values.len() == 1 => values[0].exp(),
                "log" if values.len() == 1 => values[0].ln(),
                "floor" if values.len() == 1 => values[0].floor(),
                "ceil" if values.len() == 1 => values[0].ceil(),
                "round" if values.len() == 1 => values[0].round(),
                "min" if !values.is_empty() => values.into_iter().fold(f64::INFINITY, f64::min),
                "max" if !values.is_empty() => values.into_iter().fold(f64::NEG_INFINITY, f64::max),
                _ => f64::NAN,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_evaluates_expression() {
        let expr = parse_expression("sin(pi / 2) + 2 ^ 3").unwrap();
        let scope = Scope::new(0.0);
        assert!((eval_expr(&expr, &scope) - 9.0).abs() < 1e-12);
    }

    #[test]
    fn rk4_step_matches_exponential_decay() {
        let system = OdeSystem::new("y", "", "-y").unwrap();
        let y1 = system.rk4_step(0.0, vec![1.0], vec![], 0.01).unwrap();
        assert!((y1[0] - (-0.01_f64).exp()).abs() < 1e-10);
    }
}
