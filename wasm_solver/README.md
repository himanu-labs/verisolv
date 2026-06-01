# verisolv-wasm

Browser-native bindings for verisolv's ODE kernels.

Build:

```sh
wasm-pack build --target web --out-dir pkg --release
```

The package exposes:

- `OdeSystem.new(state_names, param_names, ode_expressions)`
- `system.rk4_step(t, y, params, dt)`
- `system.derivatives(t, y, params)`
- `version()`

`state_names` and `param_names` are comma-separated names. `ode_expressions`
contains one expression per state, separated by newlines. Expressions support
numbers, state variables, parameters, `t`, constants `pi`, `e`, `tau`, binary
operators `+ - * / % ^`, unary signs, parentheses, and scalar math calls such as
`sin`, `cos`, `sqrt`, `exp`, `log`, `min`, and `max`.
