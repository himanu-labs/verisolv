/* tslint:disable */
/* eslint-disable */

export class OdeSystem {
    free(): void;
    [Symbol.dispose](): void;
    derivatives(t: number, y: Float64Array, params: Float64Array): Float64Array;
    dimension(): number;
    constructor(state_names: string, param_names: string, ode_expressions: string);
    rk4_step(t: number, y: Float64Array, params: Float64Array, h: number): Float64Array;
}

export function version(): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_odesystem_free: (a: number, b: number) => void;
    readonly odesystem_derivatives: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly odesystem_dimension: (a: number) => number;
    readonly odesystem_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly odesystem_rk4_step: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly version: (a: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number) => void;
    readonly __wbindgen_export3: (a: number, b: number, c: number, d: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
