# Host arrays of a model on Reactant

- **Type:** improvement
- **Status:** in main since 2026-11-23, always on under `ReactantBackend`
- **Commits:** 0e36113 (feature)
- **Code:** `HostArray` (`ext/LaddieReactantExt.jl`); `_host_arrays`, `_host_array` (`src/utils/backend.jl`)

## Summary

The first call of a compiled program that returned the model, or its shadow, took minutes to hours, growing as cells^1.5.
Reactant rebuilds every plain `Array` in a program's result with one generated statement per element, and the model carries the integer masks of `Grid` and `Geometry` (and the coordinate vectors) on the host.
On Reactant these arrays are now wrapped in a read-only `HostArray`, which Reactant traces like a `BitArray` and embeds as one object in the result code; moving the model back unwraps them.
The first call at 1000×1000 dropped from about 3 h to 5–9 s.

## Evidence

First call of a compiled program, ISOMIP geometry, before the fix:

| program, setup | first call | later calls |
|---|---|---|
| `m.H.present .+= 1`, returns the model, CPU, 120×60 | 9.9 s | 0.2 ms |
| same, 240×120 | 53 s | 0.2 ms |
| same, 480×240 | 423 s | 0.4 ms |
| same, 240×120, returns `nothing` / one array | 0.8 s / 0.7 s | |
| same, 240×120, masks shrunk to 1×1 | 1.0 s | |
| reverse mode of `integrate!`, leapfrog, 20 checkpoints, H100, 480×240 | 387 s | 0.17 s |
| one forward step / `dq .+= q` with the model passed, H100, 480×240 | 387 s / 388 s | 1 ms |

- `@time` reports 100 % compilation time: Julia compiles the result code, whatever the backend and the program body.
- XLA settings change nothing: autotuning off 382 s, eager module loading 385 s, command buffers off 385 s.
- Only `grid.mask` and `geometry.resolved_mask` matter (22 s each at 240×120); the coordinate vectors cost 0.1 s.
- No 1000×1000 reverse-mode run had ever finished: about 3 h per compiled program.

After the fix: 240×120, program returning the model, 52.8 s → 1.0 s; 1000×1000 reverse mode of `integrate!`, first call 5 s (leapfrog) and 9 s (FB-RK3).

## Caveats

- `HostArray` extends three Reactant internals (`make_tracer`, `traced_type_inner`, `Compiler.create_result`); a Reactant upgrade can break them. The test "host-only arrays pass through a compiled program" (`test/reactant/runtests.jl`) checks the wrapping, the round trip and that a returned model keeps the same mask objects.
- `HostArray` behaves as an `AbstractArray` for indexing, broadcasting and reductions, but code that checks `isa Matrix` on a Reactant model fails.
- A new host array in `Grid` or `Geometry` is wrapped through `_host_array`; any other plain `Array` must stay out of what compiled programs return (AGENTS.md, Reactant).
