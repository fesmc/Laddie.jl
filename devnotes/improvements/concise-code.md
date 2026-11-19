# Concise code before v0.1

- **Type:** improvement
- **Status:** in main since 2026-11-19
- **Commits:** 931d884 (shared kernels and launchers), 014d1b6 (docs, comments, devnotes)
- **Code:** `_ambient_interp_kernel!`, `_depth_levels` in `src/bcs/forcing.jl`; `_tracer_transport!`, `_step_momentum!` in `src/numerics/leapfrog.jl`; `_uv`, `_advect!`, `_laplace!` in `src/numerics/stencils.jl`; `_parts`, `_forward` in `src/simulation/model.jl`
- **Docs:** `docs/src/API_published.md`

## Summary

`src/` and `ext/` shrank from 11 970 to 11 305 lines with no change in results.
Duplicated kernels and launchers were merged, pre-release restart compatibility was dropped, the values of the reproduction aliases moved to one docs page, and design rationale moved from comments to devnotes.
The U and V momentum launchers are one function each, dispatched on the C-grid location of the `Var` (`_uv(m, ::Var{Face,Center})`).

## Evidence

- Bit-identical to the parent commit in 28 scenarios: every `Params` alias, v1 walls, Float64 and Float32, smooth kinks with frazil, past closures and tracer bounds, the other entrainment, melt and convection schemes, both viscosities and momentum advections, both tracer and momentum forms, the three gap conditions, per-basin, 2D and ramped forcings, adaptive dt, NetCDF output and a restart.
- A forwarded `Model` property (`m.T_ice_farfield`) still compiles to two `getfield`s, inferred and allocation-free.

| file | lines before | after |
|---|---|---|
| `src/bcs/forcing.jl` | 615 | 504 |
| `src/numerics/leapfrog.jl` | 989 | 865 |
| `src/numerics/stencils.jl` | 925 | 776 |
| `src/simulation/model.jl` | 488 | 411 |
| `src/simulation/presets.jl` | 906 | 729 |

## Caveats

- The U and V kernels themselves stay separate.
  One kernel per operator would change the summation order of one component (no longer bit-identical, so every reproduction pin moves), the wrap forms Reactant raises are fragile (`reactant-kernel-forms.md`), and the paired kernels map line by line to the reference Fortran.
- Kernel constants were not bundled into NamedTuples: it changes the kernel arguments, and so the Reactant lowering and CPU cost, for little gain.
- Restarts written before the D → H rename, or without their grid, no longer load.
- Not done, by decision: removing the experimental and dead-end schemes (they stay opt-in), and adopting a formatter style (about 500 lines from packing arguments with `YASStyle`, margin 120).
