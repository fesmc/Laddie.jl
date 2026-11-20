# Source layout in subfolders

- **Type:** improvement
- **Status:** in main since 2026-11-18
- **Commits:** 9624c82 (generic helpers to `utils.jl`, launches to `backend.jl`), 4c27868 (pure rename into folders), f5a2220 (`api.jl` and `build.jl` split, `simulationend.jl` merged), 0ae56e6 (`bcs/` split), f07bf2c (processes into their scheme files)
- **Code:** `src/Laddie.jl` (include order); `src/utils/`, `src/physics/`, `src/numerics/`, `src/bcs/`, `src/simulation/`
- **Docs:** the Module Structure section of `AGENTS.md`

## Summary

`src/` was flat, with 35 files, and several names said little about their contents (`api.jl`, `build.jl`, `physics.jl`, `numerics.jl`).
It is now split into five folders: `physics/`, `numerics/`, `bcs/`, `simulation/` and `utils/`.
Each process file in `physics/` holds its scheme types and also the `update_*!` and kernels that apply them; before, those lived in `physics.jl`.
Every commit only moves code: no computation changed.

## Evidence

- Full test suite: 2913/2913 pass after the helper move. After the whole restructure it passes as well, with the same count, the `REPRODUCTION_MEAN_MELT` pins and Aqua.
- Reactant suite (`test/reactant`, GPU target): every testset passes, sharding included (21/21), except "output, log and restarts". That testset errors in the same way at 9624c82, before the moves: scalar indexing of a `ConcretePJRTArray` inside ProtoBuf's `_encoded_size`.
- Docs build: every `@docs` entry and cross-reference resolves.
- Bit-identical `meltstats` against 9624c82 (before the moves), ISOMIP 40×20, Float64:

| setup | days | max melt (m/yr) | mean melt (m/yr) | total melt (Gt/yr) |
|---|---|---|---|---|
| default | 0.2 | 95.08308546204395 | 90.98429169490335 | 218.36230006776802 |
| `HollandEntrainment`, `TwoEquationMelting`, `RelaxedFrazilIce`, `ResetToAmbient` | 0.2 | 11.289483582560303 | 10.441457563638668 | 25.0594981527328 |
| `UStarGamTMelting`, `OutflowOnly`, `ConnectedGapsBC` | 0.5 | 9.959433026829918 | 9.028964800623363 | 21.66951552149607 |

## Parts that were needed

- Julia resolves names when a function is called, so a function can live with its scheme even when its file is included before `Model`.
  Only type annotations in signatures, struct field types and top-level code depend on include order.
  The `update_*!` functions take the model untyped (`m`) and dispatch on scheme types from their own file.
- `DEFAULT_PAPERS_DIR` (`src/simulation/presets.jl`) now uses `dirname(dirname(@__DIR__))`, because the file sits one folder deeper.
- `ustar!` stays in `src/physics/secondary_fields.jl`, next to `step_secondary_fields!`: the friction velocity is shared by every melt scheme and by `GasparEntrainment`.
- The order of updates in `step_secondary_fields!` is kept apart from `leapfrog.jl`, so that another time integrator can reuse it.

## Caveats

- `git log --follow` tracks a file through the rename commit 4c27868. Code that was split out of a file afterwards (from `api.jl`, `build.jl`, `boundary_conditions.jl`, `physics.jl`) shows up as new lines in its new file. Use `git log -S` or `git blame -C` to trace it back.
- The other devnotes now point to the new paths.
