# Defaults, names and open scheme families for v0.1

- **Type:** improvement
- **Status:** default in main since 2026-11-18
- **Commits:** 18142ef (`tracer_bounds` slot), a87bdd0 (`ice_slope` slot, gradient renames), 9acc89d (reproduction pins), 6fdbc74 (default flip), e796099 (net melt and refreezing), 8fc311e (`T_ice_farfield`, `OceanForcing2D` export), 9444cd2 (open families), the commit of this note (`meltstats` under Reactant, `OceanForcing2D` traced)
- **Code:** `Params` (`src/simulation/params.jl`); `AbstractTracerBounds` (`src/numerics/tracer_bounds.jl`); `CentredGradient`, `MaskedGradient` (`src/simulation/grid.jl`); `meltstats` (`src/utils/diagnostics.jl`); the time series in `src/utils/io.jl`; `PrescribedIceForcing` (`src/bcs/forcing.jl`); `Cache` initialisers (`src/simulation/cache.jl`); the aliases in `src/simulation/presets.jl`
- **Docs:** `docs/src/versions.md` (defaults and their rationale), `docs/src/extending.md`, `docs/src/numerics/tracer_bounds.md`

## Summary

Before the registration, the defaults became those of the tuned pan-Antarctic runs ([laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin)): `RelaxToAmbient(3600)`, `MeltOnlyConduction`, `FluxMomentumForm` with `PastClosures`, and no tracer bounds.
Tracer bounds and the ice-slope stencil became `Params` slots, `T_ice_base` became `T_ice_farfield`, and the melt diagnostics report the net basal melt (melt − frazil) and the refreezing.
Every scheme family is open: a type defined outside Laddie.jl runs like a built-in one.
The reproduction aliases pin the old settings, so their results are unchanged except where the v1 aliases now select v1's slope stencil.

## Evidence

| change | old | new |
|---|---|---|
| convection | `ResetToAmbient(0.005)` | `RelaxToAmbient(3600)` (struct default 10000 s → 3600 s) |
| conduction | `HollandJenkinsConduction` | `MeltOnlyConduction` (no longer experimental) |
| momentum form / closures | `AdvectiveMomentumForm` / `PresentClosures` | `FluxMomentumForm` / `PastClosures` |
| tracer bounds | `T_min … S_max` fields, −5…5 °C, 32…36 psu | `tracer_bounds = NoTracerBounds()` |
| ice slope | `Model(grid; gradient = JlGradient())` | `Params(; ice_slope = MaskedGradient())` |
| `meltstats` | `(max_meltrate, mean_meltrate, max_speed, total_melt)` | `(max_melt, mean_melt, max_speed, total_melt, refreezing)`, net of frazil |
| series in `output.nc` | `ts_m`, `ts_m_gt` | `ts_mean_melt`, `ts_total_melt`, `ts_refreezing` |

Reproduction pins (`REPRODUCTION_MEAN_MELT`, ISOMIP+ 40×20, 200 steps of 210 s): the v2 aliases are unchanged throughout.
The v1 aliases move once, on purpose, when they select `CentredGradient` themselves:

| alias | before (m/yr) | after (m/yr) |
|---|---|---|
| `ParamsCrossonDotsonv1`, `ParamsPIGv1` | 3.667 | 41.942 |
| `ParamsFRISv1` | 26.957 | 56.338 |
| `ParamsASEv1` | 3.665 | 42.721 |

On that cavity, `z_draft = 0` at the grounded cells, so the centred stencil has |∂z_b/∂x| = 0.179 in 40 of 600 active cells, against 0.00118 everywhere under `MaskedGradient`.
The 1-day comparison with Python LADDIE v1 passes unchanged (13/13).

Tests: full suite after each step; ForwardDiff against central differences for the former defaults and for the new ones where they act (`RelaxToAmbient` on a convecting start, `MeltOnlyConduction` on a refreezing one); the budget `total_melt + refreezing = Σ max(melt, 0)` on a supercooled start with `RelaxedFrazilIce`; a user-defined melting scheme, convection scheme and ocean forcing run.
Reactant: `OceanForcing2D` runs on the Reactant backend with a melt field identical to the native run (ISOMIP+ 20×10, 0.05 d), and joins the scheme coverage.
Its traced AD needed the profile arrays' element type untied from the scalar `FT` (`dz`, `z0` become traced numbers, the profiles do not).

## Parts that were needed

- The Python v1 verification, the docs examples that reproduce published runs (`crosson-dotson.jl`, `lambert-ase.jl`, `jesse-gaps.jl`, `isomip.jl`) and the aliases now spell out the settings whose defaults changed.
- `max_melt` is taken over ice cells only, so it is negative where every ice cell refreezes; before, the zeros outside the ice made it 0.
- A user forcing's fields are forwarded by the model, so their names must not clash with other model fields (`T`, `S`, …); the model refuses to build otherwise.
- `meltstats(sim)` reduces the model the log line reads (`_diag_model`): on the Reactant backend a CPU copy, since the speed kernel did not compile on device arrays and eager reductions there compile one program per call.

## Caveats

- Neither LADDIE v1 nor v2 bounds T or S, so the aliases' `TracerBounds()` pin keeps Laddie.jl's earlier behaviour, not the published code's.
- Most extension points of the open families are internal, underscore-prefixed functions (`_compute_entrainment!`, `_eos_coefficients`, …); `docs/src/extending.md` lists them.
