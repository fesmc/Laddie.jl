# Reproduction aliases for published LADDIE runs

- **Type:** improvement
- **Status:** in main since 2026-10-28
- **Commits:** f3f5fd8 (feature), b6e25e8 (docs)
- **Code:** `Params<case>v1|v2` and `Simulation<case>v1|v2` (e.g. `ParamsASEv1`, `SimulationISOMIPv2`, `SimulationMISOMIPGapsv2`) in `src/reproductions.jl`; `test/reproductions.jl`
- **Docs:** `docs/src/examples/reproductions.jl`, `docs/src/API_published.md`

## Summary

Aliases rebuild the published LADDIE runs: v1 from Lambert et al. (2023), and v2 from Lambert et al. (2026) and Jesse et al. (2026).
Only published runs get an alias. There is none for current Laddie.jl, whose aim is one parameter set.
Where the code and the paper disagree, the alias follows the code, and its docstring warns.
v2's numerics (FB-RK3) are not reproduced.

## Evidence

Full runs against the stored reference output (mean melt, spatial correlation):

| run | Δ mean melt | correlation |
|---|---|---|
| Crosson–Dotson v1 | +0.05 % | 0.9996 |
| Pine Island v1 | +0.3 % | 0.991 |
| ASE v1, as run | −0.06 % | 0.999 |
| ASE v1, theoretical dx | +9 % | 0.85 |
| ISOMIP+ v2 | +19 % | 0.93 |
| ASE v2, upstream momentum advection | +5 % | 0.91 |

## Caveats

- **v1 dx bug:** v1 sets dx before coarsening (`geometry.py:20-35`).
  `SimulationASEv1` reproduces the run as made (dx 500 m on 1 km cells); `SimulationASEv1Theoretical` uses 1 km.
  FRIS v1 may be affected too, but this cannot be verified.
- v1 aliases use the shipped configs, not the paper's Table 2 where they differ (μ 2.5, not 0.5), plus `AdvectiveTracerForm`, `PyGradient` and the v1 walls (`wall-slip-convention.md`).
- v2 aliases take their values from the shipped `.cfg` files. There are no `SimulationAISv2` or `SimulationPIGv2`, because their inputs are not bundled.
- ASE v2 with centred momentum advection blows up (H 153 m vs 32 m): see `deadends/centred-momentum-advection.md`.
