# Ice-conduction closures

- **Type:** improvement
- **Status:** opt-in since 2026-10-27 (`HollandJenkinsConduction()` is the default); `MeltOnlyConduction()` is used in the pan-Antarctic reference runs
- **Commits:** 4ae24ba (feature), b9e8aae (`T_ice_base` documented as the ice temperature above the basal boundary layer)
- **Code:** `AbstractIceConduction`, `HollandJenkinsConduction`, `MeltOnlyConduction`, `NoConduction`, `LinearProfileConduction` (`Params(; ice_conduction)`)
- **Docs:** `docs/src/physics/ice_conduction.md`

## Summary

The heat conducted into the ice in the three-equation melt is now an exchangeable closure.
The options follow Wiskandt & Jourdain (2025): `NoConduction` (their 2A), `LinearProfileConduction` (2B) and `MeltOnlyConduction` (2C).
`HollandJenkinsConduction` is the previous behaviour.
Both melt kernels take the closure as a per-cell c_i and a fixed flux.

## Evidence

Pan-Antarctic 2 km, v2 physics, −20 °C ice base, first-guess parameters, 200 d from rest; Gt/yr at day 200 ([laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin)):

| | 42 shelves | bias / RMSE vs observed mean | basal refreezing | supercooled cells |
|---|---|---|---|---|
| `HollandJenkinsConduction` | 1757 | +665 / 40 | 4.4 | 7010 |
| `MeltOnlyConduction` | 1756 | +665 / 40 | 4.8 | 7034 |

- `MeltOnlyConduction` differs only where T ≤ T_f (0.09 % of cells). Per shelf the ratio is 0.98–1.00.
- In the same setup with the 0.8 relative cap over 300 d, the default closure changes nothing either.
- It is chosen for physical consistency (no conduction where the ice freezes on), not for its effect on melt.

## Caveats

- `HollandJenkinsConduction` refreezes less than `MeltOnlyConduction` (4.4 vs 4.8 Gt/yr above; 5–7 % in the review tests). Whether to flip the default is undecided.
- Under `MeltOnlyConduction`, L_eff jumps where T crosses T_f. `SmoothKinks` blends that jump with a smoothstep.
- `NoConduction` and `LinearProfileConduction` have not been run on the pan-Antarctic setup.
