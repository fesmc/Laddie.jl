# Nonlinear equation of state

- **Type:** improvement
- **Status:** opt-in since 2026-10-27 (`LinearEOS()` is the default); used in the pan-Antarctic reference runs
- **Commits:** dc073f5 (feature)
- **Code:** `AbstractEquationOfState`, `LinearEOS`, `RoquetEOS` in `src/eos.jl` (`Params(; equation_of_state)`)
- **Docs:** `docs/src/physics/equation_of_state.md`

## Summary

`RoquetEOS` is the simplified polynomial of Roquet et al. (2015, Eq. 17), with cabbeling and thermobaricity.
All density contrasts share one kernel formula, β·ΔS − (a₀ + a_T·T̄ + a_Z·Z)·ΔT, whose scalar coefficients are chosen at the launch site.
Z is the layer mid-depth.
`LinearEOS` reproduces the previous contrast bit for bit.

## Evidence

Pan-Antarctic 2 km, v2 physics, −20 °C ice base, first-guess parameters, 200 d from rest; Gt/yr at day 200 ([laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin)):

| | default | `MeltOnlyConduction` | + `RoquetEOS` |
|---|---|---|---|
| 42 shelves | 1757 | 1756 | **1631** |
| bias / RMSE vs observed mean | +665 / 40 | +665 / 40 | +539 / 36 |
| bias / RMSE vs RISE | +892 / 39 | +892 / 39 | +766 / 34 |
| basal refreezing | 4.4 | 4.8 | 4.8 |

- Per shelf, melt is 0.91–1.00× the default, with a mean of 0.94 (George VI 222 → 205, Pine Island 102 → 93, Ronne 130 → 119). That is a near-uniform scale.
- The pattern, the refreezing and the supercooled area do not change.
- Thermobaricity: Roquet's α is 1.4–2× the linear α at 1000–1500 m (the FRIS grounding lines).
- Reverse check (2 km, 300 d, relative cap): `LinearEOS` adds +128 Gt/yr net, more melt almost everywhere.

## Caveats

- No EOS-only run: the −7 % is attributed to the EOS because `MeltOnlyConduction` alone changes nothing (`ice-conduction.md`).
- The salinity-dependent zero-expansion temperature (Roquet Eq. 18) is not implemented. It would need cross terms in the shared kernel formula.
