# Convection by relaxation to ambient

- **Type:** improvement
- **Status:** opt-in (`ResetToAmbient(0.005)` is the default); used in the pan-Antarctic reference runs since 2026-10-30, and in gap cells under `LeakyGapsBC` whatever the scheme
- **Commits:** in Laddie.jl since the first port of v1 (June 2026), as v1's `convop = 2`; gap use in adf6cb0
- **Code:** `AbstractConvectionScheme`, `ClampDensity`, `ResetToAmbient`, `RelaxToAmbient` in `src/convection.jl`; `_update_conv2!` in `src/numerics.jl`
- **Docs:** `docs/src/physics/convection.md`

## Summary

Where the layer is denser than the ambient beneath it (δρ < 0), it is statically unstable and would overturn and mix.
`RelaxToAmbient` relaxes T and S towards T_a and S_a on `convection_time` (default 10⁴ s), through the sink `−(qᵖ − q_a)·conv2` with `conv2 = [δρ < 0]·H/τ`, at the past level.
LADDIE v2 has only a buoyancy floor (`ClampDensity`). It holds an unstable layer in place with no buoyancy driving, so under warm, freshened forcing the plume collapses and melt falls.
None of the LADDIE papers describes a convection closure.

## Evidence

Pan-Antarctic 2 km, BedMachine v4, v2 physics, ISMIP6 J20 front-cell ocean, first-guess parameters ([laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin)).
Warm forcing = Mathiot NEMO's warm − present-day anomaly on J20 (about +3 K, −3 psu). Target warm response ΔM = 12 338 Gt/yr.

Under `ClampDensity(0.02)` (v2's setting):

- Cells with δρ < 0 make up 0–10 % of each shelf in the cold state, and 50–95 % in the warm state (Ronne 73 %, Cosgrove 93 %, Brunt 96 %).
  The ambient change alone does it: at George VI the ambient at the plume base becomes ~2e-3 lighter, against a plume contrast of 2.5e-4.
- At George VI, melt falls from 9.6 to 2.9 m/yr. |U| drops from 0.091 to 0.032 m/s, H grows from 12.9 to 26 m, and the layer stays at −1.33 °C with the ambient 3 K warmer.

Warm state, 200 d from a collapsed warm state, 42 shelves (Gt/yr):

| convection | melt | ΔM | ΔM RMSE | ΔM George VI / Bach / Cosgrove |
|---|---|---|---|---|
| `ClampDensity(0.02)` | 11 312 | 9 704 | 162 | −94 / −9 / +1 |
| `ResetToAmbient(0.02)` | 16 495 | 14 887 | 151 | +9 / +16 / +35 |

The front boundary condition (`NoInflow`) changes melt by +5 %, the convection scheme by +46 %.

`RelaxToAmbient` (τ 10⁴ s), cold spin-up then warm jump, 300 d each, against the clamp's cold state and its first 200 warm days:

| | `ClampDensity` (200 d warm) | `RelaxToAmbient` (300 d warm) |
|---|---|---|
| cold melt, 42 shelves | 1608 | 1682 (+5 %) |
| ΔM / target | 0.73 (collapsed) | 1.17 |
| ΔM RMSE | 185 | 149 |
| George VI ΔM (target +2.8) | −142 | −7.4 |
| Cosgrove ΔM (target 25) | −1 | +27 |

- In the joint cold + warm inversion, the two misfits now fall together.
  At the estimate the warm response is 0.68 of the target, against 0.47 under the clamp.
  The estimate carries over from 2 km to 500 m (warm response 0.70).
- τ is differentiable but not identifiable from 2-day windows (gradient ≤ 0.04), because it acts only in the few unstable cells of the cold state.

## Caveats

- It conserves neither heat nor salt: T and S are relaxed without exchanging volume. The same holds for `ResetToAmbient`.
- It has no δρ floor, so δρ stays negative in unstable cells.
  Gaspar entrainment uses `max(δρ, drho_floor)` and stays finite, but the warm dt falls (2 km: 233 s vs 372 s cold; 500 m: 32 s vs 93 s). The cause is not traced yet.
- `ResetToAmbient` fires below `d_rho_min/ρ₀`, not only below zero, so it also resets marginally stable cells.
- The ambient in the cavity is the front profile at every depth, whatever the plume has done to it. A strong halocline in that profile is what makes the layer unstable.
- Not compared on ISOMIP+ Ocean1–3 or against v2 there yet.
- Warm-scenario results from v2 with its floor may underestimate the melt response where the ambient becomes lighter than the plume. That is an inference, not tested on v2.
