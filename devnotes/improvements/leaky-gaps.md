# Leaky gaps

- **Type:** improvement
- **Status:** opt-in since 2026-11-10 (`SinkGapsBC()` is the default)
- **Commits:** adf6cb0 (current form: mélange cover α); b6df8c4 (transmission form, replaced)
- **Code:** `LeakyGapsBC(; ...)` in `src/boundary_conditions.jl`, with `surface_heat_loss` Q and `ice_salinity` S_i
- **Docs:** `docs/src/physics/gaps.md`, "Leaky gaps"

## Summary

`LeakyGapsBC` keeps gaps active as `ConnectedGapsBC` does, and covers a fraction α of each gap with mélange and sea ice (default 0.9).
α scales the ice–ocean drag.
The open fraction loses (1 − α)·Q: first it cools the layer, then it forms sea ice with brine rejection.
Gap cells with δρ < 0 convect (T, S → T_a, S_a and H → H_min, on 10⁴ s), whatever the model's convection scheme.

## Evidence

ISOMIP+ channel with a 12 × 24 km gap across the boundary current; downstream melt, days 18–20 (sink 103.9, connected 164.2):

| α | Q = 0 | Q = 300 W m⁻² | Q = 600 W m⁻² |
|---|---|---|---|
| 0 (open water) | 209.6 | 183.3 | 174.2 |
| 0.5 | 184.9 | 173.1 | 160.8 |
| **0.9** | **168.0** | **165.8** | **163.7** |
| 1 (full cover) | 164.2 | 164.2 | 164.2 |

Year-200 geometries of Jesse et al. (2026), total melt over days 20–30 (Gt/yr):

| geometry | sink | α = 0 | α = 0.9 | connected |
|---|---|---|---|---|
| small gap | 38.7 | 68.6 | 68.6 | 68.6 |
| large gap | 52.0 | 65.4 | 65.6 | 65.6 |

- An enclosed gap behaves much like a connected one: within 2.5 % at the default α, and at most 2 % below connected even for open water.
  The sink gives 20–45 % less.
- No process here comes near the sink at a realistic Q. The sink remains the end-member for a gap the layer leaves through.

## Parts that were needed

- **The gap lens is physical.** The gap is an inverted basin that fills to its spill depth (layer 230–250 m thick) and is renewed in about two weeks.
  Cooling 250 m to neutral buoyancy would take ~4 months at 100 W m⁻².

## Caveats

- Q = 0 by default until a regional value is chosen. Winter polynyas reach a few hundred W m⁻².
- The earlier transmission form (b6df8c4: T → T_a, H → H_min, U, V → 0 at a rate −ln α·|u|/L) owed its "in between" response to heating the gap. It was replaced.
