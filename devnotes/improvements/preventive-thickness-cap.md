# Preventive layer-thickness cap

- **Type:** improvement
- **Status:** in main since 2026-10-23; acts whenever an `AbstractMaxLayerThickness` other than `NoMaxLayerThickness()` (the default) is set
- **Commits:** 45b4d3e (feature)
- **Code:** `detr_cap` in `Cache`, computed in `_net_entrainment_kernel!` (`src/physics.jl`); `_clamp_thickness!` kept as a safety net (`src/numerics.jl`)
- **Docs:** `docs/src/numerics/max_layer_thickness.md`

## Summary

LADDIE v1 caps H correctively: it clips H after the step and keeps the tracer content.
At the cap, the inputs (advection, entrainment at T_a, meltwater at T_b) are then added with no matching volume change.
T drifts to −∞ and S to +∞, which T/S bounds then hide.
The preventive cap detrains the excess over `max(cap, H_min)` before the step, at the layer's own T, S and velocity, as LADDIE v2 does (`laddie_thickness.f90:105-127`).

## Evidence

Pan-Antarctic 1 km, BedMachine v4, v2 physics, ISMIP6 J20 front-cell ocean, advective tracer form, relative cap 0.8 × water column.

Corrective clamp:

| T/S bounds | outcome |
|---|---|
| [−5, 5] °C, [32, 36] psu | stable over 200 d; at day 200, 60 % of the cells at T = −5 or S = 36 are at the cap |
| none | 15 351 cells beyond the bounds within 0.05 d (T −28 °C, S 463 psu); 84 % sit at their H limit, 10 % next to one |

Preventive cap, no bounds:

- About 19 000 cells sit at the cap with ordinary T and S.
- Only 15–30 cells leave [−5, 5] °C / [32, 36] psu.
  All of them are at H = H_min = 1 m, where the cap (0–3 m) cannot act.
- NaN at day 17.6, from those thin water columns.
  That was fixed by the flux form with implicit outflow (`flux-form-tracers.md`).

## Caveats

- The detrainment is sized for a 2Δt step: see `open-questions/restart-step-sizing.md`.
- Where the cap is below H_min, H_min wins: the upper limit is `max(cap, H_min)`.
- The relative cap that this note made usable was later dropped for reference runs: see `deadends/relative-thickness-cap.md`.
