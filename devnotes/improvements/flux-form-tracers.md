# Flux-form tracers

- **Type:** improvement
- **Status:** default since 2026-10-27
- **Commits:** 5d38792 (feature), 9f6b8d7 (merge), 865303b (default flip)
- **Code:** `FluxTracerForm`, `AdvectiveTracerForm` in `src/advection.jl`; `_tracer_update`, `_outflow_rate!` in `src/numerics.jl`

## Summary

Step the tracer content H·T and H·S, then divide by H⁺, as LADDIE v2 does (`laddie_tracers.f90:59-88`).
The advective form steps T itself, with `−T·∂H/∂t` in the RHS.
That term stops cancelling the volume sources wherever H is clamped or the leapfrog restarts.
With the flux form, the pan-Antarctic runs need no T/S bounds.

## Evidence

Time-step error, warm ISOMIP+ 40 × 20, mean melt at day 2 (m/yr):

| dt | advective | flux |
|---|---|---|
| 210 s | 5.066 | 4.526 |
| 52.5 s | 4.635 | 4.503 |
| change | 9 % | 0.5 % |

Pan-Antarctic, 1 km, BedMachine v4, v2 physics, ISMIP6 J20 front-cell ocean, no T/S bounds ([laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin)):

| form | max layer thickness | outcome |
|---|---|---|
| advective | 10 000 m | NaN, day 0.35 |
| advective | 0.8 × water column | NaN, day 17.6 |
| flux | 10 000 m | 200 d; T −3.43…1.54 °C, S 32.99…34.79 psu |
| flux, corrective clamp | 0.8 × water column | NaN, iteration 10 |
| flux, coherent clamp | 0.8 × water column | NaN, day 94.3 (ent2 29 Sv) |
| flux, coherent clamp, implicit outflow, subgrid cap | 0.8 × water column | 200 d at CFL 0.4, dt 191 s |

## Parts that were needed

1. **Coherent clamp:** volume cut at the cap leaves at the layer's q, and volume added at the floor enters at q_a.
   `q⁺ = (HQ⁺ + max(H⁺ − H*, 0)·q_a) / max(H⁺, H*)`.
2. **Implicit outflow:** face outflow and `detr_cap` are taken at the new level.
   Without it, thin cells (water column 1.4–2.4 m) flushed at 10–100× their volume per step flip S sign (Larsen C regional run, 8 days).
3. **Subgrid cap** (in laddie-perbasin, not Laddie): the cap uses the deepest 500 m BedMachine column per 1 km cell.

## Caveats

- `AdvectiveTracerForm()` is kept for v1 reproduction and pinned in `test/verification.jl`, `docs/src/examples/isomip.jl` and `crosson-dotson.jl`.
- Re-baselined: the warm/cold melt ratio in `test/forcing.jl`, 1.072 → 1.049.
- At CFL 0.4, the 2Δt mode inflates entrainment in thin cells by 3.6 Sv of 19 Sv.
  See `flux-form-momentum.md`.
