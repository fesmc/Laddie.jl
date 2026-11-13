# Centred momentum advection

- **Type:** dead end
- **Status:** code in main (opt-in); upstream momentum advection is used in all reference runs
- **Commits:** none specific
- **Code:** the `momentum_advection` slot of `Params`

## Idea

Centred momentum advection was tried twice:

- as a cure for the 10 000 m cap blow-up, suspected to be a momentum-scheme instability;
- as a less diffusive alternative for the boundary currents.

## Why it is a dead end

| setup | upstream | centred |
|---|---|---|
| Larsen C regional, 1 km, v2 physics, flux form, implicit outflow, 8 d | dt ~134 s, peak melt 88 m/yr | dt ~90 s, peak melt 159–168 m/yr |
| `SimulationASEv2` against v2's stored output | mean melt +5 %, correlation 0.91 | blows up (H 153 m vs 32 m) |
| pan-Antarctic 1 km, 10 000 m cap, advective tracers, A_h 25 | NaN at day 0.35 (no T/S bounds) | ~900 cells at 10 000 m, volume flux −100 Sv by day 14 (loose T/S bounds) |

- It is more energetic: peak melt doubles and the adaptive dt drops by a third.
- It is not needed for stability: the flux forms (`improvements/flux-form-tracers.md`, `improvements/flux-form-momentum.md`) give clean runs with upstream advection.
- In the pan-Antarctic case it did not address the cause: floating patches with no ice front, which fill up until they hit the cap.
- In the 2 km corner probe, centred advection removes the 2Δt alternation, but only by changing the state (entrainment ≈ 60 m/yr instead of 5 400).

## What would reopen it

A time scheme without a computational mode (e.g. v2's FB-RK3), if a less diffusive boundary current turns out to matter for melt.
