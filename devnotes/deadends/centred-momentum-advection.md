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
| same, `FluxMomentumForm`, 20 d, FB-RK3 at CFL 0.4 (leapfrog at 0.3) | mean melt +5.1 %, correlation 0.916 (both) | mean melt +62.5 % (leapfrog +61.6 %), correlation 0.45, mean H 154 m (v2 31.7 m) |
| pan-Antarctic 1 km, 10 000 m cap, advective tracers, A_h 25 | NaN at day 0.35 (no T/S bounds) | ~900 cells at 10 000 m, volume flux −100 Sv by day 14 (loose T/S bounds) |

- It is more energetic: peak melt doubles and the adaptive dt drops by a third.
- It is not needed for stability: the flux forms (`improvements/flux-form-tracers.md`, `improvements/flux-form-momentum.md`) give clean runs with upstream advection.
- In the pan-Antarctic case it did not address the cause: floating patches with no ice front, which fill up until they hit the cap.
- In the 2 km corner probe, centred advection removes the 2Δt alternation, but only by changing the state (entrainment ≈ 60 m/yr instead of 5 400).
- It is not the leapfrog's computational mode: under FB-RK3, which has none (`improvements/fbrk3-integrator.md`), the Amundsen Sea run fails the same way.

## What would reopen it

A time scheme without a computational mode was tested in 2026-11 (FB-RK3, Amundsen Sea) and does not reopen it.
What would: a diagnosis of why centred advection thickens the layer 4–5× on the Amundsen Sea under both schemes (for example a wall or ice-front treatment that the donor-cell fluxes get right by construction), and evidence that a less diffusive boundary current matters for melt.
