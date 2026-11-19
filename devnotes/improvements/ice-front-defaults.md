# Ice-front defaults: outflow-only valve and zero-gradient pressure

- **Type:** improvement
- **Status:** default in main since 2026-11-19
- **Commits:** fba9318 (default flip, v1/v2 pins), d4a2b02 (v1 comparisons pin v1's front settings)
- **Code:** `OutflowOnlyOpenOceanBC`, `clip_front_inflow!`, `ZeroGradientFrontPressure`, `FullDepthFrontPressure`, `TruncatedDepthFrontPressure`, `front_pgf_weight`, `front_drho_weight` (`src/bcs/open_ocean.jl`)
- **Docs:** `docs/src/bcs/open_ocean.md`, `docs/src/bcs/front_pressure.md`

## Summary

At one-sided faces (the ice front, and gap edges under `SinkGapsBC`) the default drops both the thickness- and density-gradient parts of the pressure-gradient force, and lets the layer leave but never enter.
This removes the front-row melt anomaly of the former default (LADDIE v1's settings) while keeping the total melt within 0.4 %.
The v1 aliases pin `FullDepthFrontPressure`, the v2 setups pin `ZeroGradientOpenOceanBC`, so no `REPRODUCTION_MEAN_MELT` value moves.

## Evidence

Warm ISOMIP+ at 4 km, 20 days, against `FullDepthFrontPressure` + `ZeroGradientOpenOceanBC` (the former default):

| front pressure + open ocean | mean melt (m/yr) | total (Gt/yr) | max speed (m/s) | front/interior melt | H front (m) |
|---|---|---|---|---|---|
| `FullDepthFrontPressure` + `ZeroGradientOpenOceanBC` | 1.59 | 56.41 | 0.538 | 1.87 | 6.3 |
| `ZeroGradientFrontPressure` + `OutflowOnlyOpenOceanBC` | 1.58 | 56.21 | 0.340 | 0.93 | 11.1 |
| `TruncatedDepthFrontPressure` + `OutflowOnlyOpenOceanBC` | 1.59 | 56.32 | 0.429 | 1.27 | 8.4 |

The front row is the last ice cell of each row, the interior three cells upstream.
Under `FullDepthFrontPressure` the thickness-gradient term at the ice front is about 150× its interior value on the same run: the masked neighbour's `H = 0` is read as a real thickness.

## Parts that were needed

- **A clip, not a relaxation.** The valve is the τ → 0 limit of a relaxation of the inward normal velocity. A relaxation fast enough to act as a valve would be unstable, because the sinks of `AdvectiveMomentumForm` are explicit.
- **The clip acts on the new level only.** The filtered present level, which advects thickness and tracers, becomes outward-only within a few steps; an inward start (a restart from another condition) fades through the Robert–Asselin filter. `inflow_weight` of the valve is that of `NoInflowOpenOceanBC`, so the fluxes carry nothing in during that transient.
- **Order with the speed cap.** `clip_front_inflow!` runs after both momentum steps and before `clamp_velocities!`, which scales by a positive factor and so keeps the outward sign.
- **Bit-identity of v1.** `front_pgf_weight` and `front_drho_weight` gate each term by `1 + w*(tmask_stag - 2)`; `w = 0` multiplies by exactly 1, so `FullDepthFrontPressure` reproduces LADDIE v1 bit for bit.
- **A test cavity with a flat draft** needs `FullDepthFrontPressure`: nothing else drives a flow through its front from rest.

## Caveats

- LADDIE v1 keeps both terms (`integrate.py`, `-g·ip_t(Ddrho)·(Dxm1 - D)/dx` with `Dxm1 = roll(D*tmask)`).
- The LADDIE v2 code drops the thickness term at calving-front faces (`laddie_velocity.f90`, `IF (mask_cf_b .OR. mask_gl_b)`), and its comment says "Assume dH/dx and ddrho/dx = 0", but it keeps the `ddrho_amb_dx_b` term. `ZeroGradientFrontPressure` follows the description paper (Lambert et al. 2026, "zero lateral gradients are applied to all main variables"), not the v2 code.
- Under `ZeroGradientOpenOceanBC` an inflow face recirculates the front cell's own water, which thickens and cools the front row; under `NoInflowOpenOceanBC` the inward velocity carries nothing, so the front row thins.
