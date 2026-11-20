# Ice-front defaults: outflow-only valve and zero-gradient pressure

- **Type:** improvement
- **Status:** default in main since 2026-11-19
- **Commits:** fba9318 (default flip, v1/v2 pins), d4a2b02 (v1 comparisons pin v1's front settings)
- **Code:** `OutflowOnlyOpenOceanBC`, `clip_front_inflow!`, `ZeroGradientFrontPressure`, `ZeroExteriorFrontPressure`, `V1FrontPressure` (named `TruncatedDepthFrontPressure` and `FullDepthFrontPressure` until 2026-11-20), `front_pgf_weight`, `front_drho_weight` (`src/bcs/open_ocean.jl`)
- **Docs:** `docs/src/bcs/open_ocean.md`, `docs/src/bcs/front_pressure.md`

## Summary

At one-sided faces (the ice front, and gap edges under `SinkGapsBC`) the default drops both the thickness- and density-gradient parts of the pressure-gradient force, and lets the layer leave but never enter.
This removes the front-row melt anomaly of the former default (LADDIE v1's settings) while keeping the total melt within 0.4 %.
The v1 aliases pin `V1FrontPressure`, the v2 setups pin `ZeroGradientOpenOceanBC`, so no `REPRODUCTION_MEAN_MELT` value moves.

## Evidence

Warm ISOMIP+ at 4 km, 20 days, against `V1FrontPressure` + `ZeroGradientOpenOceanBC` (the former default):

| front pressure + open ocean | mean melt (m/yr) | total (Gt/yr) | max speed (m/s) | front/interior melt | H front (m) |
|---|---|---|---|---|---|
| `V1FrontPressure` + `ZeroGradientOpenOceanBC` | 1.59 | 56.41 | 0.538 | 1.87 | 6.3 |
| `ZeroGradientFrontPressure` + `OutflowOnlyOpenOceanBC` | 1.58 | 56.21 | 0.340 | 0.93 | 11.1 |
| `ZeroExteriorFrontPressure` + `OutflowOnlyOpenOceanBC` | 1.59 | 56.32 | 0.429 | 1.27 | 8.4 |

The front row is the last ice cell of each row, the interior three cells upstream.
Under `V1FrontPressure` the thickness-gradient term at the ice front is about 150× its interior value on the same run: the masked neighbour's `H = 0` is read as a real thickness.

Pan-Antarctic, fitted v2 setup, warm forcing, 50-day continuations at 2 and 1 km ([laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin)), `ZeroGradientFrontPressure`:

| open ocean | melt 2 km / 1 km (Gt/yr) | resolution gap | front row at 2 km: melt (m/yr), \|u\| (m/s), TF (°C), H (m) |
|---|---|---|---|
| `ZeroGradientOpenOceanBC` | 13636 / 12604 | 1032 (8 %) | 36.0, 0.218, 1.81, 99.5 |
| `NoInflowOpenOceanBC` | 16515 / 13770 | 2745 (17 %) | 71.8, 0.290, 2.40, 21.0 |
| `OutflowOnlyOpenOceanBC` | 13123 / 12466 | 657 (5 %) | 28.0, 0.127, 2.39, 24.2 |

Under the valve the front row is as fast as the second row (0.136 m/s) and barely thicker (18 m there), so the pile-up is gone; the interior 10 km from the front has H ≈ 30–43 m.
The front-pressure scheme alone changes little: the front row stays thick under all three with zero-gradient inflow.

## Parts that were needed

- **A valve, not only no inflow.** Under `ZeroGradientOpenOceanBC` an inflow face recirculates the front cell's own water, which thickens and cools the front row: it is not ambient water either.
  `NoInflowOpenOceanBC` carries nothing in but keeps the inward velocity, so the face is closed to mass and open to momentum: the inward velocity still enters momentum advection, Coriolis and friction, while the continuity equation sees no flux.
  The front row thins (21.0 m at 2 km) and melts fastest (71.8 m/yr), with the largest resolution gap of the three (17 %).
  The valve closes an inward face to both, so velocity and fluxes agree.
- **A clip, not a relaxation.** The valve is the τ → 0 limit of a relaxation of the inward normal velocity. A relaxation fast enough to act as a valve would be unstable, because the sinks of `AdvectiveMomentumForm` are explicit.
- **The clip acts on the new level only.** The filtered present level, which advects thickness and tracers, becomes outward-only within a few steps; an inward start (a restart from another condition) fades through the Robert–Asselin filter. `inflow_weight` of the valve is that of `NoInflowOpenOceanBC`, so the fluxes carry nothing in during that transient.
- **Order with the speed cap.** `clip_front_inflow!` runs after both momentum steps and before `clamp_velocities!`, which scales by a positive factor and so keeps the outward sign.
- **Bit-identity of v1.** `front_pgf_weight` and `front_drho_weight` gate each term by `1 + w*(tmask_stag - 2)`; `w = 0` multiplies by exactly 1, so `V1FrontPressure` reproduces LADDIE v1 bit for bit.
- **Zero gradient, not zero exterior.** The thickness and density terms are the split of −∇P, P = ½ g δρ H² the layer's reduced pressure. At a one-sided face, with the masked neighbour at 0, `V1FrontPressure` gives an outward tendency of 3/2 g δρ H²/Δx, `ZeroExteriorFrontPressure` ½ g δρ H²/Δx, which is exactly the one-sided difference of P = 0 one cell out, and `ZeroGradientFrontPressure` 0 (∂P/∂n = 0).
  So v1 counts the P = 0 condition 3×, because the averaged product rule is not exact there.
  The truth is a decay of P beyond the front over a length L (∂P/∂n = −P/L): zero exterior is L = Δx, zero gradient L → ∞.
  Observed outflows stay buoyant beyond the front (geostrophic at Dotson, still rising at Pine Island), so L ~ Rd ≈ 10 km > Δx = 1–4 km: zero gradient is the closer limit, and the only one whose front-row force does not grow as 1/Δx.
  A Robin condition with a physical L would be resolution-independent but adds a tuning knob; not built, since laddie-perbasin finds little difference between zero exterior and zero gradient (2026-11-20).
- **A test cavity with a flat draft** needs `V1FrontPressure`: nothing else drives a flow through its front from rest.
- **Alternatives considered.** Ambient inflow (inflow faces carry the ocean's T and S at the draft) imports warm water with no entrainment law behind it. A relaxation zone of physical width behind the front (5–10 km) is resolution-independent, but what to relax towards is arbitrary and adds a tuning knob. A Froude condition on outflow faces is a dead end (`deadends/froude-front-condition.md`).

## Caveats

- The front row still melts about 2× the interior, and the warm pan-Antarctic melt still drops by 5 % from 2 to 1 km.
  The ambient forcing is the same at every resolution. The remaining gap is a plume speed-up over the last two cells before the front, whose width scales with the cell size (`open-questions/front-resolution-dependence.md`).
- **Real inflow at the ice base is not represented**, under any of the three options.
  Ambient water does enter at ice-base depth near some fronts: summer surface water under the Ross front (Stewart et al. 2019, Nature Geoscience), and plausibly at the Ronne front in winter, where convected shelf water fills the water column.
  The layer sees that water's T and S through entrainment and the ambient profile at the draft, but not its momentum (u\*, hence melt) nor its heat carried further into the cavity than entrainment puts it.
  Representing it needs ambient inflow (see Alternatives considered), not `ZeroGradientOpenOceanBC` or `NoInflowOpenOceanBC`.
- Not mapped yet: how many front faces are inflow under the valve, and where (Ross, Ronne, Amundsen), i.e. how much of the front it changes.
  Test: map the inward faces and their volume transport under `ZeroGradientOpenOceanBC` (e.g. FRIS: Ronne Depression, Berkner Island, Filchner).
  If they line up with observed inflow and carry real transport, the valve removes something physical there; if they are scattered with little transport, they are the along-front current curling back across the front, and the clip is harmless.
  The valve's front row being as fast as the second row (0.127 vs 0.136 m/s, pan-Antarctic at 2 km) points to the latter.
- LADDIE v1 keeps both terms (`integrate.py`, `-g·ip_t(Ddrho)·(Dxm1 - D)/dx` with `Dxm1 = roll(D*tmask)`).
- The LADDIE v2 code drops the thickness term at calving-front faces (`laddie_velocity.f90`, `IF (mask_cf_b .OR. mask_gl_b)`), and its comment says "Assume dH/dx and ddrho/dx = 0", but it keeps the `ddrho_amb_dx_b` term. `ZeroGradientFrontPressure` follows the description paper (Lambert et al. 2026, "zero lateral gradients are applied to all main variables"), not the v2 code.
