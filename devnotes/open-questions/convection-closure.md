# Convective instability: which closure, and what it costs

- **Type:** open question
- **Status:** `RelaxToAmbientConvection(3600)` is the default since 2026-11-18 (6fdbc74); `ClampDensityConvection` is LADDIE v2's floor, `ResetToAmbientConvection` LADDIE v1's reset
- **Code:** `src/physics/convection.jl`; `convective_relaxation!` in `src/numerics/leapfrog.jl`
- **Docs:** `docs/src/physics/convection.md`

## Question

A layer denser than the ambient at its base (δρ < 0) is statically unstable. No LADDIE paper describes a closure for it.
Relaxation to ambient fixes the collapse of the v2 floor under warm, freshened forcing (`improvements/relax-to-ambient.md`), but it conserves neither heat nor salt and costs time step.
Is it the right closure, and what time scale should it have?

## What we know

From [laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin), pan-Antarctic 2 km to 500 m, v2 physics, `FluxMomentumForm`; warm forcing is Mathiot NEMO's warm − present-day change on J20:

- The floor rarely acts in present-day runs (0–10 % of cells), so its `d_rho_min` was never tuned.
  Under the warm forcing 50–95 % of shelf cells reach δρ < 0: the ambient at the plume base (the front profile at the cell's depth) becomes lighter than a plume formed from deeper water (George VI: ~2e-3 against a plume contrast of 2.5e-4).
- Under the floor, melt falls on exactly the shelves with the least warming at the draft (George VI, Bach, Cosgrove), under a slow, thick (26 m), cold layer.
- Reset and relaxation restore the response. From the collapsed state, `ResetToAmbient(0.02)` raises the warm melt by 46 %. From a cold spin-up and a warm jump, `RelaxToAmbient(10⁴ s)` gives George VI ΔM −7 against a target of +3, and a total ΔM 1.17 × the target; the cold melt moves by +5 %.
- Relaxation has no δρ floor: δρ stays negative in unstable cells. Gaspar entrainment floors it (`drho_floor`), but the time step falls: 233 s against 372 s cold at 2 km, 32 s against 93 s at 500 m. Not traced to a term.
- `convection_time` is differentiable: dJ/dlog τ ≤ 0.04 at the first guess, so a 2-day window does not constrain it. The joint cold + warm window sweep is clean to 2 d at CFL 0.2 and blows up at 5 d, as under the floor (`leapfrog-stability.md`).
- Neither closure conserves heat or salt. The reset also fires on marginally stable cells (below `d_rho_min/ρ₀`, not only below zero) and sets S so the cell has exactly that contrast.
- With `OceanForcing2D` or `PerBasinOceanForcing1D`, the ambient at depth is the front profile everywhere in the cavity, whatever the plume has done; a strong halocline in that profile is what makes the layer unstable.

## Candidate explanations

- The time-step drop under relaxation: negative δρ entering the CFL through the gravity-wave speed √(g H δρ), or a stiff relaxation rate where H is large.

## Next test

- Compare the three closures on ISOMIP+ Ocean1–3 (cold and warm), with LADDIE v2.
- Trace the warm time step under relaxation to the term that limits it.
- A conserving closure: mix the layer with the ambient volume needed to reach neutral buoyancy, instead of overwriting its T and S.
- A cold like-for-like comparison: `ResetToAmbient(0.02)` against `ClampDensity(d)`, d = 0.005 and 0.08.
