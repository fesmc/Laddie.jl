# Stability limits of leapfrog with the Robert–Asselin filter

- **Type:** open question
- **Status:** leapfrog + RA (ν = 0.8) with `AdaptiveDt`; reference runs at CFL target 0.4
- **Code:** `src/timestepping.jl`; `leapfrog_step!` in `src/numerics.jl`
- **Docs:** `docs/src/numerics/index.md`

## Question

How large can dt get, and is there a computational-mode instability left that the flux forms do not remove?

## What we know

Largest stable ω·Δt of leapfrog + RA on dq/dt = iωq:

| ν | 0 | 0.1 | 0.2 | 0.5 | 0.8 |
|---|---|---|---|---|---|
| ω·Δt | 1.00 | 0.95 | 0.90 | 0.77 | 0.655 |

- At ν = 0.8, the limit in Laddie's CFL number is 0.655 where advection dominates, and 0.655/√2 ≈ 0.46 where gravity waves dominate (C-grid, Δx = Δy).
  CFL target 0.4 sits just below both.
- On damping terms (y' = −λy) the limit is λΔt ≈ 0.575. The flux forms move it to > 20 (`improvements/flux-form-momentum.md`).
- At CFL 0.4 (pan-Antarctic 1 km), the 2Δt mode raised the logged entrainment from 15.4 to 19.0 Sv, in a handful of cells with H < 5 m. Melt moved by only +0.18 %.
  This was measured before `FluxMomentumForm`.
- Under warm (Mathiot) forcing, the adjoint still grows, at thick (150–370 m), slow (≤ 0.13 m/s) cells, not thin ones.
  The usable gradient window is ~1 d at CFL 0.4 and ~5 d at CFL 0.2. `PastClosures` does not help.
  The 1 m/s speed cap made it worse (one Filchner cell held at the cap).

## Candidate explanations

- The convective switch in freshened warm water (S ≈ 32.4).
- Another damping term still taken at the centred level.
- A baroclinic mode of the leapfrog step.

## Next test

- Every-step primal probe at the warm hot cells: does anything alternate every step?
- A warm window sweep at CFL 0.1: does the blow-up move in proportion to dt?
- ν = 0.2 with CFL target ~0.6, over 200 d (about 50 % more dt, untested).
- A scheme without a computational mode (v2's FB-RK3), if the probe shows a 2Δt alternation.
