# Flux-form momentum and past closures

- **Type:** improvement
- **Status:** opt-in since 2026-10-29 (`AdvectiveMomentumForm()` and `PresentClosures()` stay the defaults); used in all pan-Antarctic reference runs
- **Commits:** cf87dcb (feature), daa756e (docs)
- **Code:** `FluxMomentumForm`, `PastClosures` (`Params(; momentum_form, closure_level)`); `_momentum_update` in `src/numerics.jl`, `_store_outflow!` in `src/stencils.jl`
- **Docs:** `docs/src/numerics/momentum_form.md`

## Summary

`FluxMomentumForm` steps H̄U and divides by H̄⁺, as LADDIE v2 does (`laddie_velocity.f90`).
The momentum sinks (dilution, detrainments, frazil, drag C_d|u| and the upstream advective outflow) are taken at the new level.
`PastClosures` computes u★, the entrainment H and the drag speed from the past level.
Together they remove the leapfrog 2Δt mode in thin, fast cells, which had made the adjoint blow up within 2–3 days.

## Diagnosis

On y' = −λy, leapfrog with a Robert–Asselin filter (ν = 0.8) is unstable above λΔt ≈ 0.575.
The momentum dilution e·U/H has λ ≈ 4e/H, because Gaspar's e ∝ u★³ ∝ |U|³.
At two wall-corner cells of the 2 km pan-Antarctic grid (H 1.07 m and 6.3 m, dt 413 s), λΔt = 0.61 and 0.57.
The second cell has a second mode: the donor-cell momentum outflow is divided by a thin H̄ (1.5 m next to 60–100 m).

Stability thresholds of a scalar model (λΔt, ν = 0.8):

| treatment | threshold |
|---|---|
| all centred (advective form) | 0.575 |
| flux-form momentum | 1.15 |
| flux-form momentum + lagged closures | > 20 |

## Evidence

Pan-Antarctic 2 km, BedMachine v4, v2 physics, `MeltOnlyConduction` + `RoquetEOS`, Float32, dt 413 s, 3 days from a day-200 restart.
Second-half means at the two corner cells:

| run | cell 1: entr, 2Δt amplitude (m/yr) | cell 2: entr, 2Δt amplitude (m/yr); U 2Δt (m/s) |
|---|---|---|
| advective form | 12 550, 21 200 | 70 400, 31 400; 0.46 |
| advective form, dt/2 | 5 446, 0.1 | 6 700, 11 900; 0.02 |
| flux form without implicit outflow | 5 440, ≤ 0.2 | still alternating, U ±0.14 |
| **flux form with implicit outflow** | **5 445, 0.1** | **1 209, 0.0; 9e-9** |
| same + past closures | 5 443, 0.0 | 1 203, 0.0; 9e-9 |

- The oscillation had inflated the mean entrainment 2–60× and the melt in those cells (26 vs 32 m/yr; 1.5 vs 10.7 m/yr).
- Gradient window sweep (dJ/dlog p, same grid): finite and smooth to 20 d; before, it blew up between 2 and 3 d.
  At 20 d: C_d_top 0.992, Pr −0.602, μ 0.167, C_d −0.378, Sc −0.306, against central differences of 1.03, −0.52, 0.24, −0.40, −0.26.
- The primal hardly moves: 1890.0 vs 1890.0 Gt/yr.

## Caveats

- `PastClosures` adds nothing visible at these cells. It is kept because its scalar threshold is much higher.
- It does not cure the adjoint growth in the warm (Mathiot) state, which sits in thick, slow cells: see `open-questions/leapfrog-stability.md`.
- Both defaults are bit-identical to 7bb33da (ISOMIP+, centred and upstream momentum advection).
