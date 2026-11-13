# Smooth kinks

- **Type:** improvement
- **Status:** opt-in since 2026-10-28 (`ExactKinks()` is the default); used in the pan-Antarctic reference runs
- **Commits:** 7bb33da (feature)
- **Code:** `ExactKinks`, `SmoothKinks` in `src/kinks.jl` (`Params(; kinks)`); helpers `_smax0`, `_smax`, `_smin`, `_ssqrt`, `_kink_width`
- **Docs:** `docs/src/numerics/kinks.md`

## Summary

`SmoothKinks()` replaces floors, caps, upwind switches and √ at zero by a C¹ quadratic blend over ±ε, exact outside the band.
There is one ε per quantity group (H, u, uH, w, drho, T, S, log).
ε = 0 keeps a group exact, for bisecting.
It was aimed at the adjoint blow-up on long windows, and **did not cure it**: that was the leapfrog 2Δt mode (`flux-form-momentum.md`).

## Evidence

- 3-day ISOMIP+, v2 style: each group moves mean melt by < 0.1 %. ForwardDiff matches finite differences at 1e-4 steps.
- Pan-Antarctic 2 km, reverse gradient of the melt misfit over a 0.3 d window: the C_visc gradient was 2× off with exact kinks and now matches central differences.
  So the Smagorinsky √ was part of the error.
- Over a 2 d window the gradient is still wrong: growth is ×10⁵ instead of ×10⁶.
  With ε × 10 it is only 3–10× smaller per decade of ε, and J itself moves by +1 % over 2 d.
- Pan-Antarctic 2 km, 300 d from rest: `SmoothKinks()` + `PastClosures()` reproduce the run without them shelf by shelf.
  Only Ekstrom (5.9 → 9.3) and Cosgrove (11.1 → 8.2 Gt/yr) move noticeably.

## Caveats

- **Never smooth a kink the state sits on.** A blend there biases every step by ~ε/4.
  The first version smoothed the flux-form `max(H⁺ − H*, 0)` and the `detr_cap` guard, mixed ambient water in every step, and wrecked T.
  These, and the H clamp after the step, stay exact.
- √(x + ε²) − ε biased melt by −ε/2. It was replaced by a compact quadratic below ε².
