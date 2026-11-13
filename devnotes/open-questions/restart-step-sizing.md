# Volume corrections sized for a 2Δt step

- **Type:** open question
- **Status:** unchanged since v1
- **Code:** `update_entrainment!` and `_net_entrainment_kernel!` in `src/physics.jl` (`FT(2) * dt`)

## Question

`ent2` (the H_min top-up) and `detr_cap` (the cap detrainment) are sized to reach their target over a leapfrog step of 2Δt.
In a restart step (the first step and every dt change under `AdaptiveDt`) the step spans only Δt.
They then remove or add only half the excess, and the corrective clamp does the rest.
Should they take the actual step length?

## What we know

- Under the flux tracer form, the clamp is coherent (the clipped volume leaves with its own tracer), so the remainder no longer corrupts T and S.
- This was first suspected for a NaN at iteration 10 (pan-Antarctic 1 km, flux form, 0.8 cap). The NaN went away with the coherent clamp, so the sizing was not shown to matter.
- It is a v1 convention (`precompute_integration_terms!`).
- `AdaptiveDt` restarts the leapfrog at every dt change. Splitting a run into many short `run!` calls also adds restarts: +41 % mean melt in one 1 km test.

## Candidate explanations

The volume clamp still fires in every restart step, which is extra work at best. With the advective form it is a real tracer error.

## Next test

Pass the actual step length and check the Python-v1 verification. If that moves, apply the change only under `FluxTracerForm`.
Count clamp activations per restart step on a pan-Antarctic run, before and after.
