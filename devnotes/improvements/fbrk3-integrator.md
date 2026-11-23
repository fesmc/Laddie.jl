# FB-RK3 time integration

- **Type:** improvement
- **Status:** opt-in since 2026-11-23 (`LeapfrogIntegrator()` stays the default); default CFL target 2.0 under FB-RK3
- **Commits:** 26f5091 (integrator as a `Params` scheme, ν moved into `LeapfrogIntegrator`), f79ed26 (feature), 4b92132 (tests; lateral terms in every stage), 3790f8d (default CFL target 2.0)
- **Code:** `AbstractIntegrator`, `LeapfrogIntegrator`, `FBRK3Integrator`, `default_cfl_target` (`src/numerics/integrator.jl`); `_step_model!`, `_fbrk3_stage!`, `_end_fbrk3_step!` (`src/numerics/fbrk3.jl`); `_step_fields!`, `_momentum_thickness!` (`src/numerics/leapfrog.jl`)
- **Docs:** `docs/src/numerics/time_stepping.md`

## Summary

`FBRK3Integrator` is LADDIE v2's three-stage forward–backward Runge–Kutta scheme (Lilly et al. 2023; Lambert et al. 2026, eqs. 16–24).
It has no computational mode, so it needs no Robert–Asselin filter and no bootstrap step, and its linear gravity-wave limit is a CFL number of 3.53 against 0.46 for the leapfrog at ν = 0.8.
On ISOMIP and the Amundsen Sea it runs clean to CFL 3–4 with melt within 0.2 %, where the leapfrog breaks at 0.8–1.0.
A step costs about three leapfrog steps, so the gain is robustness rather than forward speed; the leapfrog stays the default and is bit-identical to before.

## Evidence

Linear stability, one Fourier mode of linear gravity waves on the C-grid, the schemes as implemented (FB-RK3 with the H blends of `_momentum_thickness!`).
The CFL number is Laddie's, which sums both directions; on a square grid the fastest wave has ωΔt = √2 C.

| scheme | largest stable ωΔt | CFL limit, square grid | ωΔt per tendency evaluation |
|---|---|---|---|
| leapfrog, no filter | 1.00 | 0.71 | 1.00 |
| leapfrog + Robert–Asselin, ν = 0.8 (Laddie's default) | 0.65 | 0.46 | 0.65 |
| classic RK3 | 1.73 | 1.22 | 0.58 |
| FB-RK3, β = 0.5, 0.5, 0.344 (v2's) | 4.99 | 3.53 | 1.66 |

On elongated cells the CFL number understates the fastest wave by up to √2: FB-RK3's limit is 3.35 at an aspect ratio of 2 and 2.5 in the limit.

CFL sweeps, adaptive time stepping, melt against the same scheme at CFL 0.3:

| setup | leapfrog clean to | leapfrog fails at | FB-RK3 clean to | FB-RK3 fails at |
|---|---|---|---|---|
| ISOMIP warm, 2 km, 60 d | 0.8 | 1.0 (mean H 0.5–1.1 km, melt 50–90×) | 4.0 (melt ±0) | 5.0 (melt +31 %, max H 1.9 km) |
| Amundsen Sea, `ParamsASEv2` + `FluxMomentumForm`, 1 km, 20 d | 0.6 | 0.8 degraded (max H 1.1 km), 1.0 broken | 3.0 (melt −0.15 %) | 4.0 degraded (max H 1.5×, melt −0.7 %) |

- Runs above either limit never produce NaN: `run!` finishes and the blow-up check stays silent while the layer drifts to a nonsense state.
- The leapfrog runs survive above its linear limit (0.46): drag, entrainment and the implicit sinks damp the fastest waves.
- Steps on the Amundsen Sea (20 d): FB-RK3 at CFL 2.0 3 000 (9 000 stage evaluations), at 3.0 2 252; leapfrog at 0.6 8 215.

Accuracy, ISOMIP warm 20×10, 0.5 d from rest, max-norm relative error against Δt = 7.5 s of the same scheme:

| Δt (s) | 240 | 120 | 60 | 30 |
|---|---|---|---|---|
| leapfrog H | 4.7e-3 | 2.3e-3 | 1.1e-3 | 4.7e-4 |
| FB-RK3 H | 8.6e-4 | 4.0e-4 | 1.9e-4 | 8.1e-5 |
| leapfrog U | 8.1e-3 | 4.0e-3 | 1.9e-3 | 8.3e-4 |
| FB-RK3 U | 3.2e-3 | 1.5e-3 | 7.2e-4 | 3.1e-4 |

Both are first order; FB-RK3 is 2.5–5.5× more accurate at equal Δt.
The implicit sinks of the flux forms set most of the error (explicit sinks cut it 15–80×) but not the order, which stays about 1 with lateral terms in every stage, with no past-level terms, and with smooth kinks.
No cheap second order exists.

Cost per simulated day, one run at a time:

| setup | leapfrog 0.3 | leapfrog 0.6 | FB-RK3 0.4 | FB-RK3 2.0 |
|---|---|---|---|---|
| ISOMIP warm 2 km, CPU 16 threads | 1.60 s (5.9 ms/step) | ~0.95 s (est. from steps) | 3.55 s (16 ms/step) | 1.40 s (21 ms/step) |
| ISOMIP warm 2 km, H100 | 0.12 s | 0.09 s | 0.21 s | 0.11 s |
| Amundsen Sea 1 km, H100 | 0.28 s | 0.18 s | 0.53 s | 0.23 s |

On the GPU per-call overhead dominates at these sizes; the differences are within the 0.1 s resolution of the log.

Against LADDIE v2's stored output (last-day mean against v2's day 20):

| setup | leapfrog | FB-RK3 |
|---|---|---|
| ISOMIP+ Ocean1, `ParamsISOMIPv2` (leapfrog also with the flux form), CFL 0.3 / 0.4 and 2.0 | mean melt +18.2 %, correlation 0.932, mean H 26.0 m (v2 31.5 m) | identical to three digits |
| Amundsen Sea, `ParamsASEv2` + `FluxMomentumForm`, upstream advection | mean melt +5.1 %, correlation 0.916 | identical to three digits |

The remaining gap to v2 is not the time scheme.

## Parts that were needed

- The integrator as a `Params` slot; ν moved from `Simulation` into `LeapfrogIntegrator(; nu)`. `Simulation`, `integrate!` and `adaptive_schedule` lost their `nu` keyword (breaking for scripts that passed it).
- `step_secondary_fields!(m, span)` and `step_entrainment!(m, span)` take the time span of the step or stage; the preventive bounds (`ent2`, `detr_cap`) act over that span.
- `Hfb` in `Cache` for the blended thickness of the pressure and Coriolis terms.
- `FBRK3Integrator` requires `PresentClosureLevel` and the flux forms (checked in `Params`): the stages take the closures at the stage level, and `PastClosures` exists only to cure the leapfrog mode.
- Lateral viscosity and diffusion act in every stage, from level n. v2 adds them in the last stage only; zeroing them in the first two stages made Reactant's compiler crash (CSE on `stablehlo.slice`).
- Where we differ from v2 on purpose: the tracer flux uses the stage thickness of the continuity equation, not H\*, so a uniform tracer stays uniform (tested to 1e-12); the ambient fields and Δρ stay at the stage level instead of being recomputed on H\* (not tested against v2's variant: H\* differs from the stage thickness by O(Δt ∂H/∂t)); the secondary fields are refreshed at n+1 after the last stage for output and diagnostics.

## Caveats

- Not run on the pan-Antarctic warm window, and the adjoint has not been run at large Δt. For the linear wave part the adjoint step is the transpose of the tangent-linear step and shares its stability limit; thin cells, kinks and clamps may differ.
- Reverse mode through Reactant's traced loop needs more memory than the leapfrog: 1840 against 1302 arrays per cell (14.8 against 10.5 GB) at 1000×1000 with 20 checkpoints, ISOMIP geometry.
  A prototype adjoint of the time scheme (one Enzyme VJP per stage, prognostic-only checkpoints on the host, a fixed buffer pool) gets 377 arrays per cell (3.0 GB) at equal gradients; it is not in main.
- The speed cap (`_speed_scale_kernel!`) reads the border ring of `V.future`, which no step kernel writes; it is zero in practice, so results are unaffected, but a stage map of FB-RK3 (and a leapfrog step) formally depends on the level two stages back.
- Centred momentum advection still fails under FB-RK3 (`deadends/centred-momentum-advection.md`).
- Restarts were out of scope and are untested under FB-RK3: the restart path refreshes the secondary fields as for the leapfrog.
