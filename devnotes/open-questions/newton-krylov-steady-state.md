# Newton–Krylov steady state and sparse Jacobians

- **Type:** open question
- **Status:** not implemented (2026-11-20). Only the sparsity detection has been tried, outside the package.
- **Commits:** none
- **Code:** `integrate!` in `src/simulation/autodiff.jl`; `_step_model!`, `_bootstrap!` and `leapfrog_step!` in `src/numerics/leapfrog.jl`; `FBRK3Integrator` in `src/numerics/fbrk3.jl`; `SteadyStateSimulationEnd` in `src/simulation/simulation.jl`
- **Docs:** `docs/src/numerics/simulation_end.md` (the current steady-state criterion)

## Question

Can we reach the quasi-steady state of a cavity by Newton–Krylov iteration on the model's residual, instead of time-stepping through the spin-up?
Can SparseConnectivityTracer.jl (SCT) make the Jacobian cheap enough to precondition that iteration?

## What we know

- **Spin-up cost.**
  Today we reach a steady state by integrating in time until `SteadyStateSimulationEnd` holds (relative daily change of the mean melt < `tol`, at most 365 d by default).
  The ISOMIP example spins up for 30 d.
- **Residual.**
  The steady state is F(q) = 0, where q = (H, U, V, T, S) and F is the tendency of `leapfrog_step!`.
  Laddie has no standalone tendency function.
  One candidate (unchecked): set past = present = q, take the forward step of `_bootstrap!` over Δt, and use R(q) = (future − q)/Δt.
  `FBRK3Integrator` is a one-level map qⁿ → qⁿ⁺¹, so the fixed point Φ(q) = q of one step or of k steps (shooting) is also well defined.
- **Jacobian-free Newton–Krylov (JFNK) needs only J·v products, and both AD paths already give them.**
  ForwardDiff through a model built at `FT = Dual` (`test/autodiff.jl`), and Enzyme forward mode inside `reactant_compile` on GPU.
  An explicit Jacobian is only needed for a preconditioner or a direct sparse solve.
- **SCT works on Laddie unchanged.**
  Tried with SCT 1.2.4 in a scratch environment on ISOMIP+ warm, defaults (leapfrog, `ExactKinks`), state after 0.2 d.
  The traced map is one `_step_model!`, state → state, with the model built at `FT = eltype(x)`.
  `_step_model!` rotates the levels first (past ← present ← future), so the map seeds `future` and reads `future` back.
  Column colors come from SparseMatrixColorings, `GreedyColoringAlgorithm`.

  | grid | n (5 fields, with halo) | nonzeros per wet row (mean / max) | rows with no entries (halo, land) | column colors |
  |---|---|---|---|---|
  | 12×6 | 560 | 11.4 / 19 | 508 | 21 |
  | 24×12 | 1820 | 14.2 / 20 | 1006 | 29 |

  The number of colors is set by the stencil, not by the grid size.
  The full Jacobian of one step costs about 30 forward passes, instead of n.
  On 24×12, the largest block is ∂U/∂H (1028 nonzeros, from the pressure gradient and advection).
  T and S couple through melt and density (168 nonzeros in ∂T/∂S).
  Tracing takes 18 s the first time (compilation) and 0.02 s afterwards.
- **Only the local detector works.**
  `TracerLocalSparsityDetector` works.
  `TracerSparsityDetector` (global) fails already in `build_isomip`, which branches on values.
  The kernels also compare values (upwinding, kinks, thresholds on H), so the global detector could not trace them either.
  A local pattern is valid only near the state it was traced at, because upwind direction, kinks and clamps switch entries on and off.
- **SCT runs on CPU only.**
  Its tracers are not plain-bits types, so they cannot run on GPU or under Reactant.
  The pattern and coloring would be computed once on CPU; the compressed J·v products would then run on any backend.
- **Dependency.** SCT 1.x requires DocStringExtensions ≤ 0.9; Laddie otherwise resolves 0.10. The downgrade worked.
- **Where it does not help.** The gradient of a scalar loss gains nothing: Enzyme reverse mode is already optimal, and ForwardDiff over three parameters is already cheap.
  Over a long run the dependencies spread by about one stencil per step, so the Jacobian becomes dense.

## Candidate explanations

These are the reasons it might not work:

- **No steady state.**
  An eddying or oscillating cavity has no fixed point that represents its time mean.
  The Jesse gaps example, for instance, stays unsteady near the grounding line.
  Newton may then converge to an unstable steady state, or not at all.
- **The map is not smooth.**
  Newton needs a differentiable residual.
  `ExactKinks`, the thickness clamp, the speed cap `v_cut`, the H_min top-up and `detr_cap` all make it non-smooth.
  `SmoothKinks` handles the kinks, but the clamps must be inactive at the solution, which thin columns near H_min may prevent.
  The rule from `improvements/smooth-kinks.md` also applies: a smoothed kink at which the state sits biases it by ε/4.
- **Poor conditioning.**
  Gravity waves are fast and tracer adjustment is slow, so the spectrum of J is wide and unpreconditioned GMRES may stall.
  Two remedies: a k-step shooting map, which damps the fast modes, or a preconditioner built from the sparse Jacobian (SCT pattern and about 30 passes, then ILU).

## Next test

1. On ISOMIP+ warm 24×12, with `FBRK3Integrator` and `SmoothKinks`, spin up 30 d by time stepping as the reference.
2. From the day-2 state, run JFNK (Krylov.jl GMRES, or NonlinearSolve.jl) with ForwardDiff J·v products on the bootstrap residual.
   Compare the final residual and the mean melt with the reference, and the number of model evaluations with the steps of the spin-up.
3. If GMRES stalls, precondition it with the sparse Jacobian.
   Use the union of the local SCT patterns over several states, and refresh the Jacobian every few Newton steps.
4. Only if steps 2–3 pay off, move the J·v products to Enzyme inside `reactant_compile`, and add SCT and the solver as test or extension dependencies.
