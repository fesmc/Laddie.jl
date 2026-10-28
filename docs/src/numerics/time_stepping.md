# Time stepping

## Leapfrog and Robert–Asselin filter

The prognostic variables advance with a **leapfrog** scheme. Each variable
carries three time levels — past, present, and future — updated each step as

```math
q^{n+1} = q^{n-1} + 2\,\Delta t\; \mathcal{F}(q^n, q^{n-1}),
```

where advection, pressure, Coriolis, drag, entrainment and melt are evaluated at
the present level, while the Laplacian terms act on the **lagged** (past) level —
a diffusive term evaluated at the centre of a leapfrog step is unconditionally
unstable. After the leapfrog advance, a **Robert–Asselin filter** (strength
``\nu``, `Simulation(model; nu)`, 0.8 by default) damps the computational
(``2\Delta t``) mode:

```math
q^n \;\leftarrow\; q^n + \frac{\nu}{2}\bigl(q^{n-1} + q^{n+1} - 2\,q^n\bigr).
```

The three levels are then cycled (past ← filtered present, present ← future)
before the next step. The very first step, and the first step after an adaptive
time-step change, is a forward step over ``\Delta t`` from identical levels.

The filter and the explicit Laplacian terms (``A_h``, ``K_h``) are what keep the
leapfrog computational mode in check alongside the upwind advection; weakening
either makes long runs prone to divergence.

## Step-size control

The step size is controlled by `Simulation(model; tstep = ...)`:

- [`FixedDt`](@ref), the default: the step stays at the `dt` the `Simulation` was
  built with, as in LADDIE v1.
- [`AdaptiveDt`](@ref): a CFL controller sets the step.

### Adaptive time stepping

The controller compares the **diagnosed CFL** ``C``, a measurement of the current state
defined on the [CFL number](cfl.md) page, with a **target CFL**
(`AdaptiveDt(; cfl_target)`, default 0.3), the setpoint it steers ``C`` towards. ``C`` is
linear in ``\Delta t``, so at each check it rescales the step by
``(C_\text{target}/C)^q``:

- ``C > C_\text{target}``: the step shrinks at once. With ``q = 1`` the new step
  would give exactly the target, if the flow stayed the same.
- ``C < \text{grow\_hyst}\cdot C_\text{target}``: the step grows, by no more than
  `max_growth` per check.
- In between, the step stays the same.

So in practice the diagnosed CFL does not sit *on* the target. It moves within the
band ``[\text{grow\_hyst}\cdot C_\text{target},\, C_\text{target}]``, which is
``[0.24, 0.3]`` by default. The band is there because every change of step
restarts the leapfrog with a forward step (see above), so changes should be rare.
The diagnosed CFL can also leave the band:

- **Above the target, briefly.** The controller predicts; it never rejects a step.
  A flow that speeds up between checks takes ``C`` past the target for up to
  `ncheck` steps before the next check catches it. This is why the target needs to
  be well below 1 and not treated as a hard limit. On the
  [Amundsen Sea example](../generated/lambert-ase.md),
  `cfl_target = 0.4` blows up at about day 8, while 0.3 holds.
- **Below the band, for good**, when the step is pinned at `dtmax` (or held up at
  `dtmin`, which can take ``C`` past the target).
- **Right after a change.** The CFL that is printed and logged was measured
  *before* the adjustment, i.e. at the old step. A log line
  `dt 120.0 → 100.0 s (CFL 0.36)` means the step was cut *because* ``C`` was 0.36;
  the new step should give about 0.3.

Before the first step, the controller also checks the initial `dt` against the
worst-case CFL (see [CFL number](cfl.md)), which can only shrink the step.

## API

```@docs
AbstractTimeStepper
FixedDt
AdaptiveDt
```
