# Time stepping

The time integration is a scheme of `Params(; integrator)`: the leapfrog with a
Robert–Asselin filter ([`LeapfrogIntegrator`](@ref), the default, as LADDIE v1) or a
forward–backward Runge–Kutta scheme ([`FBRK3Integrator`](@ref), as LADDIE v2). The step
size is a separate choice of the `Simulation`, [below](#Step-size-control).

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
``\nu``, `Params(; integrator = LeapfrogIntegrator(; nu))`, 0.8 by default) damps the
computational (``2\Delta t``) mode:

```math
q^n \;\leftarrow\; q^n + \frac{\nu}{2}\bigl(q^{n-1} + q^{n+1} - 2\,q^n\bigr).
```

The three levels are then cycled (past ← filtered present, present ← future)
before the next step. The very first step, and the first step after an adaptive
time-step change, is a forward step over ``\Delta t`` from identical levels.

The filter and the explicit Laplacian terms (``A_h``, ``K_h``) are what keep the
leapfrog computational mode in check alongside the upwind advection; weakening
either makes long runs prone to divergence.

## Forward–backward Runge–Kutta (FB-RK3)

[`FBRK3Integrator`](@ref) is the scheme of LADDIE v2
([Lambert et al., 2026](@cite lambert_one-layer_2026), after
[Lilly et al., 2023](@cite lilly_cfl_2023)). Each step takes three stages from ``q^n``,
each an explicit step with the tendencies of the previous stage:

```math
q^{n+1/3} = q^n + \tfrac{\Delta t}{3}\,\mathcal{F}(q^n),\qquad
q^{n+1/2} = q^n + \tfrac{\Delta t}{2}\,\mathcal{F}(q^{n+1/3}),\qquad
q^{n+1} = q^n + \Delta t\,\mathcal{F}(q^{n+1/2}).
```

The thickness is stepped first in each stage. The momentum equations then see it in their
pressure-gradient and Coriolis terms as a blend of the new and the old thicknesses, the
*forward–backward* part:

```math
H^* = \beta_1 H^{n+1/3} + (1-\beta_1) H^n,\quad
H^{**} = \beta_2 H^{n+1/2} + (1-\beta_2) H^n,\quad
H^{***} = \beta_3 H^{n+1} + (1-2\beta_3) H^{n+1/2} + \beta_3 H^n,
```

with ``\beta_1 = \beta_2 = 0.5`` and ``\beta_3 = 0.344``, the values of the v2
configurations. Everything else follows the stage level: advection, drag, melt and
entrainment, which are recomputed in each stage (so the stages need
`closure_level = PresentClosureLevel()`). The thickness in the tracer fluxes is the stage
thickness of the continuity equation, so a uniform tracer stays uniform (v2 uses ``H^*``
there). Viscosity and diffusion act only in the last stage, from ``q^n``, as in v2. Under the
flux forms, which the scheme requires, the sinks at the layer's own state stay at the new
level of each stage.

FB-RK3 has no computational mode, so it needs no filter, and a step starts from one level:
no bootstrap at the start or after a change of ``\Delta t``. A step costs about three
leapfrog steps.

## Step-size control

The step size is controlled by `Simulation(model; time_stepper = ...)`:

- [`FixedTimeStepper`](@ref), the default: the step stays at the `dt` the `Simulation` was
  built with, as in LADDIE v1.
- [`AdaptiveTimeStepper`](@ref): a CFL controller sets the step.

### Adaptive time stepping

The controller compares the **diagnosed CFL** ``C``, a measurement of the current state
defined on the [CFL number](cfl.md) page, with a **target CFL**
(`AdaptiveTimeStepper(; cfl_target)`, default 0.3 under the leapfrog and 0.4 under FB-RK3,
[`default_cfl_target`](@ref)), the setpoint it steers ``C`` towards. ``C`` is
linear in ``\Delta t``, so at each check it rescales the step by
``(C_\text{target}/C)^q``:

- ``C > C_\text{target}``: the step shrinks at once. With ``q = 1`` the new step
  would give exactly the target, if the flow stayed the same.
- ``C < \text{grow\_hyst}\cdot C_\text{target}``: the step grows, by no more than
  `max_growth` per check.
- In between, the step stays the same.

So in practice the diagnosed CFL does not sit *on* the target. It moves within the
band ``[\text{grow\_hyst}\cdot C_\text{target},\, C_\text{target}]``, which is
``[0.24, 0.3]`` by default under the leapfrog. The band is there because every change of step
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

!!! warning "Do not sample in many short `run!` calls"
    Each change of ``\Delta t`` re-bootstraps the leapfrog, which disturbs the fields for a
    few steps, and the check interval is at most 5 % of the steps of the `run!` call. At a
    few dozen steps per call, ``\Delta t`` is checked and usually changed nearly every step:
    sampling a 10-day mean in hourly calls on a 1 km geometry inflated the melt rate by
    41 %. Use one longer call with time-averaged output instead.

## API

```@docs
AbstractIntegrator
LeapfrogIntegrator
FBRK3Integrator
default_cfl_target
AbstractTimeStepper
FixedTimeStepper
AdaptiveTimeStepper
```
