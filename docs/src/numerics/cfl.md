# CFL number

The **diagnosed CFL** is a measurement of the current state at the current step,

```math
C = \Delta t\,\max_{\text{active cells}}\left(\frac{|u| + c}{\Delta x} + \frac{|v| + c}{\Delta y}\right),
\qquad c = \sqrt{g\,\delta\rho\,H},
```

covering advection and internal gravity waves only (there is no viscous term). It is
computed in one of two ways, `Simulation(model; cfl = ...)`:

- [`ExactCFL`](@ref), the default, takes the maximum cell by cell, at the cost of one
  kernel pass.
- [`ConservativeCFL`](@ref) adds up the separate domain maxima of ``|U|``, ``|V|`` and
  ``c``. The worst velocity and the worst wave speed rarely occur in the same cell, so it
  reads higher.

`run!` measures ``C`` at its check points: every `ncheck` steps, or more often on short
runs. It is measured with or without the [adaptive controller](time_stepping.md), and
appears in the progress display as `dt [s] / CFL` and in each dt-change log line.

Both diagnostics feed the same `cfl_target` of [`AdaptiveDt`](@ref). Since
`ConservativeCFL` reads higher, the same target gives a smaller, safer step with it than
with `ExactCFL`.

## Worst-case CFL

A flow starting from rest has ``C \approx 0``, which says nothing about the flow it will
develop. Before the first step, the **worst-case CFL** uses the speed cap
``v_\text{cut}`` in place of the velocity. The adaptive controller checks the initial
`dt` against it, which can only shrink the step, and `run!` warns when it exceeds 1,
whatever the time stepper.

## API

```@docs
AbstractCFL
ExactCFL
ConservativeCFL
```
