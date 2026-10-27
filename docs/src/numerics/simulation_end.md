# Simulation end

A call to `run!` stops according to an end criterion, set once on the simulation as
`Simulation(model; stop = ...)` or per call as `run!(sim; until = ...)`. The keyword
`run!(sim; days = 30)` is shorthand for `until = FixedSimulationEnd(t_end = 30)`. Both
criteria count from the start of the `run!` call, not of the simulation.

- [`FixedSimulationEnd`](@ref), the default: run a fixed duration `t_end` (days, 30 by
  default).
- [`SteadyStateEnd`](@ref): stop once the cavity reaches a quasi-steady state, or after
  `t_end` days (365 by default), whichever comes first. The run is steady when the
  relative change in the mean melt rate over floating ice between two daily samples falls
  below `tol`,
  ```math
  \frac{|\bar m_{k+1} - \bar m_k|}{|\bar m_k|} < \mathrm{tol}.
  ```
  `tol` is relative, so it means the same thing in a warm cavity (melt of order
  100 m yr⁻¹) and a cold one (order 1 m yr⁻¹). Daily sampling is cheap, insensitive to the
  leapfrog computational mode, and independent of the run length.

```julia
run!(sim; until = SteadyStateEnd(tol = 1e-3, t_end = 365.0))
```

## API

```@docs
AbstractSimulationEnd
FixedSimulationEnd
SteadyStateEnd
```
