
# ============================================================================
# Simulation-end criteria — decide when `run!` stops.  Passed as the `until`
# keyword to `run!` and dispatched in the time loop.  The `days` keyword is the
# shorthand for `until = FixedSimulationEnd(t_end = days)`.
# ============================================================================

abstract type AbstractSimulationEnd end

"""
$(TYPEDSIGNATURES)

Stop after a fixed simulated duration `t_end` (days, the duration of this `run!`
call).  This is the default and exactly reproduces `run!(m; days = t_end)`.

Select via `run!(m; until = FixedSimulationEnd(t_end = 30.0))`.
"""
Base.@kwdef struct FixedSimulationEnd <: AbstractSimulationEnd
    t_end::Float64 = 30.0
end

"""
$(TYPEDSIGNATURES)

Stop early once the cavity reaches a quasi-steady state, or after `t_end` days,
whichever comes first.  Steadiness is detected when the **relative** change in
the domain-mean basal melt rate (over floating ice) between successive
diagnostic samples falls below `tol`:

```math
|\\bar{m}_{k+1} - \\bar{m}_k| / |\\bar{m}_k| < \\mathrm{tol}
```

`tol` is dimensionless, so it means the same thing in a warm cavity (melt of
order 100 m yr⁻¹) and a cold one (order 1 m yr⁻¹).  Samples are taken at the
diagnostic check cadence (not every step), which is cheap and robust to the
leapfrog computational mode.  `t_end` is a safety cap on the total duration.

Select via `run!(m; until = SteadyStateEnd(tol = 1e-3, t_end = 365.0))`.
"""
Base.@kwdef struct SteadyStateEnd <: AbstractSimulationEnd
    tol::Float64 = 1e-3
    t_end::Float64 = 365.0
end

# Hard time cap in seconds (both criteria carry one).
_end_seconds(e::AbstractSimulationEnd, spd) = e.t_end * spd

# Whether `run!` must sample the mean melt rate at each diagnostic check.
_needs_melt_sample(::AbstractSimulationEnd) = false
_needs_melt_sample(::SteadyStateEnd) = true

# Early-stop test from successive domain-mean melt samples (m yr⁻¹).
# FixedSimulationEnd never stops early; SteadyStateEnd stops once the relative
# change drops below tol (the first sample, with no predecessor, never stops).
_steady_reached(::AbstractSimulationEnd, mean_new, mean_prev) = false
function _steady_reached(e::SteadyStateEnd, mean_new, mean_prev)
    (isfinite(mean_prev) && mean_prev != 0) || return false
    return abs(mean_new - mean_prev) / abs(mean_prev) < e.tol
end
