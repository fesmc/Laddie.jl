
# ============================================================================
# Simulation-end criteria — decide when `run!` stops.  Set as the `stop` keyword
# of `Simulation`, or per call as the `until` keyword of `run!`, and dispatched in
# the time loop.  The `days` keyword is the shorthand for
# `until = FixedSimulationEnd(t_end = days)`.
# ============================================================================

"""
Abstract supertype for the stopping criterion of `run!`.  Pass a concrete
instance as `Simulation(model; stop = ...)` or `run!(sim; until = ...)`:
[`FixedSimulationEnd`](@ref) (run a set
duration) or [`SteadyStateEnd`](@ref) (stop once the solution settles).
"""
abstract type AbstractSimulationEnd end

"""
$(TYPEDEF)

Stop after a fixed simulated duration `t_end` (days, the duration of this `run!`
call).  This is the default and exactly reproduces `run!(sim; days = t_end)`.

Select via `Simulation(model; stop = FixedSimulationEnd(t_end = 30.0))` or
`run!(sim; until = FixedSimulationEnd(t_end = 30.0))`.

# Fields
$(TYPEDFIELDS)
"""
Base.@kwdef struct FixedSimulationEnd{T} <: AbstractSimulationEnd
    "duration of the `run!` call (days, default `30`)"
    t_end::T = 30.0
    FixedSimulationEnd{T}(t_end) where {T} = new{T}(t_end)
end

# Positional (and keyword) construction converts to floating point, as the
# former `Float64` field did: `FixedSimulationEnd(t_end = 30)` holds 30.0.
FixedSimulationEnd(t_end) = (t = float(t_end); FixedSimulationEnd{typeof(t)}(t))

"""
$(TYPEDEF)

Stop early once the cavity reaches a quasi-steady state, or after `t_end` days,
whichever comes first.  Steadiness is detected when the **relative** change in
the domain-mean basal melt rate (over floating ice) between successive
diagnostic samples falls below `tol`:

```math
|\\bar{m}_{k+1} - \\bar{m}_k| / |\\bar{m}_k| < \\mathrm{tol}
```

`tol` is dimensionless, so it means the same thing in a warm cavity (melt of
order 100 m yr⁻¹) and a cold one (order 1 m yr⁻¹).  Samples are taken once per
simulated day (not every step), which is cheap, robust to the leapfrog
computational mode, and independent of the run length.  `t_end` is a safety cap on
the duration of the `run!` call.

Select via `run!(sim; until = SteadyStateEnd(tol = 1e-3, t_end = 365.0))`.

# Fields
$(TYPEDFIELDS)
"""
Base.@kwdef struct SteadyStateEnd{T} <: AbstractSimulationEnd
    "relative change in the mean melt rate between daily samples below which the run stops (default `1e-3`)"
    tol::T = 1e-3
    "cap on the duration of the `run!` call (days, default `365`)"
    t_end::T = 365.0
    SteadyStateEnd{T}(tol, t_end) where {T} = new{T}(tol, t_end)
end

SteadyStateEnd(tol, t_end) =
    (p = promote(float(tol), float(t_end)); SteadyStateEnd{eltype(p)}(p...))

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
