
# ============================================================================
# Time steppers — control how `dt` evolves during a run.  Held by the
# `Simulation` (`Simulation(model; tstep = ...)`); the runtime step lives in its
# `Clock` as `sim.clock.dt`.
# ============================================================================
"""
Abstract supertype for the time-step control of a [`Simulation`](@ref).  Pass a
concrete instance as `Simulation(model; tstep = ...)`: [`FixedDt`](@ref) (the
default) or [`AdaptiveDt`](@ref).
"""
abstract type AbstractTimeStepper end

"""
$(TYPEDEF)

Constant time step (the default): `dt` stays at the value the `Simulation` was
constructed with for the whole run.  Reproduces LADDIE v1.x behaviour exactly,
so the default integration is bit-for-bit unchanged.

Select via `Simulation(model; tstep = FixedDt())` (the default).
"""
struct FixedDt <: AbstractTimeStepper end

"""
$(TYPEDEF)

Predictive CFL-limited adaptive time stepping.  Every `ncheck` steps the
advective–gravity-wave CFL number is measured and `dt` is nudged toward the
step that would hit `cfl_target`:

```
factor = (cfl_target / cfl)^q
```

Because the CFL number is linear in `dt`, `q = 1` would land exactly on the target
if the flow did not change in the meantime; `q < 1` under-corrects (smoother, less
oscillatory), `q > 1` over-corrects.
The adjustment is asymmetric and predictive — it never rejects a step:

- above target (`cfl > cfl_target`) → shrink immediately by `factor`;
- well below target (`cfl < grow_hyst * cfl_target`) → grow by `factor`, but
  no more than `max_growth` per adjustment;
- in between → hold (hysteresis band keeps changes rare, which matters because
  each change re-bootstraps the leapfrog).

Select via `Simulation(model; tstep = AdaptiveDt())`; override fields as needed, e.g.
`AdaptiveDt(; cfl_target = 0.4, ncheck = 10)`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct AdaptiveDt{FT,I} <: AbstractTimeStepper
    "CFL setpoint the controller aims for (default `0.3`)"
    cfl_target::FT = 0.3
    "response exponent: `1` is CFL-exact, `< 1` damps, `> 1` overshoots (default `1.0`)"
    q::FT = 1.0
    "largest dt increase per adjustment, e.g. `1.1` = +10 % (default `1.1`)"
    max_growth::FT = 1.1
    "grow only when `cfl < grow_hyst * cfl_target` (default `0.8`)"
    grow_hyst::FT = 0.8
    "steps between CFL checks (default `20`)"
    ncheck::I = 20
    "lower clamp on dt (s, default `1.0`)"
    dtmin::FT = 1.0
    "upper clamp on dt (s, default `1000.0`)"
    dtmax::FT = 1000.0
    AdaptiveDt{FT,I}(args...) where {FT,I} = new{FT,I}(args...)
end

# The float fields are promoted to one type, so `AdaptiveDt(; dtmax = 1000)` works.
function AdaptiveDt(cfl_target, q, max_growth, grow_hyst, ncheck, dtmin, dtmax)
    f = promote(map(float, (cfl_target, q, max_growth, grow_hyst, dtmin, dtmax))...)
    return AdaptiveDt{eltype(f),typeof(ncheck)}(f[1:4]..., ncheck, f[5:6]...)
end

# (Float-type promotion of the stepper to the model's FT is handled generically by
# `_promote_param` in params.jl, called from the `Simulation` constructor.)

# Proposed dt from the predictive CFL controller.  Returns the clamped dt; the
# input dt unchanged when inside the hysteresis band or when there is no usable
# CFL signal.  `allow_grow = false` (startup rescue) only ever shrinks, so a
# small initial CFL (e.g. zero velocity at t = 0) can never inflate dt0.
function _controller_dt(ts::AdaptiveDt, dt, cfl; allow_grow::Bool)
    target = _float64(ts.cfl_target)
    dt = _float64(dt)
    dtmin = _float64(ts.dtmin)
    dtmax = _float64(ts.dtmax)
    (cfl > 0 && isfinite(cfl)) || return clamp(dt, dtmin, dtmax)
    if cfl > target                                            # above target → shrink now
        dtn = dt * (target / cfl)^_float64(ts.q)
    elseif allow_grow && cfl < _float64(ts.grow_hyst) * target  # well below → grow slowly
        dtn = dt * min((target / cfl)^_float64(ts.q), _float64(ts.max_growth))
    else                                                       # hysteresis band → hold
        return clamp(dt, dtmin, dtmax)
    end
    return clamp(dtn, dtmin, dtmax)
end

# Apply a controller decision: when dt actually changes, set it, re-bootstrap
# the leapfrog at the new dt, and log the change.  Returns whether dt changed.
function _apply_dt!(sim, ts::AdaptiveDt, cfl; allow_grow::Bool)
    dt_old = sim.clock.dt
    dt_new = sim.model.FT(_controller_dt(ts, dt_old, cfl; allow_grow))
    dt_new == dt_old && return false
    sim.clock.dt = dt_new
    _rebootstrap_leapfrog!(sim)
    _log_dt_change!(sim, dt_old, dt_new, cfl)
    return true
end

# Whether the stepper adapts dt (controls whether `run!` computes the CFL on
# the non-verbose path).
_adapts(::AbstractTimeStepper) = false
_adapts(::AdaptiveDt) = true

# Cadence of the shared blow-up/CFL/controller sync point.  FixedDt checks every
# ~5 % of the run.  AdaptiveDt checks at least every
# `ncheck` steps, and more often on short runs so a too-large dt0 is caught
# before it can blow up.
_check_interval(::AbstractTimeStepper, nt) = max(1, nt ÷ 20)
_check_interval(ts::AdaptiveDt, nt) = clamp(ts.ncheck, 1, max(1, nt ÷ 20))

# In-loop hook called at the check cadence from `run!`, with the CFL already
# measured at the shared sync point.  FixedDt is a pure no-op (the default
# integration is unchanged).  AdaptiveDt adjusts dt predictively for the
# upcoming steps — it never rejects a step, so there is no rollback.
_maybe_adapt_dt!(sim, ::FixedDt, cfl) = nothing
_maybe_adapt_dt!(sim, ts::AdaptiveDt, cfl) =
    (_apply_dt!(sim, ts, cfl; allow_grow = true); nothing)

# Pre-loop hook: rescue a too-large dt0 before the first step (shrink-only).
# Uses the worst-case CFL (advection at the velocity cap), because at t = 0 the
# flow is ~stationary and the actual CFL underestimates what the spinning-up
# flow will reach.
_init_adaptive_dt!(sim, ::FixedDt) = nothing
function _init_adaptive_dt!(sim, ts::AdaptiveDt)
    _apply_dt!(sim, ts, _cfl_worstcase(sim); allow_grow = false)
    return
end

# ============================================================================
# CFL diagnostics — select how the in-loop CFL number is computed.
# Held by the Simulation as its `cfl` field; dispatches `_cfl_number`.
# ============================================================================
"""
Abstract supertype for how the in-loop CFL number is computed (for the progress
display and for [`AdaptiveDt`](@ref)).  Pass a concrete instance as
`Simulation(model; cfl = ...)`: [`ExactCFL`](@ref) (the default) or
[`ConservativeCFL`](@ref).
"""
abstract type AbstractCFL end

"""
$(TYPEDEF)

Conservative CFL estimate: takes `max|U|`, `max|V|`, and `max c` independently
over the active domain, then adds them.  This overestimates the true CFL (the
worst velocity and worst wave speed rarely occur at the same cell) but avoids a
per-cell kernel pass.
"""
struct ConservativeCFL <: AbstractCFL end

"""
$(TYPEDEF)

Per-cell CFL: interpolates U and V to T-points, computes the CFL at every
active cell, and returns the maximum (the default).  Tighter than
[`ConservativeCFL`](@ref) at the cost of one kernel pass.
"""
struct ExactCFL <: AbstractCFL end
