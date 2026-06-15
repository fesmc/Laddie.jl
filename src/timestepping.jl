
# ============================================================================
# Time steppers — control how `dt` evolves during a run.  Threaded through
# `Params` as the `TS` type parameter (`Params(; tstep = ...)`); the runtime
# step lives in `IOState` and is read as `m.dt`.
# ============================================================================

abstract type AbstractTimeStepper end

"""
$(TYPEDSIGNATURES)

Constant time step (the default): `dt` stays at `Params.dt0` for the whole
run.  Reproduces LADDIE v1.x behaviour exactly, so the default integration is
bit-for-bit unchanged.

Select via `Params(; tstep = FixedDt())` (the default).
"""
struct FixedDt <: AbstractTimeStepper end

"""
$(TYPEDSIGNATURES)

Predictive CFL-limited adaptive time stepping.  Every `ncheck` steps the
advective–gravity-wave CFL number is measured and `dt` is nudged toward the
step that would hit `cfl_target`:

```
factor = (cfl_target / cfl)^q
```

Because the CFL number is linear in `dt`, `q = 1` lands exactly on the target;
`q < 1` under-corrects (smoother, less oscillatory), `q > 1` over-corrects.
The adjustment is asymmetric and predictive — it never rejects a step:

- above target (`cfl > cfl_target`) → shrink immediately by `factor`;
- well below target (`cfl < grow_hyst · cfl_target`) → grow by `factor`, but
  no more than `max_growth` per adjustment;
- in between → hold (hysteresis band keeps changes rare, which matters because
  each change re-bootstraps the leapfrog).

`dt` is always clamped to `[dtmin, dtmax]`.

Select via `Params(; tstep = AdaptiveDt())`; override fields as needed, e.g.
`AdaptiveDt(; cfl_target = 0.4, ncheck = 10)`.
"""
struct AdaptiveDt{FT} <: AbstractTimeStepper
    cfl_target :: FT     # CFL setpoint the controller aims for
    q          :: FT     # response exponent (1 = CFL-exact, <1 damps)
    max_growth :: FT     # max dt increase per adjustment (e.g. 1.1 = +10%)
    grow_hyst  :: FT     # grow only when cfl < grow_hyst · cfl_target
    ncheck     :: Int    # steps between CFL checks
    dtmin      :: FT     # lower clamp on dt (s)
    dtmax      :: FT     # upper clamp on dt (s)
end

"""
$(TYPEDSIGNATURES)

Keyword constructor for [`AdaptiveDt`](@ref); all scalar fields are converted
to `FT`.  Defaults are conservative ISOMIP+-scale values.
"""
function AdaptiveDt(;
    FT         = Float64,
    cfl_target = 0.5,
    q          = 1.0,
    max_growth = 1.1,
    grow_hyst  = 0.8,
    ncheck     = 20,
    dtmin      = 1.0,
    dtmax      = 1000.0,
)
    AdaptiveDt{FT}(FT(cfl_target), FT(q), FT(max_growth), FT(grow_hyst),
                   Int(ncheck), FT(dtmin), FT(dtmax))
end

# (Float-type promotion of the stepper to match Params{FT} is handled generically
# by `_promote_param` in params.jl, alongside the other parameterization objects.)

# Proposed dt from the predictive CFL controller.  Returns the clamped dt; the
# input dt unchanged when inside the hysteresis band or when there is no usable
# CFL signal.  `allow_grow = false` (startup rescue) only ever shrinks, so a
# small initial CFL (e.g. zero velocity at t = 0) can never inflate dt0.
function _controller_dt(ts::AdaptiveDt, dt, cfl; allow_grow::Bool)
    target = Float64(ts.cfl_target)
    dt = Float64(dt); dtmin = Float64(ts.dtmin); dtmax = Float64(ts.dtmax)
    (cfl > 0 && isfinite(cfl)) || return clamp(dt, dtmin, dtmax)
    if cfl > target                                            # above target → shrink now
        dtn = dt * (target / cfl)^Float64(ts.q)
    elseif allow_grow && cfl < Float64(ts.grow_hyst) * target  # well below → grow slowly
        dtn = dt * min((target / cfl)^Float64(ts.q), Float64(ts.max_growth))
    else                                                       # hysteresis band → hold
        return clamp(dt, dtmin, dtmax)
    end
    return clamp(dtn, dtmin, dtmax)
end

# Apply a controller decision: when dt actually changes, set it, re-bootstrap
# the leapfrog at the new dt, and log the change.  Returns whether dt changed.
function _apply_dt!(m, ts::AdaptiveDt, cfl; allow_grow::Bool)
    dt_old = m.dt
    dt_new = m.FT(_controller_dt(ts, dt_old, cfl; allow_grow))
    dt_new == dt_old && return false
    m.dt = dt_new
    _rebootstrap_leapfrog!(m)
    _log_dt_change!(m, dt_old, dt_new, cfl)
    return true
end

# Whether the stepper adapts dt (controls whether `run!` computes the CFL on
# the non-verbose path).
_adapts(::AbstractTimeStepper) = false
_adapts(::AdaptiveDt) = true

# Cadence of the shared blow-up/CFL/controller sync point.  FixedDt keeps the
# historical ~5 % blow-up cadence (unchanged).  AdaptiveDt checks at least every
# `ncheck` steps, and more often on short runs so a too-large dt0 is caught
# before it can blow up.
_check_interval(::AbstractTimeStepper, nt) = max(1, nt ÷ 20)
_check_interval(ts::AdaptiveDt, nt) = clamp(ts.ncheck, 1, max(1, nt ÷ 20))

# In-loop hook called at the check cadence from `run!`, with the CFL already
# measured at the shared sync point.  FixedDt is a pure no-op (the default
# integration is unchanged).  AdaptiveDt adjusts dt predictively for the
# upcoming steps — it never rejects a step, so there is no rollback.
_maybe_adapt_dt!(m, ::FixedDt, cfl) = nothing
_maybe_adapt_dt!(m, ts::AdaptiveDt, cfl) =
    (_apply_dt!(m, ts, cfl; allow_grow = true); nothing)

# Pre-loop hook: rescue a too-large dt0 before the first step (shrink-only).
# Uses the worst-case CFL (advection at the velocity cap), because at t = 0 the
# flow is ~stationary and the actual CFL underestimates what the spinning-up
# flow will reach.
_init_adaptive_dt!(m, ::FixedDt) = nothing
function _init_adaptive_dt!(m, ts::AdaptiveDt)
    _apply_dt!(m, ts, _cfl_worstcase(m); allow_grow = false)
    return
end
