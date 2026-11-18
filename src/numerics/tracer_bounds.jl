"""
Abstract supertype for bounds on the layer temperature and salinity, applied inside the
active domain after every tracer step.  Pass a concrete instance as
`Params(; tracer_bounds = ...)`: [`TracerBounds`](@ref) (the default) or
[`NoTracerBounds`](@ref).

Bounds are a stability safeguard, not physics: LADDIE v2 has no counterpart, it
assigns T and S only from the flux-form integration.  They carry a real cost: with
the tracers pinned the melt rate stays finite, so a diverging run is hidden from the
blow-up check.  The clamp is an instantaneous adjustment; with
[`SmoothKinks`](@ref) its corners are smoothed by `ε_T` and `ε_S`.
"""
abstract type AbstractTracerBounds end

"""
$(TYPEDEF)

Leave the layer temperature and salinity unbounded: they are set by their budgets
alone, as in LADDIE v2.

Select via `Params(; tracer_bounds = NoTracerBounds())`.
"""
struct NoTracerBounds <: AbstractTracerBounds end

"""
$(TYPEDEF)

Clamp the layer temperature to `[T_min, T_max]` and the salinity to `[S_min, S_max]`
inside the active domain after every tracer step (the default).  Outside `tmask` the
tracers stay zero.  Widen the bounds for fresher settings such as Greenland fjords; an
infinite bound switches that side off.

```jldoctest
julia> Params(; tracer_bounds = TracerBounds(; T_max = 2.0, S_min = 30.0)).tracer_bounds
TracerBounds{Float64}(-5.0, 2.0, 30.0, 36.0)
```

# Fields
$(TYPEDFIELDS)
"""
struct TracerBounds{FT} <: AbstractTracerBounds
    "lower temperature bound (°C, default `-5`)"
    T_min::FT
    "upper temperature bound (°C, default `5`)"
    T_max::FT
    "lower salinity bound (psu, default `32`)"
    S_min::FT
    "upper salinity bound (psu, default `36`)"
    S_max::FT
    TracerBounds{FT}(T_min, T_max, S_min, S_max) where {FT} = new{FT}(T_min, T_max, S_min, S_max)
end

function TracerBounds(; T_min = -5.0, T_max = 5.0, S_min = 32.0, S_max = 36.0)
    T_min < T_max || throw(ArgumentError("TracerBounds: T_min = $T_min must be below T_max = $T_max"))
    S_min < S_max || throw(ArgumentError("TracerBounds: S_min = $S_min must be below S_max = $S_max"))
    bounds = float.(promote(T_min, T_max, S_min, S_max))
    return TracerBounds{eltype(bounds)}(bounds...)
end

# Rebuilt unchecked: a zero AD tangent (`Enzyme.make_zero`) must pass.
_rebuild(::Type{<:TracerBounds}, T_min, T_max, S_min, S_max) =
    TracerBounds{typeof(T_min)}(T_min, T_max, S_min, S_max)

_tracer_range(b::TracerBounds, ::Val{:T}) = (b.T_min, b.T_max)
_tracer_range(b::TracerBounds, ::Val{:S}) = (b.S_min, b.S_max)

"""
$(TYPEDSIGNATURES)

Apply the tracer bounds of `m.params` to the tracer field `q`, temperature for
`quantity = Val(:T)`, salinity for `Val(:S)`.  Only inside the domain: unmasked, the
salinity floor would raise every inactive cell from S = 0 to `S_min`, breaking the
invariant that prognostics are zero outside `tmask` (see `_clamp_thickness!`).
"""
apply_tracer_bounds!(m, q, quantity) = apply_tracer_bounds!(m, m.params.tracer_bounds, q, quantity)

apply_tracer_bounds!(m, ::NoTracerBounds, q, quantity) = nothing

function apply_tracer_bounds!(m, bounds::TracerBounds, q, quantity)
    q_min, q_max = _tracer_range(bounds, quantity)
    launch!(_clamp_tracer_kernel!, q, m.tmask, q_min, q_max, m.params.kinks, quantity)
    return nothing
end

@kernel function _clamp_tracer_kernel!(q, @Const(tmask), q_min, q_max, kinks, quantity)
    q_min, q_max = _val(q_min), _val(q_max)
    i, j = @index(Global, NTuple)
    @inbounds q[i, j] =
        ifelse(tmask[i, j] > 0, _sclamp(kinks, q[i, j], q_min, q_max, _kink_width(kinks, quantity)), q[i, j])
end
