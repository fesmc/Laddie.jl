"""
$(TYPEDEF)

Abstract supertype for how the model treats its kinks: the switches of `max`, `min`,
`clamp` and upwinding, and the unbounded slope of `√x` at zero.  Pass a concrete
instance as `Params(; kinks = ...)`: [`ExactKinks`](@ref) (the default) or
[`SmoothKinks`](@ref).

The kinks make the model piecewise smooth, which can stall gradient-based optimisers and
inflate long-window adjoints; see [Kinks](@ref).
"""
abstract type AbstractKinks end

"""
$(TYPEDEF)

Keep every `max`, `min`, `clamp`, upwind switch and `√` as it is (the default).  The
model is then bit-identical to one without a kink treatment.

Select via `Params(; kinks = ExactKinks())` (the default).
"""
struct ExactKinks <: AbstractKinks end

"""
$(TYPEDEF)

Smooth every kink over a band of half-width ε, one ε per physical quantity (the fields), and
leave it exact outside: `max(x, 0)` becomes the ``C^1`` blend ``(x + ε)^2 / 4ε`` for
``|x| < ε``, from which `max`, `min`, `clamp`, `|x|` and the upwind split follow (still
conservative); `√x` becomes ``(x + 3ε^2)^2 / 16ε^3`` below ``x = ε^2``. An ε of zero keeps its
group exact. Which kinks each ε smooths, and which stay exact, is on the [Kinks](@ref) page.

Select via `Params(; kinks = SmoothKinks())`, or e.g. `SmoothKinks(; H = 0.5)`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct SmoothKinks{FT} <: AbstractKinks
    "layer thickness (m, default `0.1`)"
    H::FT = 0.1
    "velocity (m s⁻¹, default `1e-3`)"
    u::FT = 1e-3
    "face mass flux H·u (m² s⁻¹, default `1e-3`)"
    uH::FT = 1e-3
    "entrainment, detrainment and melt rate (m s⁻¹, default `1e-8`, about 0.3 m yr⁻¹)"
    w::FT = 1e-8
    "reduced density contrast δρ (–, default `1e-6`)"
    drho::FT = 1e-6
    "temperature (K, default `0.01`)"
    T::FT = 0.01
    "salinity (psu, default `0.01`)"
    S::FT = 0.01
    "log-layer term `log(u★H/ν₀)` (–, default `0.1`)"
    log::FT = 0.1
end
SmoothKinks(H, u, uH, w, drho, T, S, log) = SmoothKinks(promote(H, u, uH, w, drho, T, S, log)...)

# The kernels take the object whole, so its fields are adapted with the other kernel
# arguments, as the caps' are (see `max_layer_thickness.jl`).
Adapt.@adapt_structure SmoothKinks

"""
    kink_width(kinks, group)

The half-width ε of the kink `group` (a field name of [`SmoothKinks`](@ref): `:H`, `:u`, …),
read inside kernels; `false` under [`ExactKinks`](@ref), whose helpers ignore it.
"""
function kink_width end

@inline kink_width(::ExactKinks, ::Symbol) = false
@inline kink_width(k::SmoothKinks, s::Symbol) = _val(getfield(k, s))
@inline kink_width(k, ::Val{s}) where {s} = kink_width(k, s)

# The smooth helpers.  Each smooth form is the exact one plus a correction that
# vanishes outside the band, computed from -|a - b| rather than from a - b: a blend
# formed as b + max₀(a - b) would round a small b away next to a large a (as
# min(max_detrainment = 1e6, detr) in Float32).  The band never sees an Inf (beds of
# -Inf): `clamp` takes it to the band edge, where the blend has zero slope, so
# reverse-mode AD does not multiply a zero adjoint by Inf.

# max(x, 0)
@inline _smax0(::ExactKinks, x, ε) = max(x, zero(x))
@inline function _smax0(::SmoothKinks, x, ε)
    e = ifelse(iszero(ε), one(ε), ε)
    xc = clamp(x, -e, e)
    s = ifelse(x > e, x, (xc + e)^2 / (4 * e))
    return ifelse(iszero(ε), max(x, zero(x)), s)
end
# min(x, 0), the downwind half of an upwind split.
@inline _sneg(::ExactKinks, x, ε) = min(x, zero(x))
@inline _sneg(k::SmoothKinks, x, ε) = min(x, zero(x)) - _smax0(k, -abs(x), ε)

"""
    smooth_max(kinks, a, b, ε)

`max(a, b)`, blended over the half-width `ε` under [`SmoothKinks`](@ref) and exact under
[`ExactKinks`](@ref). Use it, or [`smooth_min`](@ref), [`smooth_clamp`](@ref), for a new switch.
"""
function smooth_max end

@inline smooth_max(::ExactKinks, a, b, ε) = max(a, b)
@inline smooth_max(k::SmoothKinks, a, b, ε) = max(a, b) + _smax0(k, -abs(a - b), ε)

"""
    smooth_min(kinks, a, b, ε)

`min(a, b)`, as [`smooth_max`](@ref).
"""
function smooth_min end

@inline smooth_min(::ExactKinks, a, b, ε) = min(a, b)
@inline smooth_min(k::SmoothKinks, a, b, ε) = min(a, b) - _smax0(k, -abs(a - b), ε)

"""
    smooth_clamp(kinks, x, lo, hi, ε)

`clamp(x, lo, hi)`, as [`smooth_max`](@ref).
"""
function smooth_clamp end

@inline smooth_clamp(::ExactKinks, x, lo, hi, ε) = clamp(x, lo, hi)
@inline smooth_clamp(k::SmoothKinks, x, lo, hi, ε) = smooth_min(k, smooth_max(k, x, lo, ε), hi, ε)
# |x| as sign(x)·x, the form of the centred momentum advection.
@inline _sabs(::ExactKinks, x, ε) = sign(x) * x
@inline _sabs(k::SmoothKinks, x, ε) = abs(x) + 2 * _smax0(k, -abs(x), ε)
# m · (m > 0 ? a : b): a flux m that carries a from upstream on either side.  A flux
# that is zero exactly (a wall) stays zero: the blend would carry ε/4 · (a - b).
@inline _supwind(::ExactKinks, m, a, b, ε) = m * ifelse(m > zero(m), a, b)
@inline _supwind(k::SmoothKinks, m, a, b, ε) =
    ifelse(iszero(m), zero(m * a), _smax0(k, m, ε) * a + _sneg(k, m, ε) * b)
# √max(x, 0).  Smoothed below x = ε² by the quadratic (x + 3ε²)² / 16ε³, which meets
# √x with value and slope at ε² and reaches zero, flat, at -3ε²: the slope stays below
# 1/2ε, and √x is exact above the band.  The root of the unused branch is taken of ε².
@inline _ssqrt(::ExactKinks, x, ε) = _safe_sqrt(max(x, zero(x)))
@inline function _ssqrt(::SmoothKinks, x, ε)
    e = ifelse(iszero(ε), one(ε), ε)
    e2 = e * e
    xc = clamp(x, -3 * e2, e2)
    s = ifelse(x > e2, sqrt(ifelse(x > e2, x, e2)), (xc + 3 * e2)^2 / (16 * e2 * e))
    return ifelse(iszero(ε), _safe_sqrt(max(x, zero(x))), s)
end
# x > 0 ? a : b, blended with a cubic smoothstep over [-ε, ε].
@inline _sswitch(::ExactKinks, x, a, b, ε) = ifelse(x > zero(x), a, b)
@inline function _sswitch(::SmoothKinks, x, a, b, ε)
    e = ifelse(iszero(ε), one(ε), ε)
    t = clamp((x + e) / (2 * e), zero(x), one(x))
    s = b + (a - b) * (t * t * (3 - 2 * t))
    return ifelse(iszero(ε), ifelse(x > zero(x), a, b), s)
end

Base.show(io::IO, ::ExactKinks) = print(io, "ExactKinks()")
Base.show(io::IO, k::SmoothKinks) =
    print(io, "SmoothKinks(", join(("$f = $(getfield(k, f))" for f in fieldnames(SmoothKinks)), ", "), ")")
