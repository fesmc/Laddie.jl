"""
$(TYPEDEF)

Abstract supertype for how the model treats its kinks: the switches of `max`, `min`,
`clamp` and upwinding, and the unbounded slope of `√x` at zero.  Pass a concrete
instance as `Params(; kinks = ...)`: [`ExactKinks`](@ref) (the default) or
[`SmoothKinks`](@ref).

The kinks make the model piecewise smooth.  A derivative exists almost everywhere, but
it jumps where a switch flips, and gradient-based optimisers can stall there.  Over a
long window, an adjoint that multiplies through such switches every step can also grow
much faster than the finite-difference response does.  `SmoothKinks` replaces each
switch by a ``C^1`` blend over a band of width ε, outside which the model is unchanged.
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

Smooth every kink over a band of half-width ε, one ε per physical quantity:

  * `max(x, 0)` becomes the ``C^1`` quadratic blend ``(x + ε)^2 / 4ε`` for
    ``|x| < ε`` and is exact outside the band.  `max(a, b)`, `min(a, b)`, `clamp`,
    `|x|` and the upwind splits ``u^\\pm`` follow from it, so ``u^+ + u^- = u``
    still holds and the upwind fluxes stay conservative.
  * `√x` (of `max(x, 0)`) is blended into the quadratic
    ``(x + 3ε^2)^2 / 16ε^3`` below ``x = ε^2``, which removes the unbounded slope at
    zero, and is exact above it.
  * The step of the conductive `c_i` at the freezing point is blended with a cubic
    smoothstep over ``T - T_f \\in [-ε_T, ε_T]``.

Where each ε acts:

| ε | where |
|---|---|
| `H` | `H_min` top-up, cap detrainment against the cap and the `H_min` floor |
| `u` | upwind tracer and thickness fluxes, the centred momentum advection's `sign(u)`, the speed cap, `√` of the drag, speed, deformation and Holland's shear |
| `uH` | donor of the momentum under [`UpstreamMomentumAdvection`](@ref) (sign of the face mass flux) |
| `w` | split of the Gaspar rate into entrainment and detrainment, `max_detrainment`, the melt discriminant |
| `drho` | `drho_floor` of Gaspar entrainment, the `ClampDensity` floor |
| `T` | conductive `c_i` at the freezing point, frazil supercooling, `T_min`/`T_max` |
| `S` | `S_min`/`S_max` |
| `log` | floor of the log-layer term of [`TurbulentGamTMelting`](@ref) |

An ε of zero keeps that group's kinks exact, so the groups can be switched on one at a
time.

Not smoothed are the kinks that the state sits on exactly rather than crosses, where
a blend would bias every step by about ε/4: the outer `max(0, ⋅)` of the cap
detrainment (zero wherever the cap does not bind), the clamp of `H` after the
thickness step (a safety net behind `ent2` and the cap detrainment, which bring the
stepped `H` exactly to its bounds) and the share of the flux-form tracer update that
follows that clamp.  Nor are the convective switches of `ResetToAmbient` and
`RelaxToAmbient`, the γ_T of [`PrescribedMelting`](@ref), the depth interpolation of
the forcing, or the time-step controller.

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
KA.Adapt.@adapt_structure SmoothKinks

# The half-width ε of one group, read inside a kernel.  `false` under ExactKinks,
# whose helpers ignore it; it still survives the arithmetic of a call site (ε/dt).
@inline _kink_width(::ExactKinks, ::Symbol) = false
@inline _kink_width(k::SmoothKinks, s::Symbol) = _val(getfield(k, s))
@inline _kink_width(k, ::Val{s}) where {s} = _kink_width(k, s)

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
@inline _smax(::ExactKinks, a, b, ε) = max(a, b)
@inline _smax(k::SmoothKinks, a, b, ε) = max(a, b) + _smax0(k, -abs(a - b), ε)
@inline _smin(::ExactKinks, a, b, ε) = min(a, b)
@inline _smin(k::SmoothKinks, a, b, ε) = min(a, b) - _smax0(k, -abs(a - b), ε)
@inline _sclamp(::ExactKinks, x, lo, hi, ε) = clamp(x, lo, hi)
@inline _sclamp(k::SmoothKinks, x, lo, hi, ε) = _smin(k, _smax(k, x, lo, ε), hi, ε)
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
