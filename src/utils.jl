"""
Abstract supertype for the upper bound applied to the layer thickness `D` after
each step.  Pass a concrete instance as `Params(; max_layer_thickness = ...)`:
[`NoMaxLayerThickness`](@ref) (the default), [`TopographicMaxLayerThickness`](@ref),
[`AbsoluteMaxLayerThickness`](@ref), or [`RelativeMaxLayerThickness`](@ref).

!!! warning "Every bound here is corrective, and that costs more than it looks"
    All of these clamp `D` *after* the thickness step, discarding volume while leaving
    the momentum and tracer content of the layer untouched. On a real cavity that acts
    as a mass sink wherever it binds, and the damage is not local: capping a
    Crosson–Dotson run (see the example page) at the water column collapsed the mean
    melt rate from 9.9 to 1.1 m yr⁻¹ — including in cells where the cap never binds,
    because the plume that feeds them no longer develops.

    Neither reference bounds `D` this way. Python LADDIE v1 has no upper bound at all
    (its `maxD = 3000 m` never binds), and LADDIE v2 folds a fixed `Hmax` into the
    entrainment *before* the step, so the integration lands in range by itself. Prefer
    the default, and reach for a cap only as a stability stopgap on a run that is
    already misbehaving.
"""
abstract type AbstractMaxLayerThickness end

"""
$(TYPEDEF)

Leave the layer thickness unbounded from above (the default): `D` is set by the
volume budget alone, exactly as in Python LADDIE v1, and only the `D_min` floor is
applied after the step.

The layer is then free to exceed the local water column where the budget takes it
there — as it does in the published Crosson–Dotson run, in 10 % of the shelf cells.
That is a known limitation of a one-layer model, and it is *not* worth fixing with a
corrective clamp; see the warning on [`AbstractMaxLayerThickness`](@ref).

Select via `Params(; max_layer_thickness = NoMaxLayerThickness())` (the default).
"""
struct NoMaxLayerThickness <: AbstractMaxLayerThickness end

"""
$(TYPEDEF)

Cap the layer thickness at the local water-column depth, `D <= z_draft - z_bed`.

This bound is only meaningful when a bed elevation was given to [`Grid`](@ref)
(`z_bed`); without one the bed is `-Inf`, the cap is `+Inf` and `D` is unbounded
from above.  Read the warning on [`AbstractMaxLayerThickness`](@ref) before
selecting it: on a real cavity this cap can suppress melt by an order of magnitude.

Select via `Params(; max_layer_thickness = TopographicMaxLayerThickness())`.
"""
struct TopographicMaxLayerThickness <: AbstractMaxLayerThickness end
"""
$(TYPEDEF)

Cap the layer thickness at a fixed value, `D <= D_max`, independent of
bathymetry: the cap is the same everywhere, whether or not a bed elevation was
given.  Read the warning on [`AbstractMaxLayerThickness`](@ref) first.

Select via `Params(; max_layer_thickness = AbsoluteMaxLayerThickness(100.0))`.

See also [`NoMaxLayerThickness`](@ref) (the default) and
[`RelativeMaxLayerThickness`](@ref).

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct AbsoluteMaxLayerThickness{FT} <: AbstractMaxLayerThickness
    "maximum layer thickness (m, default `100`; converted to the model precision by `Params`)"
    D_max::FT = 100.0f0
end

"""
$(TYPEDEF)

Cap the layer thickness at a fraction of the local water-column depth,
`D <= f_D_max * (z_draft - z_bed)`, leaving some ambient column beneath the
plume.  Like [`TopographicMaxLayerThickness`](@ref) this only bites when a bed
elevation `z_bed` was given to [`Grid`](@ref).

Select via `Params(; max_layer_thickness = RelativeMaxLayerThickness(0.8))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct RelativeMaxLayerThickness{FT} <: AbstractMaxLayerThickness
    "fraction of the local water column (default `4/5`)"
    f_D_max::FT = 4/5
end

# Per-cell upper bound on the thickness `D`, applied in `_clamp_thickness_kernel!`
# before the D_min floor.  No upper bound: the floor and the domain mask that
# follow are all there is.
@inline _max_layer_thickness(::NoMaxLayerThickness, D, z_draft, z_bed, tmask) = D
@inline _max_layer_thickness(::TopographicMaxLayerThickness, D, z_draft, z_bed, tmask) =
    min(D, z_draft - z_bed) * tmask
@inline _max_layer_thickness(c::AbsoluteMaxLayerThickness, D, z_draft, z_bed, tmask) =
    min(D, _val(c.D_max)) * tmask
@inline _max_layer_thickness(c::RelativeMaxLayerThickness, D, z_draft, z_bed, tmask) =
    min(D, _val(c.f_D_max) * (z_draft - z_bed)) * tmask

# The caps reach `_clamp_thickness_kernel!` whole, so their field has to be adapted
# with the other kernel arguments: a traced parameter (Reactant) becomes a device
# scalar there.  Identity on the CPU and CUDA backends.
KA.Adapt.@adapt_structure AbsoluteMaxLayerThickness
KA.Adapt.@adapt_structure RelativeMaxLayerThickness

# ============================================================================
# Shift / interpolation primitives  (≡ np.roll & tools.py, GPU-capable)
# ============================================================================

# Periodic one-cell shifts along each axis.  Arrays are stored as [ix, iy]: the
# first index runs along x, the second along y.  The domain is wrapped in a
# grounded border so periodic wrap is harmless (masked off everywhere).
#
# The names are historical and refer to the *index shift*, not to a compass
# direction: `xm1(a)[i, j] == a[i+1, j]` is the neighbour at the next x index.
# Grid axes need not align with east/north — a polar stereographic projection
# rotates them by an arbitrary angle — so nothing here means "east" or "north".
@inline xm1(a) = circshift(a, (-1, 0))   # next x neighbour : a[i+1, j]
@inline xp1(a) = circshift(a, (1, 0))    # prev x neighbour : a[i-1, j]
@inline ym1(a) = circshift(a, (0, -1))   # next y neighbour : a[i, j+1]
@inline yp1(a) = circshift(a, (0, 1))    # prev y neighbour : a[i, j-1]

# Arithmetic-mean interpolation to cell-face midpoints.
# Naming convention: `im` = value at i−½, `ip` = i+½, `jm` = j−½, `jp` = j+½.
im_half(a) = (a .+ xp1(a)) ./ 2
ip_half(a) = (a .+ xm1(a)) ./ 2
jm_half(a) = (a .+ yp1(a)) ./ 2
jp_half(a) = (a .+ ym1(a)) ./ 2

# Safe division: returns 0 where the denominator is zero.
div0(a, b) = ifelse.(b .== 0, zero(eltype(a)), a ./ b)

# Masked staggered interpolation — normalises by the count of live neighbours
# to avoid gradient artefacts across boundaries (tools.py in the reference).
# The counts (0, 1 or 2 active cells) are formed like the kernels form them inline.
im_count(mask) = mask .+ xp1(mask)
ip_count(mask) = mask .+ xm1(mask)
jm_count(mask) = mask .+ yp1(mask)
jp_count(mask) = mask .+ ym1(mask)
ip_t(m, a) = div0(a .+ xm1(a), ip_count(m.tmask))
jp_t(m, a) = div0(a .+ ym1(a), jp_count(m.tmask))
im_u(m, a) = div0(a .+ xp1(a), im_count(m.umask))
jm_v(m, a) = div0(a .+ yp1(a), jm_count(m.vmask))

# Cells with at least one open-ocean neighbour.
next_to_ocean(ocn) = xm1(ocn) .+ xp1(ocn) .+ ym1(ocn) .+ yp1(ocn) .> 0

# Numpy-style gradient: second-order central differences on the interior,
# first-order one-sided at the two boundary rows/columns.
function gradient_x(a, dx)
    g = similar(a)
    n = size(a, 1)
    @views g[2:(n-1), :] .= (a[3:n, :] .- a[1:(n-2), :]) ./ (2dx)
    @views g[1, :] .= (a[2, :] .- a[1, :]) ./ dx
    @views g[n, :] .= (a[n, :] .- a[n-1, :]) ./ dx
    return g
end
function gradient_y(a, dy)
    g = similar(a)
    n = size(a, 2)
    @views g[:, 2:(n-1)] .= (a[:, 3:n] .- a[:, 1:(n-2)]) ./ (2dy)
    @views g[:, 1] .= (a[:, 2] .- a[:, 1]) ./ dy
    @views g[:, n] .= (a[:, n] .- a[:, n-1]) ./ dy
    return g
end
