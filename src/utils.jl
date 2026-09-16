"""
Abstract supertype for the upper bound applied to the layer thickness `D` after
each step.  Pass a concrete instance as `Params(; max_layer_thickness = ...)`:
[`NoMaxLayerThickness`](@ref) (the default), [`TopographicMaxLayerThickness`](@ref),
[`AbsoluteMaxLayerThickness`](@ref), or [`RelativeMaxLayerThickness`](@ref).

!!! warning "Every bound here is corrective, and that costs more than it looks"
    All of these clamp `D` *after* the thickness step, discarding volume while leaving
    the momentum and tracer content of the layer untouched. On a real cavity that acts
    as a mass sink wherever it binds, and the damage is not local: capping a
    Crosson–Dotson run at the water column collapsed the mean melt rate from 9.9 to
    1.1 m yr⁻¹ — including in cells where the cap never binds, because the plume that
    feeds them no longer develops (`laddie-roadmap/validation.md` §5.2).

    Neither reference bounds `D` this way. Python LADDIE v1 has no upper bound at all
    (its `maxD = 3000 m` never binds), and LADDIE v2 folds a fixed `Hmax` into the
    entrainment *before* the step, so the integration lands in range by itself. Prefer
    the default, and reach for a cap only as a stability stopgap on a run that is
    already misbehaving.
"""
abstract type AbstractMaxLayerThickness end

"""
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

Cap the layer thickness at the local water-column depth, `D <= z_draft - z_bed`.

This bound is only meaningful when a bed elevation was supplied: `Model` fills
`z_bed` with `-Inf` when `z_bed_raw` is not given, in which case the cap is
`+Inf` and `D` is effectively unbounded from above — which is why passing `z_bed`
used to change the solution drastically while this was the default.  Read the
warning on [`AbstractMaxLayerThickness`](@ref) before selecting it: on a real
cavity this cap can suppress melt by an order of magnitude.

Select via `Params(; max_layer_thickness = TopographicMaxLayerThickness())`.
"""
struct TopographicMaxLayerThickness <: AbstractMaxLayerThickness end
"""
$(TYPEDSIGNATURES)

Cap the layer thickness at a fixed value, `D <= D_max`, independent of
bathymetry.  Useful when no bed elevation is available; read the warning on
[`AbstractMaxLayerThickness`](@ref) first.

- `D_max`: maximum layer thickness in metres (default `100`).

Select via `Params(; max_layer_thickness = AbsoluteMaxLayerThickness(100.0))`.

See also [`NoMaxLayerThickness`](@ref) (the default) and
[`RelativeMaxLayerThickness`](@ref).
"""
@kwdef struct AbsoluteMaxLayerThickness{FT} <: AbstractMaxLayerThickness
    D_max::FT = 100
end

"""
$(TYPEDSIGNATURES)

Cap the layer thickness at a fraction of the local water-column depth,
`D <= f_D_max * (z_draft - z_bed)`, leaving some ambient column beneath the
plume.  Like [`TopographicMaxLayerThickness`](@ref) this only bites when
`z_bed_raw` was supplied to `Model`.

- `f_D_max`: fraction of the water column (default `4/5`).

Select via `Params(; max_layer_thickness = RelativeMaxLayerThickness(0.8))`.
"""
@kwdef struct RelativeMaxLayerThickness{FT} <: AbstractMaxLayerThickness
    f_D_max::FT = 4/5
end

# No upper bound: `_clamp_thickness!` applies the `D_min` floor and the domain
# mask on the next line, so there is nothing to do here.
max_layer_thickness!(::Any, ::NoMaxLayerThickness) = nothing
function max_layer_thickness!(m, ::TopographicMaxLayerThickness)
    @. m.D.future = min(m.D.future, m.z_draft - m.z_bed) .* m.tmask
end
function max_layer_thickness!(m, c::AbsoluteMaxLayerThickness)
    @. m.D.future = min(m.D.future, c.D_max, m.z_draft - m.z_bed) .* m.tmask
end
function max_layer_thickness!(m, c::RelativeMaxLayerThickness)
    @. m.D.future = min(m.D.future, c.f_D_max * (m.z_draft - m.z_bed)) .* m.tmask
end

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
im_t(m, a) = div0(a .+ xp1(a), m.tmask_im)
ip_t(m, a) = div0(a .+ xm1(a), m.tmask_ip)
jm_t(m, a) = div0(a .+ yp1(a), m.tmask_jm)
jp_t(m, a) = div0(a .+ ym1(a), m.tmask_jp)
im_u(m, a) = div0(a .+ xp1(a), m.umask_im)
ip_u(m, a) = div0(a .+ xm1(a), m.umask_ip)
jm_u(m, a) = div0(a .+ yp1(a), m.umask_jm)
jp_u(m, a) = div0(a .+ ym1(a), m.umask_jp)
im_v(m, a) = div0(a .+ xp1(a), m.vmask_im)
ip_v(m, a) = div0(a .+ xm1(a), m.vmask_ip)
jm_v(m, a) = div0(a .+ yp1(a), m.vmask_jm)
jp_v(m, a) = div0(a .+ ym1(a), m.vmask_jp)

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
