"""
Abstract supertype for the upper bound on the layer thickness `H`.  Pass a concrete
instance as `Params(; max_layer_thickness = ...)`: [`NoMaxLayerThickness`](@ref)
(the default), [`TopographicMaxLayerThickness`](@ref),
[`AbsoluteMaxLayerThickness`](@ref), or [`RelativeMaxLayerThickness`](@ref).

The cap is enforced before the thickness step, as LADDIE v2 does with its `Hmax`:
where the first guess of the stepped `H` exceeds it, the excess is detrained
(`model.detr_cap`, output as `:detr_cap`).  The detrained water leaves at the
layer's own T, S and velocity, so the cap thins the layer without changing its
tracers or momentum.  A clamp after the step remains as a safety net.

!!! warning "A cap still removes water, and that can cost more than it looks"
    Until 2026-10 the cap was purely corrective: it clamped `H` *after* the step and
    kept the clipped volume's tracer content. Wherever it bound, T and S then
    drifted towards the incoming meltwater and entrainment without dilution, which
    is what pinned pan-Antarctic runs at the `T_min` and `S_max` bounds. Capping a
    Crosson–Dotson run (see the example page) at the water column also collapsed
    the mean melt rate from 9.9 to 1.1 m yr⁻¹, including in cells where the cap never
    binds. Both findings predate the detrainment form and have not been re-measured
    with it.

    Python LADDIE v1 has no upper bound at all (its `maxD = 3000 m` never binds),
    and LADDIE v2's Antarctic setup uses a fixed `Hmax` of 10 000 m that never binds
    either. Prefer the default, and reach for a cap only as a stability measure.
"""
abstract type AbstractMaxLayerThickness end

"""
$(TYPEDEF)

Leave the layer thickness unbounded from above (the default): `H` is set by the
volume budget alone, exactly as in Python LADDIE v1, and only the `H_min` floor
applies.

The layer is then free to exceed the local water column where the budget takes it
there — as it does in the published Crosson–Dotson run, in 10 % of the shelf cells.
That is a known limitation of a one-layer model; see the warning on
[`AbstractMaxLayerThickness`](@ref) before capping it.

Select via `Params(; max_layer_thickness = NoMaxLayerThickness())` (the default).
"""
struct NoMaxLayerThickness <: AbstractMaxLayerThickness end

"""
$(TYPEDEF)

Cap the layer thickness at the local water-column depth, `H <= z_draft - z_bed`.

This bound is only meaningful when a bed elevation was given to [`Grid`](@ref)
(`z_bed`); without one the bed is `-Inf`, the cap is `+Inf` and `H` is unbounded
from above.  Read the warning on [`AbstractMaxLayerThickness`](@ref) before
selecting it: on a real cavity this cap can suppress melt by an order of magnitude.

Select via `Params(; max_layer_thickness = TopographicMaxLayerThickness())`.
"""
struct TopographicMaxLayerThickness <: AbstractMaxLayerThickness end
"""
$(TYPEDEF)

Cap the layer thickness at a fixed value, `H <= H_max`, independent of
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
    H_max::FT = 100.0f0
end

"""
$(TYPEDEF)

Cap the layer thickness at a fraction of the local water-column depth, but never
below `H_floor`: `H <= max(f_H_max * (z_draft - z_bed), H_floor)`.  The fraction
leaves some ambient column beneath the plume.  Like
[`TopographicMaxLayerThickness`](@ref) this only bites when a bed elevation `z_bed`
was given to [`Grid`](@ref).

`H_floor` is for cells where the draft nearly meets the bed.  A cell's water column
is the vertical distance between its mean draft and its mean bed, as if the draft
were horizontal; where the two nearly meet, the draft is unlikely to be, and a plume
can be thicker than that mean distance.  Capped at a few metres, such a cell also
leaves the layer almost no room between `H_min` and the cap, and the cap
detrainment and the `H_min` top-up then have to pass all the water that neighbours
advect into it through a layer of about a metre.

Select via `Params(; max_layer_thickness = RelativeMaxLayerThickness(0.8))`, or
`RelativeMaxLayerThickness(0.8, 10.0)` for a 10 m floor.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct RelativeMaxLayerThickness{FT} <: AbstractMaxLayerThickness
    "fraction of the local water column (default `4/5`)"
    f_H_max::FT = 4/5
    "lowest value of the cap, where the water column is thin (m, default `0`, i.e. none)"
    H_floor::FT = zero(f_H_max)
end
function RelativeMaxLayerThickness(f_H_max::Real, H_floor::Real = zero(f_H_max))
    f, h = promote(f_H_max, H_floor)
    return RelativeMaxLayerThickness{typeof(f)}(f, h)
end

# Per-cell upper bound on the thickness `H`: the target of the cap detrainment in
# `_net_entrainment_kernel!`, and the clamp in `_clamp_thickness_kernel!` before the
# H_min floor.  No upper bound: the floor and the domain mask are all there is.
@inline _max_layer_thickness(::NoMaxLayerThickness, H, z_draft, z_bed, tmask) = H
@inline _max_layer_thickness(::TopographicMaxLayerThickness, H, z_draft, z_bed, tmask) =
    min(H, z_draft - z_bed) * tmask
@inline _max_layer_thickness(c::AbsoluteMaxLayerThickness, H, z_draft, z_bed, tmask) =
    min(H, _val(c.H_max)) * tmask
@inline _max_layer_thickness(c::RelativeMaxLayerThickness, H, z_draft, z_bed, tmask) =
    min(H, max(_val(c.f_H_max) * (z_draft - z_bed), _val(c.H_floor))) * tmask

# The caps reach `_net_entrainment_kernel!` and `_clamp_thickness_kernel!` whole, so their field has to be adapted
# with the other kernel arguments: a traced parameter (Reactant) becomes a device
# scalar there.  Identity on the CPU and CUDA backends.
KA.Adapt.@adapt_structure AbsoluteMaxLayerThickness
KA.Adapt.@adapt_structure RelativeMaxLayerThickness
