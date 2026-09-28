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
