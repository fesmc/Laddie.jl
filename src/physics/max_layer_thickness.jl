"""
Abstract supertype for the upper bound on the layer thickness `H`. Pass a concrete instance as
`Params(; max_layer_thickness = ...)`: [`RelativeMaxLayerThickness`](@ref) (the default,
`RelativeMaxLayerThickness(1, 500)`), [`NoMaxLayerThickness`](@ref),
[`TopographicMaxLayerThickness`](@ref) or [`AbsoluteMaxLayerThickness`](@ref). Where the first guess of the stepped `H` exceeds the cap,
the excess is detrained at the layer's own T, S and velocity (`model.detr_cap`), as LADDIE v2
does with its `Hmax`.

!!! warning
    A cap removes water, and a tight one (a fraction of the water column) suppresses the
    refreezing and the strong boundary currents; see [Maximum layer thickness](@ref) before
    tightening the default.
"""
abstract type AbstractMaxLayerThickness end

"""
$(TYPEDEF)

Leave the layer thickness unbounded from above: `H` is set by the volume budget alone, as in
LADDIE v1, and may exceed the local water column.

Select via `Params(; max_layer_thickness = NoMaxLayerThickness())`.
"""
struct NoMaxLayerThickness <: AbstractMaxLayerThickness end

"""
$(TYPEDEF)

Cap the layer thickness at the local water-column depth, `H <= z_draft - z_bed`. Binds only
where [`Grid`](@ref) was given a bed elevation. Read the warning on
[`AbstractMaxLayerThickness`](@ref) first.

Select via `Params(; max_layer_thickness = TopographicMaxLayerThickness())`.
"""
struct TopographicMaxLayerThickness <: AbstractMaxLayerThickness end
"""
$(TYPEDEF)

Cap the layer thickness at a fixed value, `H <= H_max`, everywhere. Read the warning on
[`AbstractMaxLayerThickness`](@ref) first.

Select via `Params(; max_layer_thickness = AbsoluteMaxLayerThickness(100.0))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct AbsoluteMaxLayerThickness{FT} <: AbstractMaxLayerThickness
    "maximum layer thickness (m, default `100`; converted to the model precision by `Params`)"
    H_max::FT = 100.0
end

"""
$(TYPEDEF)

Cap the layer thickness at a fraction of the local water-column depth, never below `H_floor`:
`H <= max(f_H_max * (z_draft - z_bed), H_floor)`. Binds only where [`Grid`](@ref) was given a
bed elevation. `H_floor` keeps room for the layer where the draft nearly meets the bed; see
[Maximum layer thickness](@ref).

Select via `Params(; max_layer_thickness = RelativeMaxLayerThickness(1.0, 500.0))` (the default).

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

"""
    cap_layer_thickness(scheme, kinks, H, z_draft, z_bed, tmask)

The thickness `H` of one cell bounded by the cap `scheme` ([`AbstractMaxLayerThickness`](@ref)):
`H` without a cap, else `min(H, cap) · tmask`, smoothed by `kinks` over ε_H. Called inside
kernels, as the target of the cap detrainment and of the clamp after the thickness step.
"""
function cap_layer_thickness end

@inline cap_layer_thickness(::NoMaxLayerThickness, kinks, H, z_draft, z_bed, tmask) = H
@inline cap_layer_thickness(::TopographicMaxLayerThickness, kinks, H, z_draft, z_bed, tmask) =
    smooth_min(kinks, H, z_draft - z_bed, kink_width(kinks, :H)) * tmask
@inline cap_layer_thickness(c::AbsoluteMaxLayerThickness, kinks, H, z_draft, z_bed, tmask) =
    smooth_min(kinks, H, _val(c.H_max), kink_width(kinks, :H)) * tmask
# Without a bed (z_bed = −∞) there is no cap.  The water column is zeroed there before it
# meets f_H_max, so that ∂cap/∂f_H_max is 0 rather than 0·∞ = NaN in reverse mode.
@inline function cap_layer_thickness(c::RelativeMaxLayerThickness, kinks, H, z_draft, z_bed, tmask)
    eH = kink_width(kinks, :H)
    has_bed = isfinite(z_bed)
    water_column = ifelse(has_bed, z_draft - z_bed, zero(z_draft))
    cap = smooth_max(kinks, _val(c.f_H_max) * water_column, _val(c.H_floor), eH)
    return ifelse(has_bed, smooth_min(kinks, H, cap, eH), H) * tmask
end

# The caps reach `_net_entrainment_kernel!` and `_clamp_thickness_kernel!` whole, so their field has to be adapted
# with the other kernel arguments: a traced parameter (Reactant) becomes a device
# scalar there.  Identity on the CPU and CUDA backends.
Adapt.@adapt_structure AbsoluteMaxLayerThickness
Adapt.@adapt_structure RelativeMaxLayerThickness
