#############################
# Open BC
#############################

"""
Abstract supertype for the open-ocean boundary condition at the ice front.  Pass a
concrete instance as `BoundaryConditions(; open_ocean = ...)`:
[`OutflowOnlyOpenOceanBC`](@ref) (the default), [`ZeroGradientOpenOceanBC`](@ref) or [`NoInflowOpenOceanBC`](@ref).
"""
abstract type AbstractOpenOceanBC end

"""
$(TYPEDEF)

Open-boundary condition at the ice front: zero-gradient extrapolation of all
fields, with inflow from the ambient ocean permitted.

LADDIE v1 (`boundop = 1`) and v2 use this condition; the reproduction aliases pin it.

Select via `BoundaryConditions(; open_ocean = ZeroGradientOpenOceanBC())`.
"""
struct ZeroGradientOpenOceanBC <: AbstractOpenOceanBC end

"""
$(TYPEDEF)

Open-boundary condition at the ice front: no inflow of layer properties. Where the flow
enters through an ice-front face, the thickness and tracer fluxes carry the (zero) value of
the open-ocean cell, so no volume, heat or salt is advected in. The velocities are not
clipped.

Select via `BoundaryConditions(; open_ocean = NoInflowOpenOceanBC())`.
"""
struct NoInflowOpenOceanBC <: AbstractOpenOceanBC end

"""
$(TYPEDEF)

Ice-front boundary as a one-way valve: the layer leaves through the front but never enters.
After each momentum step the new face-normal velocity at every one-sided face (the ice front,
and gap edges under [`SinkGapsBC`](@ref)) is clipped to point out of the cavity
([`clip_front_inflow!`](@ref)), smoothly under [`SmoothKinks`](@ref); the fluxes carry nothing
in, as under [`NoInflowOpenOceanBC`](@ref).

The default, with [`ZeroGradientFrontPressure`](@ref); the comparison with the other conditions
is on the [Open-ocean boundary](@ref) page.

Select via `BoundaryConditions(; open_ocean = OutflowOnlyOpenOceanBC())` (the default).
"""
struct OutflowOnlyOpenOceanBC <: AbstractOpenOceanBC end

"""
    inflow_weight(open_ocean, FT)

Weight of the zero-gradient inflow in the thickness and tracer fluxes through an inflow face
at the ice front, under the [`AbstractOpenOceanBC`](@ref) `open_ocean`: `1` carries the front
cell's own values in, `0` nothing.
"""
function inflow_weight end

inflow_weight(::ZeroGradientOpenOceanBC, FT) = one(FT)
inflow_weight(::NoInflowOpenOceanBC, FT) = zero(FT)
# As NoInflowOpenOceanBC: covers the inward present velocities the valve has not yet cleared
# (an inward start fading through the Robert–Asselin filter).
inflow_weight(::OutflowOnlyOpenOceanBC, FT) = inflow_weight(NoInflowOpenOceanBC(), FT)

# The ice-front valve of OutflowOnlyOpenOceanBC on a new face velocity `u`: `n` is the outward
# normal of a one-sided face (+1 when the ocean is on the + side, −1 when on the −
# side) and 0 at every other face, which is left untouched.
@inline _outward_only(kinks, u, n, ε) = ifelse(iszero(n), u, n * _smax0(kinks, n * u, ε))

@kernel function _front_valve_kernel!(U, V, @Const(umask), @Const(vmask), @Const(tmask), kinks)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    ε = kink_width(kinks, :u)
    @inbounds begin
        n_u = umask[i, j] * (tmask[i, j] - tmask[i + 1, j])
        n_v = vmask[i, j] * (tmask[i, j] - tmask[i, j + 1])
        U[i, j] = _outward_only(kinks, U[i, j], n_u, ε)
        V[i, j] = _outward_only(kinks, V[i, j], n_v, ε)
    end
end

"""
$(TYPEDSIGNATURES)

Clip the new face-normal velocity at every ice-front face to point out of the cavity
(`u_n ← max(u_n, 0)` along the outward normal), for [`OutflowOnlyOpenOceanBC`](@ref); the other
open-ocean conditions leave the velocities untouched. Called after both momentum steps,
before the speed cap of `clamp_velocities!`, which keeps the sign.
"""
clip_front_inflow!(m, ::AbstractOpenOceanBC) = nothing
function clip_front_inflow!(m, ::OutflowOnlyOpenOceanBC)
    launch_interior!(_front_valve_kernel!, m.U.future, m.V.future, m.umask, m.vmask, m.tmask, m.params.kinks)
    return
end

#############################
# Ice-front pressure gradient
#############################

"""
Abstract supertype for the condition on the layer's reduced pressure `P = ½ g δρ H²` beyond
one-sided faces (the ice front, and gap edges under [`SinkGapsBC`](@ref)), which sets the
layer-thickness- and density-gradient parts of the pressure-gradient force there. Pass a
concrete instance as `Params(; front_pressure = ...)`: [`ZeroGradientFrontPressure`](@ref)
(the default), [`ZeroExteriorFrontPressure`](@ref) (LADDIE v2 code) or
[`V1FrontPressure`](@ref) (LADDIE v1).
"""
abstract type AbstractFrontPressure end

"""
$(TYPEDEF)

Zero normal gradient of `P = ½ g δρ H²` at one-sided faces (the ice front, and gap edges
under [`SinkGapsBC`](@ref)): drop both the layer-thickness- and density-gradient parts of the
pressure-gradient force there, keeping the ice-base-slope part. The zero lateral gradient
at the calving front of [Lambert et al. (2026)](@cite lambert_one-layer_2026).

Choose it for runs at Δx below the Rossby radius, over which the outflow's buoyancy decays
beyond the front ([Ice-front pressure gradient](@ref)).

Select via `Params(; front_pressure = ZeroGradientFrontPressure())` (the default).
"""
struct ZeroGradientFrontPressure <: AbstractFrontPressure end

"""
$(TYPEDEF)

`P = ½ g δρ H² = 0` one cell beyond one-sided faces (the ice front, and gap edges under
[`SinkGapsBC`](@ref)): the outward tendency of `HU` is `½ g δρ H²/Δx`. Drops the
layer-thickness-gradient part of the pressure-gradient force there and keeps the density
part against the masked `δρ = 0`, as the LADDIE v2 code does at calving-front faces.

Choose it to reproduce LADDIE v2.

Select via `Params(; front_pressure = ZeroExteriorFrontPressure())`.
"""
struct ZeroExteriorFrontPressure <: AbstractFrontPressure end

"""
$(TYPEDEF)

LADDIE v1's one-sided faces (the ice front, and gap edges under [`SinkGapsBC`](@ref)): keep
both the layer-thickness- and density-gradient parts of the pressure-gradient force against
the masked neighbour's `H = δρ = 0`. Their face averages collapse to the front cell's values,
so the outward tendency of `HU` is `3/2 g δρ H²/Δx`, 3× that of
[`ZeroExteriorFrontPressure`](@ref).

Choose it to reproduce LADDIE v1; pinned in the v1 reproduction aliases.

Select via `Params(; front_pressure = V1FrontPressure())`.
"""
struct V1FrontPressure <: AbstractFrontPressure end

"""
    front_pgf_weight(scheme, x)

Weight that gates the layer-thickness-gradient part of the pressure-gradient force at
one-sided faces under the [`AbstractFrontPressure`](@ref) `scheme`: `0` keeps the term, `1` drops it.
"""
function front_pgf_weight end

# Weights `w` in the per-face gate `1 + w*(tmask_stag - 2)`, which is 1 on a
# fully-interior face (tmask_stag == 2) either way, and at a one-sided face
# (tmask_stag == 1) is 1 for w = 0 (term kept) / 0 for w = 1 (term dropped).
# w = 0 multiplies the term by exactly 1.0, so V1FrontPressure is bit-identical
# to LADDIE v1.  `front_pgf_weight` gates the thickness-gradient term,
# `front_drho_weight` the density-gradient term.
front_pgf_weight(::ZeroGradientFrontPressure, x) = one(x)
front_pgf_weight(::ZeroExteriorFrontPressure, x) = one(x)
front_pgf_weight(::V1FrontPressure, x) = zero(x)

"""
    front_drho_weight(scheme, x)

As [`front_pgf_weight`](@ref), for the density-gradient part.
"""
function front_drho_weight end

front_drho_weight(::AbstractFrontPressure, x) = zero(x)
front_drho_weight(::ZeroGradientFrontPressure, x) = one(x)
