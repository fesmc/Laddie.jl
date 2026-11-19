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

Open-boundary condition at the ice front: no inflow of layer properties.  Where
the flow enters the domain through an ice-front face, the thickness and tracer
fluxes carry the (zero) value stored in the open-ocean cell instead of the
zero-gradient extrapolation of [`ZeroGradientOpenOceanBC`](@ref), so no volume, heat or
salt is advected in.  The velocities themselves are not clipped, and momentum
advection is unaffected.

Select via `BoundaryConditions(; open_ocean = NoInflowOpenOceanBC())`.
"""
struct NoInflowOpenOceanBC <: AbstractOpenOceanBC end

"""
$(TYPEDEF)

Open-boundary condition at the ice front: a one-way valve.  The layer may leave
through an ice-front face but never enter: after each momentum step, the new
face-normal velocity at every ice-front face is clipped to point out of the cavity
(`max(u_n, 0)` along the outward normal, `clip_front_inflow!`), so an inflow
face acts as a wall that exerts no stress on the normal flow.

Under [`ZeroGradientOpenOceanBC`](@ref) an inflow face recirculates the front cell's own
water, which thickens and cools the front row; under [`NoInflowOpenOceanBC`](@ref) the inward
velocity is kept but carries nothing, so the front row thins.  Here the meltwater
layer is fed only by entrainment and melt, never by water crossing the front.
Ambient water that really enters a cavity does so below the plume.

The clip acts on the new level only.  The filtered present level, which advects
the thickness and tracers, relaxes to outward-only within a few steps (an inward
start, e.g. a restart from another condition, fades through the Robert–Asselin
filter); during that transient the fluxes carry nothing in, as under
[`NoInflowOpenOceanBC`](@ref).

The clip is the τ → 0 limit of a relaxation of the inward normal velocity.  It is
an instantaneous adjustment rather than a relaxation because the sinks of
[`AdvectiveMomentumForm`](@ref) are explicit, so a relaxation fast enough
to act as a valve would be unstable.

Gap edges under [`SinkGapsBC`](@ref) are one-sided faces too, so the valve acts
there as well.  With [`SmoothKinks`](@ref) the clip is smooth (`:u` width).

This is the default: with [`ZeroGradientFrontPressure`](@ref), on a warm ISOMIP+ cavity at
4 km, it keeps the total melt of the zero-gradient inflow within 0.4 % and removes its
melt anomaly in the front row (front-to-interior melt 0.93 instead of 1.87).

Select via `BoundaryConditions(; open_ocean = OutflowOnlyOpenOceanBC())` (the default).
"""
struct OutflowOnlyOpenOceanBC <: AbstractOpenOceanBC end

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
open-ocean conditions leave the velocities untouched.  Runs after both momentum
steps and before the speed cap of `clamp_velocities!`, which scales by a
positive factor and so keeps the outward sign.
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
Abstract supertype for the treatment of the layer-thickness- and density-gradient
parts of the pressure-gradient force at one-sided faces — the ice front, and the
edges of a gap demoted to ocean by [`SinkGapsBC`](@ref).  Pass a concrete instance
as `Params(; front_pressure = ...)`: [`ZeroGradientFrontPressure`](@ref) (the default, the
zero-lateral-gradient condition stated by [Lambert et al. 2026](@cite lambert_one-layer_2026)),
[`FullDepthFrontPressure`](@ref) (LADDIE v1) or [`TruncatedDepthFrontPressure`](@ref) (LADDIE v2).

The grounding line is unaffected either way — no momentum equation is solved there.
"""
abstract type AbstractFrontPressure end

"""
$(TYPEDEF)

Keep the layer-thickness-gradient part of the pressure-gradient force at
one-sided faces — the ice front, and the edges of a gap demoted to ocean by
[`SinkGapsBC`](@ref) — evaluating it as `(H_neighbour - H)/Δ` with the
neighbour's stored thickness, which masking pins to `0` there. The term is thus
a full one-sided gradient, as if `H` fell to zero across one grid cell.

This is what LADDIE v1 does (`integrate.py`, `-g·ip_t(Ddrho)·(Dxm1 -
D)/dx` with `Dxm1 = roll(D*tmask)`), so it is the setting under which Laddie.jl
reproduces the LADDIE v1 code, pinned in the v1 reproduction aliases.

Select via `Params(; front_pressure = FullDepthFrontPressure())`.

See also [`TruncatedDepthFrontPressure`](@ref).
"""
struct FullDepthFrontPressure <: AbstractFrontPressure end

"""
$(TYPEDEF)

Drop the layer-thickness-gradient part of the pressure-gradient force at
one-sided faces — the ice front, and the edges of a gap demoted to ocean by
[`SinkGapsBC`](@ref) — keeping only the ice-base-slope and density-gradient
parts there.

This is what the LADDIE v2 does at calving-front faces
(`laddie_velocity.f90:118-126`, `IF (mask_cf_b .OR. mask_gl_b) ... assume
dH/dx = 0`). It avoids treating a masked neighbour's stored `H = 0` as a real
thickness: under [`FullDepthFrontPressure`](@ref) that term is roughly 150× larger
at the ice front than in the interior on a warm ISOMIP+ run.

Neither choice affects the grounding line, where the velocity masks are zero
and no momentum equation is solved at all.

Select via `Params(; front_pressure = TruncatedDepthFrontPressure())`.

See also [`ZeroGradientFrontPressure`](@ref), which drops the density-gradient part as well.
"""
struct TruncatedDepthFrontPressure <: AbstractFrontPressure end

"""
$(TYPEDEF)

Drop both the layer-thickness-gradient and the density-gradient parts of the
pressure-gradient force at one-sided faces — the ice front, and the edges of a gap
demoted to ocean by [`SinkGapsBC`](@ref) — keeping only the ice-base-slope part
there.

Under [`TruncatedDepthFrontPressure`](@ref) the density-gradient term is still
evaluated as `(δρ_neighbour - δρ)/Δ`, with the neighbour's stored `δρ`, which
masking pins to `0`: a one-sided gradient, as if the layer were neutrally buoyant
across one grid cell. This option makes the momentum equation consistent with the
condition stated in the LADDIE v2 description paper ([Lambert et al. 2026](@cite lambert_one-layer_2026)), "zero
lateral gradients are applied to all main variables (H, U, V, T, and S)" at the
calving front. The v2 code comments the same intent (`laddie_velocity.f90:119`,
"Assume dH/dx and ddrho/dx = 0") but keeps the `ddrho_amb_dx_b` term in its code.

This is the default, together with [`OutflowOnlyOpenOceanBC`](@ref).

Select via `Params(; front_pressure = ZeroGradientFrontPressure())` (the default).
"""
struct ZeroGradientFrontPressure <: AbstractFrontPressure end

# Weights `w` in the per-face gate `1 + w*(tmask_stag - 2)`, which is 1 on a
# fully-interior face (tmask_stag == 2) either way, and at a one-sided face
# (tmask_stag == 1) is 1 for w = 0 (term kept) / 0 for w = 1 (term dropped).
# w = 0 multiplies the term by exactly 1.0, so FullDepthFrontPressure is bit-identical
# to LADDIE v1.  `front_pgf_weight` gates the thickness-gradient term,
# `front_drho_weight` the density-gradient term.
front_pgf_weight(::FullDepthFrontPressure, x) = zero(x)
front_pgf_weight(::TruncatedDepthFrontPressure, x) = one(x)
front_pgf_weight(::ZeroGradientFrontPressure, x) = one(x)
front_drho_weight(::AbstractFrontPressure, x) = zero(x)
front_drho_weight(::ZeroGradientFrontPressure, x) = one(x)
