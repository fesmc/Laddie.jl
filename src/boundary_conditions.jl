#############################
# Grounding Line BC
#############################

"""
Abstract supertype for the momentum boundary condition at grounding-line walls
(mask value `2`).  Pass a concrete instance as `Params(; grline_bc = ...)`:
[`FreeSlipGL`](@ref) (the default) or [`NoSlipGL`](@ref).

Walls bordering exposed rock are governed separately by [`AbstractLandBC`](@ref).
"""
abstract type AbstractGroundingLineBC end

"""
$(TYPEDSIGNATURES)

Grounding-line momentum boundary condition of LADDIE v1.x (the default):
grounding-line walls (mask value `2`) use the global `Params.slip` factor
(`1.0` = free slip) — the same factor land walls get from the default
[`FreeSlipLand`](@ref), so the two wall types are indistinguishable unless
`grline_bc` and/or `land_bc` are changed independently.

Select via `Params(; grline_bc = FreeSlipGL())` (the default).
"""
struct FreeSlipGL <: AbstractGroundingLineBC end

"""
$(TYPEDSIGNATURES)

No-slip momentum boundary condition at the grounding line: the tangential
velocity is forced to zero at walls bordering grounded ice (mask value `2`).
Implemented as a slip factor of `2` on grounding-line faces (ghost velocity =
-interior velocity).  Land walls (mask value `1`) are governed independently
by `Params.land_bc` and are unaffected by this choice — see [`NoSlipLand`](@ref)
to apply the same no-slip treatment there too.

Motivated by LADDIE v2.0 (Lambert et al., in review, 2026), where a no-slip
grounding-line condition improves melt patterns near the grounding line
compared to observations.  Not yet validated against LADDIE v2.0 output.

Select via `Params(; grline_bc = NoSlipGL())`.
"""
struct NoSlipGL <: AbstractGroundingLineBC end

# Slip factor applied at grounding-line faces.  Ghost tangential velocity is
# (1 − factor)·u: 1 → free slip, 2 → no slip.
_gl_slip(::FreeSlipGL, slip) = slip
_gl_slip(::NoSlipGL, slip) = oftype(slip, 2)

#############################
# Land BC
#############################

"""
Abstract supertype for the momentum boundary condition at land walls (mask value
`1`: exposed bedrock, islands, and the outer border ring).  Pass a concrete
instance as `Params(; land_bc = ...)`: [`FreeSlipLand`](@ref) (the default) or
[`NoSlipLand`](@ref).

The grounding line is governed separately by [`AbstractGroundingLineBC`](@ref);
at a corner touching both, the grounding line takes precedence.
"""
abstract type AbstractLandBC end

"""
$(TYPEDSIGNATURES)

Land momentum boundary condition (the default): land walls — exposed bedrock
(mask value `1`), which includes islands and ice-free coastline inside the
domain as well as the outer border ring — use the global `Params.slip` factor
(`1.0` = free slip).  Reproduces LADDIE v1.x behaviour, where land was not
distinguished from any other wall.

Select via `Params(; land_bc = FreeSlipLand())` (the default).

See also [`AbstractGroundingLineBC`](@ref), the analogous choice for walls
bordering grounded ice.
"""
struct FreeSlipLand <: AbstractLandBC end

"""
$(TYPEDSIGNATURES)

No-slip momentum boundary condition at land walls: the tangential velocity is
forced to zero at walls bordering exposed bedrock (mask value `1`).
Implemented the same way as [`NoSlipGL`](@ref) — a slip factor of `2` on land
faces (ghost velocity = -interior velocity) — so the two can be composed
independently: e.g. no-slip at the grounding line but free-slip at islands,
or vice versa.

At a coastline corner whose stencil touches both grounded ice and exposed rock,
the grounding-line condition takes precedence (`Grid.lnd??` excludes faces
already flagged by `Grid.gl??`), so the two slip factors partition the wall
faces instead of both applying to the same face.

Select via `Params(; land_bc = NoSlipLand())`.
"""
struct NoSlipLand <: AbstractLandBC end

# Slip factor applied at land faces.  Mirrors `_gl_slip` exactly; kept as a
# separate dispatch point (rather than sharing one function across both
# abstract types) so land and grounding-line treatments can diverge later
# without disturbing each other.
_land_slip(::FreeSlipLand, slip) = slip
_land_slip(::NoSlipLand, slip) = oftype(slip, 2)

#############################
# Open BC
#############################

abstract type AbstractOpenOceanBC end

"""
$(TYPEDSIGNATURES)

Open-boundary condition at the ice front: zero-gradient extrapolation of all
fields, with inflow from the ambient ocean permitted.

Select via `Params(; open_bc = ZeroGradientInflow())` (the default).
"""
struct ZeroGradientInflow <: AbstractOpenOceanBC end

"""
$(TYPEDSIGNATURES)

Open-boundary condition at the ice front: outflow only — inflow velocities are
clipped to zero so ambient water cannot advect into the domain.

Select via `Params(; open_bc = NoInflow())`.
"""
struct NoInflow <: AbstractOpenOceanBC end

#############################
# Ice-front pressure gradient
#############################

"""
Abstract supertype for the treatment of the layer-thickness-gradient part of the
pressure-gradient force at one-sided faces — the ice front, and the edges of a gap
demoted to ocean by [`SinkGapsBC`](@ref).  Pass a concrete instance as
`Params(; front_pressure = ...)`: [`FullDepthGradient`](@ref) (Python LADDIE v1.x,
the default) or [`TruncatedDepthGradient`](@ref) (LADDIE v2).

The grounding line is unaffected either way — no momentum equation is solved there.
"""
abstract type AbstractFrontPressure end

"""
$(TYPEDSIGNATURES)

Keep the layer-thickness-gradient part of the pressure-gradient force at
one-sided faces — the ice front, and the edges of a gap demoted to ocean by
[`SinkGapsBC`](@ref) — evaluating it as `(D_neighbour - D)/Δ` with the
neighbour's stored thickness, which masking pins to `0` there. The term is thus
a full one-sided gradient, as if `D` fell to zero across one grid cell.

This is what Python LADDIE v1.x does (`integrate.py`, `-g·ip_t(Ddrho)·(Dxm1 -
D)/dx` with `Dxm1 = roll(D*tmask)`), so it is **the default** and the setting
under which Laddie.jl reproduces the Python reference.

Select via `Params(; front_pressure = FullDepthGradient())` (the default).

See also [`TruncatedDepthGradient`](@ref).
"""
struct FullDepthGradient <: AbstractFrontPressure end

"""
$(TYPEDSIGNATURES)

Drop the layer-thickness-gradient part of the pressure-gradient force at
one-sided faces — the ice front, and the edges of a gap demoted to ocean by
[`SinkGapsBC`](@ref) — keeping only the ice-base-slope and density-gradient
parts there.

This is what the LADDIE v2 Fortran reference does at calving-front faces
(`laddie_velocity.f90:118-126`, `IF (mask_cf_b .OR. mask_gl_b) ... assume
dH/dx = 0`). It avoids treating a masked neighbour's stored `D = 0` as a real
thickness: under [`FullDepthGradient`](@ref) that term is roughly 150× larger
at the ice front than in the interior on a warm ISOMIP+ run.

Neither choice affects the grounding line, where `Grid.umask`/`vmask` are zero
and no momentum equation is solved at all.

Select via `Params(; front_pressure = TruncatedDepthGradient())`.
"""
struct TruncatedDepthGradient <: AbstractFrontPressure end

# Weight `w` in the per-face gate `1 + w*(tmask_stag - 2)`, which is 1 on a
# fully-interior face (tmask_stag == 2) either way, and at a one-sided face
# (tmask_stag == 1) is 1 for FullDepthGradient / 0 for TruncatedDepthGradient.
# w = 0 multiplies the term by exactly 1.0, so the default stays bit-identical
# to the pre-AbstractFrontPressure (and Python v1.x) behaviour.
_front_pgf_weight(::FullDepthGradient, x) = zero(x)
_front_pgf_weight(::TruncatedDepthGradient, x) = one(x)

#############################
# Gaps BC
#############################

"""
Abstract supertype for the treatment of ice-shelf gaps — ice-free cells inside
the shelf domain, marked `4` in the mask.  Pass a concrete instance as
`Params(; gaps_bc = ...)`: [`SinkGapsBC`](@ref) (the default, gaps act as sinks)
or [`ConnectedGapsBC`](@ref) (gaps stay dynamically connected).
"""
abstract type AbstractGapsBC end

"""
$(TYPEDSIGNATURES)

Treat ice-shelf gaps — ice-free cells inside the ice-shelf domain, marked `4` in
the domain mask — as sinks of the meltwater layer (the default, and the behaviour
of LADDIE v1.x).  Gap cells are demoted to open ocean (`0`) at model construction,
so a meltwater current reaching a gap loses its heat and momentum to the ambient
ocean exactly as it would at the calving front, and cannot drive melting further
downstream.

Select via `Params(; gaps_bc = SinkGapsBC())` (the default).

See also [`ConnectedGapsBC`](@ref).
"""
struct SinkGapsBC <: AbstractGapsBC end

"""
$(TYPEDSIGNATURES)

Treat ice-shelf gaps as dynamically connected: the meltwater layer keeps being
integrated across ice-free cells inside the ice-shelf domain, so heat and momentum
are advected downstream and can drive melting on the far side of the gap.  No
melting occurs within a gap itself — there is no glacial ice to melt — so no
meltwater volume or buoyancy is added there.

Gap cells are those marked `4` in the domain mask.  Mark them either directly (a
coupled ice-sheet driver knows its own reference geometry, and this mirrors the
`refgeo_Hi > 0` test of the reference implementation) or by passing a reference
ice footprint as `refgeo`: every ocean cell (`0`) where `refgeo` is true is
promoted to a gap at model construction.

`refgeo` may be a `Bool` matrix, or any numeric matrix in which positive entries
mark ice (e.g. a reference ice thickness).  It must match the size of the mask
*as passed to* `Model`, i.e. before domain cropping.

Motivated by Jesse et al. (2026, https://doi.org/10.5194/egusphere-2026-4237),
who show that the choice between this treatment and [`SinkGapsBC`](@ref) can alter
coupled ice-volume loss by an amount comparable to 1 °C of ocean warming.  The two
are deliberate end-members; reality lies somewhere between them.

# Example

```julia
Params(; gaps_bc = ConnectedGapsBC())              # gaps already marked as 4
Params(; gaps_bc = ConnectedGapsBC(refgeo_H))      # derive gaps from a footprint
```

# Fields
 - `refgeo`: reference ice footprint, or `nothing` (default) to use the mask as given.
"""
struct ConnectedGapsBC{R} <: AbstractGapsBC
    refgeo::R
end

ConnectedGapsBC() = ConnectedGapsBC(nothing)

# Resolve mask value 4 (gap) into the classification the selected treatment implies.
# Runs at model construction, before Grid: SinkGapsBC demotes gaps to open ocean, so
# every downstream mask derivation sees exactly the v1.x geometry; ConnectedGapsBC
# keeps them, and additionally promotes ocean cells inside `refgeo` to gaps.
_apply_gaps_bc(mask, ::SinkGapsBC) = ifelse.(mask .== 4, 0, mask)

_apply_gaps_bc(mask, bc::ConnectedGapsBC{Nothing}) = mask

function _apply_gaps_bc(mask, bc::ConnectedGapsBC)
    size(bc.refgeo) == size(mask) || throw(
        ArgumentError(
            "ConnectedGapsBC refgeo must have the same size as the mask, got " *
            "$(size(bc.refgeo)) vs $(size(mask)); note the mask is the one passed to " *
            "`Model`, before any domain cropping",
        ),
    )
    had_ice = bc.refgeo isa AbstractMatrix{Bool} ? bc.refgeo : bc.refgeo .> 0
    return ifelse.((mask .== 0) .& had_ice, 4, mask)
end