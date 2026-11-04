#############################
# Wall slip (grounding line and land)
#############################

# A wall's tangential-momentum condition is one slip factor `s` on the faces it
# bounds, with NEMO's convention (`rn_shlat`): the velocity behind the wall is a
# ghost (1 − s)·u, so 0 = free slip (no stress), 2 = no slip (u = 0 at the wall),
# 1 = u = 0 one grid spacing away, which is LADDIE v2's wall.
#   - lateral viscosity: the wall-face flux is −s·A·H·u/Δ².  The velocity stored
#     behind a wall is 0, so the stencil already holds one unit of it; the kernels
#     add the remaining (s − 1), see `_viscosity_slips`;
#   - momentum advection: the face velocity at the wall is (1 − s)·u, unless the
#     wall advection says otherwise (`AbstractWallAdvection`).

"Check a partial-slip factor lies in the admissible range [0, 2]."
function _check_slip(factor::Real)
    0 <= factor <= 2 || throw(
        ArgumentError(
            "slip factor must lie in [0, 2] (0 = free slip, 1 = LADDIE v2, 2 = no slip), got $factor",
        ),
    )
    return float(factor)
end

"""
Abstract supertype for the momentum boundary condition at grounding-line walls
(mask value `2`).  Pass a concrete instance as
`BoundaryConditions(; grounding_line = ...)`: [`PartialSlipGL`](@ref) (the default,
`PartialSlipGL(1.0)`, LADDIE v2's wall), [`NoSlipGL`](@ref) or [`FreeSlipGL`](@ref).

Walls bordering exposed rock are governed separately by [`AbstractLandBC`](@ref).
"""
abstract type AbstractGroundingLineBC end

"""
$(TYPEDEF)

No-slip momentum boundary condition at the grounding line: the tangential
velocity vanishes at walls bordering grounded ice (mask value `2`), i.e. at the
face between the last shelf cell and the grounded cell.  Implemented as a slip
factor of `2` on grounding-line faces.  Land walls (mask value `1`) are governed
independently by `BoundaryConditions.land`.

This is the viscous wall of LADDIE v1 as published (`slip = 1` in its configs:
v1 adds its factor to a stencil that already holds one unit of wall stress); see
[`V1WallAdvection`](@ref) for its advective part.  It is stronger than LADDIE v2's
"no slip", which is [`PartialSlipGL`](@ref)`(1.0)`.

Select via `BoundaryConditions(; grounding_line = NoSlipGL())`.
"""
struct NoSlipGL <: AbstractGroundingLineBC end

"""
$(TYPEDEF)

Free-slip momentum boundary condition at the grounding line: slip factor `0` at
walls bordering grounded ice, so the wall exerts no viscous stress on the flow.

Select via `BoundaryConditions(; grounding_line = FreeSlipGL())`.
"""
struct FreeSlipGL <: AbstractGroundingLineBC end

"""
$(TYPEDEF)

Partial-slip momentum boundary condition at the grounding line, with a slip
`factor` between `0` (free slip) and `2` (no slip): the velocity behind the wall
is taken as `(1 - factor)·u`.

`PartialSlipGL(1.0)` is the default and LADDIE v2's wall: v2 treats a grounded
neighbour as an ordinary cell with zero velocity, one grid spacing away
(`laddie_velocity.f90:259-267`), which v2 calls "no slip"
([Lambert et al. (2026)](@cite lambert_one-layer_2026)).

Select via `BoundaryConditions(; grounding_line = PartialSlipGL(1.0))` (the default).

# Fields
$(TYPEDFIELDS)
"""
struct PartialSlipGL{FT} <: AbstractGroundingLineBC
    "slip factor, from `0` (free slip) to `2` (no slip)"
    factor::FT
    PartialSlipGL(factor::Real) = (f = _check_slip(factor); new{typeof(f)}(f))
end

# Slip factor applied at grounding-line faces (0 = free slip, 1 = v2, 2 = no slip).
_gl_slip(::FreeSlipGL, FT) = zero(FT)
_gl_slip(::NoSlipGL, FT) = FT(2)
_gl_slip(bc::PartialSlipGL, FT) = FT(bc.factor)

"""
Abstract supertype for the momentum boundary condition at land walls (mask value
`1`: exposed bedrock, islands, and the outer border ring).  Pass a concrete
instance as `BoundaryConditions(; land = ...)`: [`PartialSlipLand`](@ref) (the
default, `PartialSlipLand(1.0)`, LADDIE v2's wall), [`NoSlipLand`](@ref) or
[`FreeSlipLand`](@ref).

The grounding line is governed separately by [`AbstractGroundingLineBC`](@ref); at a
coastline corner whose stencil touches both grounded ice and exposed rock, the
grounding-line condition takes precedence (the land wall indicators exclude faces
already flagged as grounding line), so each wall face gets exactly one slip factor.
"""
abstract type AbstractLandBC end

"""
$(TYPEDEF)

No-slip momentum boundary condition at land walls: the tangential velocity
vanishes at walls bordering exposed bedrock (mask value `1`), which includes
islands and ice-free coastline inside the domain as well as the outer border
ring.  Implemented as a slip factor of `2` on land faces.  See
[`NoSlipGL`](@ref) for its relation to LADDIE v1 and v2.

Select via `BoundaryConditions(; land = NoSlipLand())`.
"""
struct NoSlipLand <: AbstractLandBC end

"""
$(TYPEDEF)

Free-slip momentum boundary condition at land walls: slip factor `0` at walls
bordering exposed bedrock, so the wall exerts no viscous stress on the flow.

Select via `BoundaryConditions(; land = FreeSlipLand())`.
"""
struct FreeSlipLand <: AbstractLandBC end

"""
$(TYPEDEF)

Partial-slip momentum boundary condition at land walls, with a slip `factor`
between `0` (free slip) and `2` (no slip).  `PartialSlipLand(1.0)` is the default
and LADDIE v2's wall, where ice-free land cells, like grounded ones, hold zero
velocity; see [`PartialSlipGL`](@ref).

Select via `BoundaryConditions(; land = PartialSlipLand(1.0))` (the default).

# Fields
$(TYPEDFIELDS)
"""
struct PartialSlipLand{FT} <: AbstractLandBC
    "slip factor, from `0` (free slip) to `2` (no slip)"
    factor::FT
    PartialSlipLand(factor::Real) = (f = _check_slip(factor); new{typeof(f)}(f))
end

# Slip factor applied at land faces.  Mirrors `_gl_slip`; kept as a separate
# dispatch point so the two wall treatments can diverge later.
_land_slip(::FreeSlipLand, FT) = zero(FT)
_land_slip(::NoSlipLand, FT) = FT(2)
_land_slip(bc::PartialSlipLand, FT) = FT(bc.factor)

# Per-face slip factors of a model.
_wall_slips(m) =
    (_gl_slip(m.boundary.grounding_line, m.FT), _land_slip(m.boundary.land, m.FT))

# The extra wall drag the viscosity kernels add, `s − 1`: the zero velocity stored
# behind a wall already gives the stencil one unit of wall stress.  From −1 (free
# slip, cancelling it) to 1 (no slip).
_viscosity_slips(m) = map(s -> s - one(s), _wall_slips(m))

#############################
# Wall momentum advection
#############################

"""
Abstract supertype for how the wall slip condition enters the *advective* term
of the momentum equations, as opposed to the viscous wall drag.  Pass a concrete
instance as `BoundaryConditions(; wall_advection = ...)`:
[`NoWallAdvection`](@ref) (the default, LADDIE v2), [`SlipScaledWallAdvection`](@ref)
or [`V1WallAdvection`](@ref) (LADDIE v1).

The slip factor of [`AbstractGroundingLineBC`](@ref) / [`AbstractLandBC`](@ref)
always sets the viscous wall stress; this choice decides which factor, if any,
forms the momentum carried across a wall face by the upwind advection.
"""
abstract type AbstractWallAdvection end

"""
$(TYPEDEF)

Apply the wall slip factor to the upwind momentum advection as well as to the
viscous wall stress: the tangential momentum flux across a wall face is formed
with the same ghost velocity `(1 - slip)·u` as the viscosity, so free slip (`0`)
leaves the flux at its interior value and no slip (`2`) reverses it.

Select via `BoundaryConditions(; wall_advection = SlipScaledWallAdvection())`.
"""
struct SlipScaledWallAdvection <: AbstractWallAdvection end

"""
$(TYPEDEF)

Carry no momentum across walls: the upwind momentum advection sees a slip factor
of `0` at grounding-line and land faces whatever the wall condition is, while
the viscous wall stress keeps the factor of
[`AbstractGroundingLineBC`](@ref) / [`AbstractLandBC`](@ref).

This is the default and what LADDIE v2 does: both of its momentum-advection
schemes skip grounded neighbours outright (`laddie_velocity.f90`, "No flux
across grounding line"), and the wall enters only through `compute_viscUV`.
With the default [`PartialSlipGL`](@ref)`(1.0)` / [`PartialSlipLand`](@ref)`(1.0)`
it reproduces v2's walls.

Select via `BoundaryConditions(; wall_advection = NoWallAdvection())` (the default).
"""
struct NoWallAdvection <: AbstractWallAdvection end

"""
\$(TYPEDEF)

LADDIE v1's wall advection: the upwind momentum advection sees the wall slip
factor minus one, as a ghost velocity `(2 - slip)·u`.

v1 uses one parameter, `slip`, for both terms, but adds it to a viscous stencil
that already holds one unit of wall stress (`physics.py:lapU`), so its viscous
wall is one step stronger than its advective one.  v1's published `slip = 1` is
therefore [`NoSlipGL`](@ref) / [`NoSlipLand`](@ref) in the viscosity, with this
advection:

```julia
BoundaryConditions(; grounding_line = NoSlipGL(), land = NoSlipLand(),
                   wall_advection = V1WallAdvection())
```

A v1 `slip` below 1 maps to `PartialSlipGL(slip + 1)`; above 1 it has no
counterpart here.

Select via `BoundaryConditions(; wall_advection = V1WallAdvection())`.
"""
struct V1WallAdvection <: AbstractWallAdvection end

# Slip factors seen by the momentum-advection kernels, as opposed to the viscous
# wall stress, which always uses `_viscosity_slips`.
_advection_slips(m) = _advection_slips(m, m.boundary.wall_advection)
_advection_slips(m, ::SlipScaledWallAdvection) = _wall_slips(m)
_advection_slips(m, ::NoWallAdvection) = (zero(m.FT), zero(m.FT))
_advection_slips(m, ::V1WallAdvection) = _viscosity_slips(m)

# The walls of the u- and v-points as the momentum kernels take them, splatted into
# their arguments: the grounding-line and land indicators of the two wall faces
# (north and south of a u-point, east and west of a v-point), then the slip factors
# `(slip_gl, slip_land)` of `_viscosity_slips` or `_advection_slips`.  Separate arguments,
# not one NamedTuple: a NamedTuple of arrays unpacked inside a kernel costs the CPU
# kernels up to a third of their time.
_u_walls(m, (slip_gl, slip_land)) = (m.glNu, m.glSu, m.lndNu, m.lndSu, slip_gl, slip_land)
_v_walls(m, (slip_gl, slip_land)) = (m.glEv, m.glWv, m.lndEv, m.lndWv, slip_gl, slip_land)

#############################
# Open BC
#############################

"""
Abstract supertype for the open-ocean boundary condition at the ice front.  Pass a
concrete instance as `BoundaryConditions(; open_ocean = ...)`:
[`ZeroGradientInflow`](@ref) (the default) or [`NoInflow`](@ref).
"""
abstract type AbstractOpenOceanBC end

"""
$(TYPEDEF)

Open-boundary condition at the ice front: zero-gradient extrapolation of all
fields, with inflow from the ambient ocean permitted.

Select via `BoundaryConditions(; open_ocean = ZeroGradientInflow())` (the default).
"""
struct ZeroGradientInflow <: AbstractOpenOceanBC end

"""
$(TYPEDEF)

Open-boundary condition at the ice front: no inflow of layer properties.  Where
the flow enters the domain through an ice-front face, the thickness and tracer
fluxes carry the (zero) value stored in the open-ocean cell instead of the
zero-gradient extrapolation of [`ZeroGradientInflow`](@ref), so no volume, heat or
salt is advected in.  The velocities themselves are not clipped, and momentum
advection is unaffected.

Select via `BoundaryConditions(; open_ocean = NoInflow())`.
"""
struct NoInflow <: AbstractOpenOceanBC end

#############################
# Ice-front pressure gradient
#############################

"""
Abstract supertype for the treatment of the layer-thickness- and density-gradient
parts of the pressure-gradient force at one-sided faces — the ice front, and the
edges of a gap demoted to ocean by [`SinkGapsBC`](@ref).  Pass a concrete instance
as `Params(; front_pressure = ...)`: [`FullDepthGradient`](@ref) (LADDIE v1
v1.x, the default), [`TruncatedDepthGradient`](@ref) (LADDIE v2) or
[`ZeroGradientFront`](@ref) (the zero-lateral-gradient condition stated by
[Lambert et al. 2026](@cite lambert_one-layer_2026)).

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
D)/dx` with `Dxm1 = roll(D*tmask)`), so it is **the default** and the setting
under which Laddie.jl reproduces the LADDIE v1 code.

Select via `Params(; front_pressure = FullDepthGradient())` (the default).

See also [`TruncatedDepthGradient`](@ref).
"""
struct FullDepthGradient <: AbstractFrontPressure end

"""
$(TYPEDEF)

Drop the layer-thickness-gradient part of the pressure-gradient force at
one-sided faces — the ice front, and the edges of a gap demoted to ocean by
[`SinkGapsBC`](@ref) — keeping only the ice-base-slope and density-gradient
parts there.

This is what the LADDIE v2 does at calving-front faces
(`laddie_velocity.f90:118-126`, `IF (mask_cf_b .OR. mask_gl_b) ... assume
dH/dx = 0`). It avoids treating a masked neighbour's stored `H = 0` as a real
thickness: under [`FullDepthGradient`](@ref) that term is roughly 150× larger
at the ice front than in the interior on a warm ISOMIP+ run.

Neither choice affects the grounding line, where the velocity masks are zero
and no momentum equation is solved at all.

Select via `Params(; front_pressure = TruncatedDepthGradient())`.

See also [`ZeroGradientFront`](@ref), which drops the density-gradient part as well.
"""
struct TruncatedDepthGradient <: AbstractFrontPressure end

"""
$(TYPEDEF)

Drop both the layer-thickness-gradient and the density-gradient parts of the
pressure-gradient force at one-sided faces — the ice front, and the edges of a gap
demoted to ocean by [`SinkGapsBC`](@ref) — keeping only the ice-base-slope part
there.

Under [`TruncatedDepthGradient`](@ref) the density-gradient term is still
evaluated as `(δρ_neighbour - δρ)/Δ`, with the neighbour's stored `δρ`, which
masking pins to `0`: a one-sided gradient, as if the layer were neutrally buoyant
across one grid cell. This option makes the momentum equation consistent with the
condition stated in the LADDIE v2 description paper ([Lambert et al. 2026](@cite lambert_one-layer_2026)), "zero
lateral gradients are applied to all main variables (H, U, V, T, and S)" at the
calving front. The v2 code comments the same intent (`laddie_velocity.f90:119`,
"Assume dH/dx and ddrho/dx = 0") but keeps the `ddrho_amb_dx_b` term in its code.

Select via `Params(; front_pressure = ZeroGradientFront())`.
"""
struct ZeroGradientFront <: AbstractFrontPressure end

# Weights `w` in the per-face gate `1 + w*(tmask_stag - 2)`, which is 1 on a
# fully-interior face (tmask_stag == 2) either way, and at a one-sided face
# (tmask_stag == 1) is 1 for w = 0 (term kept) / 0 for w = 1 (term dropped).
# w = 0 multiplies the term by exactly 1.0, so the default is bit-identical to
# LADDIE v1.  `_front_pgf_weight` gates the thickness-gradient term,
# `_front_drho_weight` the density-gradient term.
_front_pgf_weight(::FullDepthGradient, x) = zero(x)
_front_pgf_weight(::TruncatedDepthGradient, x) = one(x)
_front_pgf_weight(::ZeroGradientFront, x) = one(x)
_front_drho_weight(::AbstractFrontPressure, x) = zero(x)
_front_drho_weight(::ZeroGradientFront, x) = one(x)

#############################
# Gaps BC
#############################

"""
Abstract supertype for the treatment of ice-shelf gaps — ice-free cells inside
the shelf domain, marked `4` in the mask.  Pass a concrete instance as
`BoundaryConditions(; gaps = ...)`: [`SinkGapsBC`](@ref) (the default, gaps act as sinks)
or [`ConnectedGapsBC`](@ref) (gaps stay dynamically connected).
"""
abstract type AbstractGapsBC end

"""
$(TYPEDEF)

Treat ice-shelf gaps — ice-free cells inside the ice-shelf domain, marked `4` in
the domain mask — as sinks of the meltwater layer (the default, and the behaviour
of LADDIE v1.x).  Gap cells are demoted to open ocean (`0`) at model construction,
so a meltwater current reaching a gap loses its heat and momentum to the ambient
ocean exactly as it would at the calving front, and cannot drive melting further
downstream.

Select via `BoundaryConditions(; gaps = SinkGapsBC())` (the default).

See also [`ConnectedGapsBC`](@ref).
"""
struct SinkGapsBC <: AbstractGapsBC end

"""
$(TYPEDEF)

Treat ice-shelf gaps as dynamically connected: the meltwater layer keeps being
integrated across ice-free cells inside the ice-shelf domain, so heat and momentum
are advected downstream and can drive melting on the far side of the gap.  No
melting occurs within a gap itself — there is no glacial ice to melt — so no
meltwater volume or buoyancy is added there.

Gap cells are those marked `4` in the domain mask.  Mark them either directly (a
coupled ice-sheet driver knows its own reference geometry) or from a reference ice
footprint with [`MarkGapsPreprocess`](@ref) in the `preprocess` list of [`Grid`](@ref).  This
boundary condition only decides what happens in the cells so marked.

Motivated by [Jesse et al. (2026, https://doi.org/10.5194/egusphere-2026-4237)](@cite jesse_ice-sheet_2026),
who show that the choice between this treatment and [`SinkGapsBC`](@ref) can alter
coupled ice-volume loss by an amount comparable to 1 °C of ocean warming.  The two
are deliberate end-members; reality lies somewhere between them.

# Example

```julia
boundary = BoundaryConditions(; gaps = ConnectedGapsBC())
Model(Grid(mask, z_draft, dx, dy); forcing, boundary)   # gaps already marked as 4
grid = Grid(mask, z_draft, dx, dy; preprocess = [MarkGapsPreprocess(refgeo_H)])
Model(grid; forcing, boundary)                          # gaps from a footprint
```
"""
struct ConnectedGapsBC <: AbstractGapsBC end

# Resolve mask value 4 (gap) into the classification the selected treatment implies.
# Runs at model construction, before the Geometry is derived: SinkGapsBC demotes gaps
# to open ocean, so every mask derivation sees exactly the v1.x geometry;
# ConnectedGapsBC keeps them.  Marking gaps in the first place is
# `MarkGapsPreprocess`'s job.
_apply_gaps_bc(mask, ::SinkGapsBC) = ifelse.(mask .== 4, 0, mask)
_apply_gaps_bc(mask, ::ConnectedGapsBC) = mask
#############################
# Boundary-condition container
#############################

"""
$(TYPEDEF)

The boundary conditions of a [`Model`](@ref): what happens at the ice front, at
grounding-line and land walls, and in ice-shelf gaps.  Pass it as
`Model(...; boundary = BoundaryConditions(...))`.  Every condition has a default.
The walls default to LADDIE v2's: zero velocity one grid spacing behind every wall
in the viscosity ([`PartialSlipGL`](@ref)`(1.0)`, [`PartialSlipLand`](@ref)`(1.0)`)
and no momentum across walls ([`NoWallAdvection`](@ref)).  The other conditions
are those of both LADDIE versions.

```julia
BoundaryConditions(; land = FreeSlipLand(), gaps = ConnectedGapsBC())

# LADDIE v1 walls (its published slip = 1)
BoundaryConditions(; grounding_line = NoSlipGL(), land = NoSlipLand(),
                   wall_advection = V1WallAdvection())
```

# Fields
$(TYPEDFIELDS)
"""
struct BoundaryConditions{
    OB<:AbstractOpenOceanBC,
    GL<:AbstractGroundingLineBC,
    LB<:AbstractLandBC,
    GB<:AbstractGapsBC,
    WA<:AbstractWallAdvection,
}
    "ice front: [`ZeroGradientInflow`](@ref) (default) or [`NoInflow`](@ref)"
    open_ocean::OB
    "grounding-line walls: [`PartialSlipGL`](@ref) (default `PartialSlipGL(1.0)`), [`NoSlipGL`](@ref) or [`FreeSlipGL`](@ref)"
    grounding_line::GL
    "land walls: [`PartialSlipLand`](@ref) (default `PartialSlipLand(1.0)`), [`NoSlipLand`](@ref) or [`FreeSlipLand`](@ref)"
    land::LB
    "ice-shelf gaps: [`SinkGapsBC`](@ref) (default) or [`ConnectedGapsBC`](@ref)"
    gaps::GB
    "wall slip in the momentum advection: [`NoWallAdvection`](@ref) (default), [`SlipScaledWallAdvection`](@ref) or [`V1WallAdvection`](@ref)"
    wall_advection::WA
end

BoundaryConditions(;
    open_ocean = ZeroGradientInflow(),
    grounding_line = PartialSlipGL(1.0),
    land = PartialSlipLand(1.0),
    gaps = SinkGapsBC(),
    wall_advection = NoWallAdvection(),
) = BoundaryConditions(open_ocean, grounding_line, land, gaps, wall_advection)
