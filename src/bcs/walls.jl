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

# The slip factor of a wall face, from the grounding-line and land indicators `gl`
# and `lnd` of the face and the slip factors `slip_gl`, `slip_land` (`_u_walls`/
# `_v_walls`); zero off the walls.  The indicators partition the wall faces (the
# grounding line takes precedence at mixed corners), so exactly one of the two
# factors applies on a wall.  It reads arrays, so it carries its own `@inbounds`, like
# the stencil helpers (see the note above `_face_avg`).
@inline _slip(slip_gl, slip_land, gl, lnd, i, j) =
    @inbounds slip_gl * gl[i, j] + slip_land * lnd[i, j]

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

```jldoctest
julia> BoundaryConditions(; grounding_line = NoSlipGL(), land = NoSlipLand(),
                          wall_advection = V1WallAdvection())
BoundaryConditions(open ocean = ZeroGradientInflow, grounding line = NoSlipGL, land = NoSlipLand, gaps = SinkGapsBC, wall advection = V1WallAdvection)
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
