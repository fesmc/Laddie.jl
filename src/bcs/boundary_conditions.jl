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

```jldoctest
julia> BoundaryConditions(; land = FreeSlipLand(), gaps = ConnectedGapsBC())
BoundaryConditions(open ocean = ZeroGradientInflow, grounding line = PartialSlipGL, land = FreeSlipLand, gaps = ConnectedGapsBC, wall advection = NoWallAdvection)

julia> BoundaryConditions(; grounding_line = NoSlipGL(), land = NoSlipLand(),   # LADDIE v1 walls
                          wall_advection = V1WallAdvection())
BoundaryConditions(open ocean = ZeroGradientInflow, grounding line = NoSlipGL, land = NoSlipLand, gaps = SinkGapsBC, wall advection = V1WallAdvection)
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
    "ice front: [`ZeroGradientInflow`](@ref) (default), [`NoInflow`](@ref) or [`OutflowOnly`](@ref)"
    open_ocean::OB
    "grounding-line walls: [`PartialSlipGL`](@ref) (default `PartialSlipGL(1.0)`), [`NoSlipGL`](@ref) or [`FreeSlipGL`](@ref)"
    grounding_line::GL
    "land walls: [`PartialSlipLand`](@ref) (default `PartialSlipLand(1.0)`), [`NoSlipLand`](@ref) or [`FreeSlipLand`](@ref)"
    land::LB
    "ice-shelf gaps: [`SinkGapsBC`](@ref) (default), [`ConnectedGapsBC`](@ref) or [`LeakyGapsBC`](@ref)"
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
