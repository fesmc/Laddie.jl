#############################
# Boundary-condition container
#############################

"""
$(TYPEDEF)

The boundary conditions of a [`Model`](@ref): what happens at the ice front, at
grounding-line and land walls, and in ice-shelf gaps.  Pass it as
`Model(...; boundary = BoundaryConditions(...))`.  Every condition has a default.
The walls default to LADDIE v2's: zero velocity one grid spacing behind every wall
in the viscosity ([`PartialSlipAtGL`](@ref)`(1.0)`, [`PartialSlipAtLand`](@ref)`(1.0)`)
and no momentum across walls ([`NoWallAdvection`](@ref)).  The other conditions
are those of both LADDIE versions.

```jldoctest
julia> BoundaryConditions(; land_slip = FreeSlipAtLand(), gaps = ConnectedGapsBC())
BoundaryConditions(open ocean = OutflowOnlyOpenOceanBC, grounding line = PartialSlipAtGL, land = FreeSlipAtLand, gaps = ConnectedGapsBC, wall advection = NoWallAdvection)

julia> BoundaryConditions(; grounding_line_slip = NoSlipAtGL(), land_slip = NoSlipAtLand(),   # LADDIE v1 walls
                          wall_advection = V1WallAdvection())
BoundaryConditions(open ocean = OutflowOnlyOpenOceanBC, grounding line = NoSlipAtGL, land = NoSlipAtLand, gaps = SinkGapsBC, wall advection = V1WallAdvection)
```

# Fields
$(TYPEDFIELDS)
"""
struct BoundaryConditions{
    OB<:AbstractOpenOceanBC,
    GL<:AbstractSlipAtGL,
    LB<:AbstractSlipAtLand,
    GB<:AbstractGapsBC,
    WA<:AbstractWallAdvection,
}
    "ice front: [`OutflowOnlyOpenOceanBC`](@ref) (default), [`ZeroGradientOpenOceanBC`](@ref) or [`NoInflowOpenOceanBC`](@ref)"
    open_ocean::OB
    "grounding-line walls: [`PartialSlipAtGL`](@ref) (default `PartialSlipAtGL(1.0)`), [`NoSlipAtGL`](@ref) or [`FreeSlipAtGL`](@ref)"
    grounding_line_slip::GL
    "land walls: [`PartialSlipAtLand`](@ref) (default `PartialSlipAtLand(1.0)`), [`NoSlipAtLand`](@ref) or [`FreeSlipAtLand`](@ref)"
    land_slip::LB
    "ice-shelf gaps: [`SinkGapsBC`](@ref) (default), [`ConnectedGapsBC`](@ref) or [`LeakyGapsBC`](@ref)"
    gaps::GB
    "wall slip in the momentum advection: [`NoWallAdvection`](@ref) (default), [`SlipScaledWallAdvection`](@ref) or [`V1WallAdvection`](@ref)"
    wall_advection::WA
end

BoundaryConditions(;
    open_ocean = OutflowOnlyOpenOceanBC(),
    grounding_line_slip = PartialSlipAtGL(1.0),
    land_slip = PartialSlipAtLand(1.0),
    gaps = SinkGapsBC(),
    wall_advection = NoWallAdvection(),
) = BoundaryConditions(open_ocean, grounding_line_slip, land_slip, gaps, wall_advection)
