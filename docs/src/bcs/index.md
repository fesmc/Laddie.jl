# Boundary conditions

The layer is bounded by walls (the grounding line and land), by the open ocean at the ice
front, by gaps in the shelf, and by the water and ice it exchanges heat and salt with.
Each of these is a family of types under one abstract supertype, as the
[physics](../physics/index.md) parameterisations are, and the code lives in `src/bcs/`:

| abstract type | selected with | default |
|---------------|---------------|---------|
| [`AbstractSlipAtGL`](@ref), [`AbstractSlipAtLand`](@ref) | `BoundaryConditions(; grounding_line_slip, land_slip)` | `PartialSlipAtGL(1.0)`, `PartialSlipAtLand(1.0)` |
| [`AbstractWallAdvection`](@ref) | `BoundaryConditions(; wall_advection)` | `NoWallAdvection()` |
| [`AbstractOpenOceanBC`](@ref) | `BoundaryConditions(; open_ocean)` | `OutflowOnlyOpenOceanBC()` |
| [`AbstractFrontPressure`](@ref) | `Params(; front_pressure)` | `ZeroGradientFrontPressure()` |
| [`AbstractGapsBC`](@ref) | `BoundaryConditions(; gaps)` | `SinkGapsBC()` |
| [`AbstractOceanForcing`](@ref) | `Model(grid; forcing)` | — |
| [`AbstractIceForcing`](@ref) | `Model(grid; forcing)` | `PrescribedIceForcing(-25)` |

The pages of this section describe one family each:

- [Wall slip](walls.md) and [wall momentum advection](wall_advection.md): the grounding line and
  land as zero-flux walls, and how their slip enters the viscous and the advective terms.
- [Open-ocean boundary](open_ocean.md) and [ice-front pressure gradient](front_pressure.md):
  inflow and outflow at the ice front, and the pressure gradient across it.
- [Ice-shelf gaps](gaps.md): melt-through holes as sinks or as part of the active domain.
- [Ocean forcing](ocean_forcing.md): the ambient temperature and salinity entrained into the layer.
- [Ice forcing](ice_forcing.md): the ice temperature behind the melting and ice-conduction schemes.

The [LADDIE versions](../versions.md) page gives the option that v1 and v2 each use.
