# Model setup and running

```@index
Pages = ["API_public.md"]
```

## Entry points

```@docs
Grid
Model
Simulation
build_isomip
run!
time_step!
meltstats
to_backend
```

## Published setups

Aliases for the runs of earlier LADDIE versions: `Params<Setup>v1|v2` returns the
[`Params`](@ref) of a run, and `Simulation<Setup>v1|v2` builds the whole run from the
geometry it left behind under `papers/` (not part of the repository). Where a paper and
the code that produced its results disagree, the values follow the code and the
docstring says how the paper differs. LADDIE v2 runs on an unstructured mesh with
forward–backward Runge–Kutta time stepping, so its aliases match its parameters, not its
numerics.

| Setup | Paper | Parameters | Simulation |
|-------|-------|------------|------------|
| Crosson–Dotson | Lambert et al. (2023) | [`ParamsCrossonDotsonv1`](@ref) | [`SimulationCrossonDotsonv1`](@ref) |
| Filchner–Ronne | Lambert et al. (2023) | [`ParamsFRISv1`](@ref) | [`SimulationFRISv1`](@ref) |
| Pine Island | Lambert et al. (2023) | [`ParamsPIGv1`](@ref) | [`SimulationPIGv1`](@ref) |
| Amundsen Sea, v1 | Lambert et al. (2026) | [`ParamsASEv1`](@ref) | [`SimulationASEv1`](@ref), [`SimulationASEv1Theoretical`](@ref) |
| ISOMIP+ Ocean1 | Lambert et al. (2026) | [`ParamsISOMIPv2`](@ref) | [`SimulationISOMIPv2`](@ref) |
| pan-Antarctic | Lambert et al. (2026) | [`ParamsAISv2`](@ref) | — |
| Amundsen Sea, v2 | Lambert et al. (2026) | [`ParamsASEv2`](@ref) | [`SimulationASEv2`](@ref) |
| Pine Island, 120 m | Lambert et al. (2026) | [`ParamsPIGv2`](@ref) | — |
| MISOMIP+ (coupled) | Lambert et al. (2026) | [`ParamsMISOMIPv2`](@ref) | [`SimulationMISOMIPv2`](@ref) |
| ice-shelf gaps (coupled) | Jesse et al. (2026) | [`ParamsMISOMIPGapsv2`](@ref) | [`SimulationMISOMIPGapsv2`](@ref) |

```@docs
ParamsCrossonDotsonv1
ParamsFRISv1
ParamsPIGv1
ParamsASEv1
SimulationCrossonDotsonv1
SimulationFRISv1
SimulationPIGv1
SimulationASEv1
SimulationASEv1Theoretical
ParamsISOMIPv2
ParamsAISv2
ParamsASEv2
ParamsPIGv2
ParamsMISOMIPv2
ParamsMISOMIPGapsv2
SimulationISOMIPv2
SimulationASEv2
SimulationMISOMIPv2
SimulationMISOMIPGapsv2
```

## Running on Reactant

See [Reactant backend](@ref) for the setup, the fusion strategies and their cost, and
[Automatic differentiation](@ref "Automatic differentiation (API)") for differentiating a run.

```@docs
ReactantBackend
```

## Geometry ingestion

```@docs
build_laddie_mask
ice_base_depth
bed_elevation
fill_ocean_holes!
fill_shelf_holes!
fill_small_shelf_patches!
fill_small_grounded_patches!
```

Mask preprocessing, domain cropping and the ice-base slope are described in
[Mask preprocessing](numerics/preprocessing.md), [Domain cropping](numerics/domain_cropping.md)
and [Ice-base slope](numerics/ice_slope.md).

## Model container

```@docs
Params
Params()
BoundaryConditions
State
Cache
Laddie.Var
```

## Simulation

```@docs
Clock
OutputConfig
DebugConfig
```


## Forcing

A model is driven by a [`CavityForcing`](@ref): an ocean forcing supplying the
ambient T/S, and an ice forcing supplying the ice temperature `T_ice_base` (the temperature of the ice
above the basal boundary layer, typically the surface temperature; see
[`PrescribedIceForcing`](@ref)). Passing an
ocean forcing on its own to `Model` is shorthand for pairing it with a uniform
`PrescribedIceForcing(-25.0)`.

```@docs
CavityForcing
```

### Ocean

The ocean forcing types are described in [Ocean forcing](physics/ocean_forcing.md).

#### From a 3D ocean field

Ocean models that force ice-sheet models mostly have no ice-shelf cavities.
[`front_ocean_forcing`](@ref) gives every cavity cell the water column at the ice
front that feeds it; [`regrid_ocean`](@ref) and [`fill_ocean!`](@ref) bring a
coarse ocean grid onto the model's cells.

```@docs
front_ocean_forcing
front_sources
nearest_seed
regrid_ocean
fill_ocean!
```

### Ice

```@docs
AbstractIceForcing
PrescribedIceForcing
```
