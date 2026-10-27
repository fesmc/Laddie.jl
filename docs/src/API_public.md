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
