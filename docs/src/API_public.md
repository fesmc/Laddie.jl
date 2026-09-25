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

## Reactant backend and automatic differentiation

See [Reactant backend](@ref) for the setup, the fusion strategies and their cost.

```@docs
ReactantBackend
integrate!
reactant_compile
trace_parameters
adaptive_schedule
DtSchedule
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

### Mask preprocessing

```@docs
AbstractPreprocess
FillOceanHolesPreprocess
FillShelfHolesPreprocess
FillSmallShelfPatchesPreprocess
FillSmallGroundedPatchesPreprocess
MarkGapsPreprocess
```

### Domain cropping

```@docs
AbstractDomainCropping
MinRectangleDomainCropping
NoDomainCropping
```

### Ice-base slope

```@docs
AbstractIceSlopeGradient
JlGradient
PyGradient
```

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
ambient T/S, and an ice forcing supplying the basal ice temperature. Passing an
ocean forcing on its own to `Model` is shorthand for pairing it with a uniform
`PrescribedIceForcing(-25.0)`.

```@docs
CavityForcing
```

### Ocean

```@docs
AbstractOceanForcing
OceanForcing1D
ISOMIPForcing
```

### Ice

```@docs
AbstractIceForcing
PrescribedIceForcing
```
