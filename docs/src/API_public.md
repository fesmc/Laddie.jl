# Public API

```@index
```

### Entry points

```@docs
Model
build_isomip
run!
meltstats
to_backend
```

### Geometry ingestion

```@docs
build_laddie_mask
ice_base_depth
bed_elevation
fill_ocean_holes!
fill_shelf_holes!
```

### Model container

```@docs
Model
Params
RunConfig
```


### Parameterizations

#### Entrainment

```@docs
LambertEntrainment
GasparEntrainment
HollandEntrainment
```

#### Melt

```@docs
FixedGamTMelting
TurbulentGamTMelting
PrescribedMelting
```

#### Convection

```@docs
ClampDensity
ResetToAmbient
RelaxToAmbient
```

#### Open boundary

```@docs
ZeroGradientInflow
NoInflow
```

#### Grounding line

```@docs
AbstractGroundingLineBC
FreeSlipGL
NoSlipGL
```

#### Land

```@docs
AbstractLandBC
FreeSlipLand
NoSlipLand
```

#### Lateral viscosity

```@docs
AbstractLateralViscosity
PrescribedLateralViscosity
NonlinearLateralViscosity
```

#### Ice-front pressure gradient

```@docs
AbstractFrontPressure
FullDepthGradient
TruncatedDepthGradient
```

#### Shelf gaps

```@docs
AbstractGapsBC
SinkGapsBC
ConnectedGapsBC
```

#### Maximum layer thickness

```@docs
AbstractMaxLayerThickness
TopographicMaxLayerThickness
AbsoluteMaxLayerThickness
RelativeMaxLayerThickness
```

#### Time stepping

```@docs
FixedDt
AdaptiveDt
ConservativeCFL
ExactCFL
```

#### Simulation end

```@docs
AbstractSimulationEnd
FixedSimulationEnd
SteadyStateEnd
```

#### Coriolis parameter

```@docs
AbstractCoriolisParameter
CoriolisParameter0D
CoriolisParameter2D
```

### Forcing

A model is driven by a [`CavityForcing`](@ref): an ocean forcing supplying the
ambient T/S, and an ice forcing supplying the basal ice temperature. Passing an
ocean forcing on its own to `Model` is shorthand for pairing it with a uniform
`PrescribedIceForcing(-25.0)`.

```@docs
CavityForcing
```

#### Ocean

```@docs
AbstractOceanForcing
OceanForcing1D
ISOMIPForcing
Laddie.OceanForcing2D
```

#### Ice

```@docs
AbstractIceForcing
PrescribedIceForcing
```
