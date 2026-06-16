# Public API

```@index
```

### Entry points

```@docs
build_model
build_isomip
run!
meltstats
to_backend
```

### Geometry ingestion

```@docs
build_laddie_mask
ice_base_depth
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
FixedGamT
TurbulentGamT
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
FreeSlipGL
NoSlipGL
```

#### Time stepping

```@docs
FixedDt
AdaptiveDt
```

#### Simulation end

```@docs
FixedSimulationEnd
SteadyStateEnd
```

### Ambient forcing

```@docs
ISOMIPForcing
LinearForcing
Linear2Forcing
TanhForcing
FileForcing
ProfileForcing
```
