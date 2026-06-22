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
ProfileForcing
```
