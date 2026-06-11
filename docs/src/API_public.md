# Public API

```@index
```

### Entry points

```@docs
build_isomip
build_from_config
run!
meltstats
to_backend!
```

### Model container


### Parameterizations

#### Entrainment

```@docs
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

### Ambient forcing

```@docs
ISOMIPForcing
LinearForcing
Linear2Forcing
TanhForcing
FileForcing
```
