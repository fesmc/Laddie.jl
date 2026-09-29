# Parameterizations and boundary conditions

```@index
Pages = ["API_physics.md"]
```

## Entrainment

```@docs
AbstractEntrainment
LambertEntrainment
GasparEntrainment
HollandEntrainment
```

## Melt

```@docs
AbstractMelting
FixedGamTMelting
TurbulentGamTMelting
UStarGamTMelting
PrescribedMelting
```

## Convection

```@docs
AbstractConvectionScheme
ClampDensity
ResetToAmbient
RelaxToAmbient
```

## Open boundary

```@docs
AbstractOpenOceanBC
ZeroGradientInflow
NoInflow
```

## Grounding line

```@docs
AbstractGroundingLineBC
NoSlipGL
FreeSlipGL
PartialSlipGL
```

## Land

```@docs
AbstractLandBC
NoSlipLand
FreeSlipLand
PartialSlipLand
```

## Wall momentum advection

```@docs
AbstractWallAdvection
SlipScaledWallAdvection
NoWallAdvection
```

## Momentum advection

```@docs
AbstractMomentumAdvection
CentredMomentumAdvection
UpstreamMomentumAdvection
```

## Lateral viscosity

```@docs
AbstractLateralViscosity
PrescribedLateralViscosity
NonlinearLateralViscosity
```

## Laplacian thickness weights

```@docs
AbstractLaplacianWeights
PresentLaplacianWeights
PastLaplacianWeights
```

## Ice-front pressure gradient

```@docs
AbstractFrontPressure
FullDepthGradient
TruncatedDepthGradient
```

## Shelf gaps

```@docs
AbstractGapsBC
SinkGapsBC
ConnectedGapsBC
```

## Maximum layer thickness

```@docs
AbstractMaxLayerThickness
NoMaxLayerThickness
TopographicMaxLayerThickness
AbsoluteMaxLayerThickness
RelativeMaxLayerThickness
```

## Time stepping

```@docs
AbstractTimeStepper
FixedDt
AdaptiveDt
AbstractCFL
ExactCFL
ConservativeCFL
```

## Simulation end

```@docs
AbstractSimulationEnd
FixedSimulationEnd
SteadyStateEnd
```

## Coriolis parameter

```@docs
AbstractCoriolisParameter
CoriolisParameter0D
CoriolisParameter2D
```

