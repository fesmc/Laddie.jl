# LADDIE versions

Three codes share the name LADDIE. **v1** ([Lambert et al. (2023)](@cite lambert_modelling_2023))
runs on a structured Arakawa C-grid. **v2** ([Lambert et al. (2026)](@cite lambert_one-layer_2026))
runs on an unstructured mesh, rewritten and integrated with the UFEMISM ice-sheet model. Laddie.jl
implements the **v2 physics on the v1 C-grid**: it keeps v1's structured grid and stencils, and takes
the physical choices that v2 made. Each choice that differs between the two versions is an option of
Laddie.jl; the [table below](#Options-in-each-version) lists them, with the default a plain run uses.

| aspect | LADDIE v1 | LADDIE v2 | Laddie.jl |
|---|---|---|---|
| language | Python | Fortran, parallel | Julia, [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl): CPU threads and GPUs |
| grid | structured Arakawa C-grid | unstructured triangular mesh, UFEMISM-integrated | structured C-grid, as v1 |
| time stepping | leapfrog with Robert–Asselin filter | forward–backward Runge–Kutta (FB-RK3) | leapfrog with Robert–Asselin filter, as v1; fixed or adaptive step (`AdaptiveTimeStepper`) |
| typical run time\* | 40 min | 5 min | 30 s |
| automatic differentiation | no | no | yes |

\* Best time achieved by each code: LADDIE v1 on one CPU, LADDIE v2 on 64 CPUs, Laddie.jl on one GPU.

## Options in each version

| option | LADDIE v1 | LADDIE v2 | Laddie.jl default |
|---|---|---|---|
| equation of state | `LinearEOS` | `LinearEOS` | `RoquetEOS`† |
| melting | `TurbulentGamTMelting` | `TurbulentGamTMelting` or `UStarGamTMelting`, per run | `TurbulentGamTMelting` |
| ice conduction | `HollandJenkinsIceConduction` | `HollandJenkinsIceConduction` | `MeltOnlyIceConduction`† |
| entrainment | `GasparEntrainment` | `GasparEntrainment` | `GasparEntrainment` |
| convection | `ResetToAmbientConvection` | `ClampDensityConvection` | `RelaxToAmbientConvection(3600)`† |
| frazil ice | `NoFrazilIce` | `NoFrazilIce` | `NoFrazilIce` |
| lateral viscosity | `PrescribedLateralViscosity` | `NonlinearLateralViscosity` | `NonlinearLateralViscosity(0.1)` |
| Coriolis parameter | `CoriolisParameter0D` | `CoriolisParameter0D` | `CoriolisParameter0D` |
| ice-shelf gaps | `SinkGapsBC` | `SinkGapsBC`, `ConnectedGapsBC` or `LeakyGapsBC`, per run | `SinkGapsBC` |
| momentum advection | `CentredMomentumAdvection` | `UpstreamMomentumAdvection` | `UpstreamMomentumAdvection` |
| tracer form | `AdvectiveTracerForm` | `FluxTracerForm` | `FluxTracerForm` |
| momentum form | `AdvectiveMomentumForm` | `FluxMomentumForm`\* | `FluxMomentumForm` |
| closure level | `PresentClosureLevel` | no counterpart (FB-RK3) | `PastClosureLevel`†, a field of `LeapfrogIntegrator` |
| Laplacian weights | `PastLaplacianWeights`\* | no counterpart | `PresentLaplacianWeights` |
| wall slip | `NoSlipAtGL`, `NoSlipAtLand` | `PartialSlipAtGL(1)`, `PartialSlipAtLand(1)` | `PartialSlipAtGL(1)`, `PartialSlipAtLand(1)` |
| wall momentum advection | `V1WallAdvection` | `NoWallAdvection` | `NoWallAdvection` |
| open-ocean boundary | `ZeroGradientOpenOceanBC` | `ZeroGradientOpenOceanBC` | `OutflowOnlyOpenOceanBC`† |
| ice-front pressure gradient | `V1FrontPressure` | `ZeroExteriorFrontPressure` | `ZeroGradientFrontPressure`† |
| maximum layer thickness | none that binds | `AbsoluteMaxLayerThickness` | `RelativeMaxLayerThickness(1, 500)`† |
| tracer bounds | none | none | `NoTracerBounds`\* |
| ice-base slope (`ice_slope`) | `CentredIceSlope` | no counterpart (mesh) | `MaskedIceSlope` |

\* The aliases below do not follow the version here: `Params<Setup>v1` has
`PresentLaplacianWeights`, `Params<Setup>v2` has `AdvectiveMomentumForm`, and both have
`TracerBounds()`, as Laddie.jl had them when the aliases were made; the v1 aliases also pin
`V1FrontPressure` and the v2 setups zero-gradient inflow at the ice front. Select the option of the
version explicitly to use it.

† Neither version's choice: the default is the setup of tuned pan-Antarctic runs. The nonlinear
equation of state carries cabbeling and thermobaricity, which raise the thermal expansion at the
depth of the deep grounding lines; the layer is capped at its water column, but never below
500 m, so that the cap binds only on the thickest layers, not where the draft nearly meets the bed
(a cap at 0.8 of the water column removed nearly all refreezing). Relaxing an
unstable layer to the ambient water over an hour replaces an instantaneous reset; conduction
stops where the base refreezes, since marine ice there sits near the freezing point
([Wiskandt & Jourdain 2025](@cite wiskandt_brief_2025)); the closures at the past level keep
strongly entraining, thin layers stable under the leapfrog (see
[Momentum form and closure level](numerics/momentum_form.md)). At the ice front, the zero lateral
gradient of the LADDIE v2 description paper ([Lambert et al. 2026](@cite lambert_one-layer_2026))
and an outflow-only valve keep the total melt of a warm ISOMIP+ run within 0.4 % and remove the
melt anomaly of the front row (see [Open-ocean boundary](bcs/open_ocean.md)).

## Using the versions

`Params<Setup>v1` and `Params<Setup>v2` return the parameters each published run used, and
`Simulation<Setup>v1` and `Simulation<Setup>v2` build the run itself. The v2 aliases match v2's
*parameters*: they select the v2 options of the [table above](#Options-in-each-version), with the exception noted there, but they run on Laddie.jl's C-grid
with leapfrog stepping, so they are not bit-for-bit reproductions. The
[reproduction page](generated/reproductions.md) compares both against the published output, and
the [published setups](API_published.md) list every alias.
