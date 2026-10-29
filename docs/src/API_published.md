# Published setups

Aliases for the runs of earlier LADDIE versions: `Params<Setup>v1|v2` returns the
[`Params`](@ref) of a run, and `Simulation<Setup>v1|v2` builds the whole run from the
geometry it left behind under `papers/` (not part of the repository). Where a paper and
the code that produced its results disagree, the values follow the code and the
docstring says how the paper differs. LADDIE v2 runs on an unstructured mesh with
forward–backward Runge–Kutta time stepping, so its aliases match its parameters, not its
numerics.

| Setup | Paper | Parameters | Simulation |
|-------|-------|------------|------------|
| Crosson–Dotson | [Lambert et al. (2023)](@cite lambert_modelling_2023) | [`ParamsCrossonDotsonv1`](@ref) | [`SimulationCrossonDotsonv1`](@ref) |
| Filchner–Ronne | [Lambert et al. (2023)](@cite lambert_modelling_2023) | [`ParamsFRISv1`](@ref) | [`SimulationFRISv1`](@ref) |
| Pine Island | [Lambert et al. (2023)](@cite lambert_modelling_2023) | [`ParamsPIGv1`](@ref) | [`SimulationPIGv1`](@ref) |
| Amundsen Sea, v1 | [Lambert et al. (2026)](@cite lambert_one-layer_2026) | [`ParamsASEv1`](@ref) | [`SimulationASEv1`](@ref), [`SimulationASEv1Theoretical`](@ref) |
| ISOMIP+ Ocean1 | [Lambert et al. (2026)](@cite lambert_one-layer_2026) | [`ParamsISOMIPv2`](@ref) | [`SimulationISOMIPv2`](@ref) |
| pan-Antarctic | [Lambert et al. (2026)](@cite lambert_one-layer_2026) | [`ParamsAISv2`](@ref) | — |
| Amundsen Sea, v2 | [Lambert et al. (2026)](@cite lambert_one-layer_2026) | [`ParamsASEv2`](@ref) | [`SimulationASEv2`](@ref) |
| Pine Island, 120 m | [Lambert et al. (2026)](@cite lambert_one-layer_2026) | [`ParamsPIGv2`](@ref) | — |
| MISOMIP+ (coupled) | [Lambert et al. (2026)](@cite lambert_one-layer_2026) | [`ParamsMISOMIPv2`](@ref) | [`SimulationMISOMIPv2`](@ref) |
| ice-shelf gaps (coupled) | [Jesse et al. (2026)](@cite jesse_ice-sheet_2026) | [`ParamsMISOMIPGapsv2`](@ref) | [`SimulationMISOMIPGapsv2`](@ref) |

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
