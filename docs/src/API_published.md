# Published setups

Aliases for the runs of earlier LADDIE versions: `Params<Setup>v1|v2` returns the
[`Params`](@ref) of a run, and any keyword overrides a field, as in `ParamsFRISv1(; H_min = 3.0)`.
`Simulation<Setup>v1|v2(backend = CPU(); FT, datadir, params, simulation_kwargs...)` builds
the whole run from the geometry it left behind under `datadir` (default `papers/` in the
package root, not part of the repository). `params` defaults to the run's `Params` alias, and
any other keyword goes to [`Simulation`](@ref), overriding the setup's `dt` and `stop`.

Where a paper and the code that produced its results disagree, the values follow the code
(the run's own configuration file), and the [differences](@ref "Where the papers differ") are
listed below. LADDIE v2 runs on an unstructured mesh with forward–backward Runge–Kutta time
stepping, and Laddie.jl on a C-grid with leapfrog, so the v2 aliases match v2's parameters,
not its numerics. Every alias pins `integrator = LeapfrogIntegrator(; nu = 0.8)`.

| Setup | Paper | Parameters | Simulation |
|-------|-------|------------|------------|
| Crosson–Dotson | [Lambert et al. (2023)](@cite lambert_modelling_2023) | [`ParamsCrossonDotsonv1`](@ref) | [`SimulationCrossonDotsonv1`](@ref) |
| Filchner–Ronne | [Lambert et al. (2023)](@cite lambert_modelling_2023) | [`ParamsFRISv1`](@ref) | [`SimulationFRISv1`](@ref) |
| Pine Island | [Lambert et al. (2023)](@cite lambert_modelling_2023) | [`ParamsPIGv1`](@ref) | [`SimulationPIGv1`](@ref) |
| Amundsen Sea, v1 | [Lambert et al. (2026)](@cite lambert_one-layer_2026) | [`ParamsASEv1`](@ref) | [`SimulationASEv1`](@ref), [`SimulationASEv1Theoretical`](@ref) |
| ISOMIP+ Ocean1 | [Lambert et al. (2026)](@cite lambert_one-layer_2026) | [`ParamsISOMIPv2`](@ref) | [`SimulationISOMIPv2`](@ref) |
| pan-Antarctic | [Lambert et al. (2026)](@cite lambert_one-layer_2026) | [`ParamsAISv2`](@ref) | — (BedMachine v3 geometry and ISMIP6 forcing not bundled) |
| Amundsen Sea, v2 | [Lambert et al. (2026)](@cite lambert_one-layer_2026) | [`ParamsASEv2`](@ref) | [`SimulationASEv2`](@ref) |
| Pine Island, 120 m | [Lambert et al. (2026)](@cite lambert_one-layer_2026) | [`ParamsPIGv2`](@ref) | — (REMA/Bedmap v3 geometry not bundled) |
| MISOMIP+ (coupled) | [Lambert et al. (2026)](@cite lambert_one-layer_2026) | [`ParamsMISOMIPv2`](@ref) | [`SimulationMISOMIPv2`](@ref) |
| ice-shelf gaps (coupled) | [Jesse et al. (2026)](@cite jesse_ice-sheet_2026) | [`ParamsMISOMIPGapsv2`](@ref) | [`SimulationMISOMIPGapsv2`](@ref) |

## Parameters

**LADDIE v1.** Every v1 config shares [Jenkins (1991)](@cite jenkins_onedimensional_1991)
transfer coefficients ([`TurbulentGamTMelting`](@ref)), `C_d_top = 1.1e-3`, μ = 2.5,
`max_detrainment = 0.5`, `v_cut = 1.414`, `H_init = 10`, `dT_init = 0`, f = −1.37 × 10⁻⁴ s⁻¹,
the prescribed viscosity `A_h`, centred momentum advection, advective tracer and momentum
forms, and v1's ice-slope stencil [`CentredIceSlope`](@ref).

| Run | Source | `A_h` | `K_h` | `H_min` | `u_tide` |
|-----|--------|------:|------:|--------:|---------:|
| Crosson–Dotson | Table 2 of the paper, Δx = 500 m | 25 | 25 | 2.8 | 0.01 |
| Filchner–Ronne | Table 2 of the paper, Δx = 1 km | 50 | 50 | 4.4 | 0.1 |
| Pine Island | App. C: "the same model tuning as Crosson–Dotson" | 25 | 25 | 2.8 | 0.01 |
| Amundsen Sea, v1 | `results/laddie1/ASE/config_ASE.toml` | 20 | 4 | 1 | 0.01 |

**LADDIE v2.** Every v2 run shares UPSY's constants (ρ₀ = 1027 kg m⁻³, L = 3.335 × 10⁵ J kg⁻¹),
μ = 2.5, `max_detrainment = 1e-3` m s⁻¹, `u_tide = 0.01`, f = −1.37 × 10⁻⁴ s⁻¹, the
shear-scaled [`NonlinearLateralViscosity`](@ref) (its coefficient is also the wall viscosity,
hence `A_h` equal to it), upstream momentum advection, flux-form tracers, an absolute cap
`H_max` on the layer thickness, and a buoyancy floor that bounds both the convection and the
entrainment's density contrast. The values come from each run's config,
`results/<run>/<run>.cfg` of [Lambert et al. (2026)](@cite lambert_one-layer_2026).

| Run | Melting | `C_d_top` | viscosity | `K_h` | `H_min` | `H_init` | `v_cut` | `H_max` (m) | buoyancy floor (kg m⁻³) |
|-----|---------|----------:|----------:|------:|--------:|---------:|--------:|------------:|------------------------:|
| ISOMIP+ Ocean1 | [`UStarGamTMelting`](@ref), Γ_T = 2.78 × 10⁻² | 2.5e-3 | 0.1 | 1 | 1 | 1 | 1 | 1500 | 0.005 |
| pan-Antarctic¹ | [Jenkins (1991)](@cite jenkins_onedimensional_1991) | 1.1e-3 | 0.1 | 1 | 1 | 4 | 1 | 10 000 | 0.02 |
| Amundsen Sea | [Jenkins (1991)](@cite jenkins_onedimensional_1991) | 1.1e-3 | 1 | 4 | 1 | 1 | 1.4 | 1500 | 0.005 |
| Pine Island, 120 m | [Jenkins (1991)](@cite jenkins_onedimensional_1991) | 1.1e-3 | 0.1 | 0.1 | 1 | 1 | 1.4 | 1500 | 0.005 |
| MISOMIP+ | [`UStarGamTMelting`](@ref), Γ_T = 2.03 × 10⁻² | 2.5e-3 | 0.1 | 1 | 1 | 1 | 1 | 1500 | 0.005 |
| ice-shelf gaps² | [`UStarGamTMelting`](@ref), Γ_T = 3 × 10⁻² | 2.5e-3 | 10 | 10 | 1 | 1 | 1 | 1500 | 0.005 |

¹ From `laddie2-description-main/Antarctica.cfg`: the run left no config of its own, and this
is the one whose 3 km triangles match the paper.
² From Table A1 of [Jesse et al. (2026)](@cite jesse_ice-sheet_2026), no run config being
bundled; the values the table leaves out (`H_init`, `v_cut`, `H_max`, buoyancy floor) are those
of MISOMIP+.

## Simulations

All files are under `datadir`. The v1 runs use v1's walls (its `slip = 1`: no slip in the
viscosity, [`V1WallAdvection`](@ref)), zero-gradient inflow at the ice front and
`T_ice_farfield = -25 °C`. The v2 runs use v2's walls (slip factor 1, no
momentum across walls), zero-gradient inflow and UPSY's default ice temperature, 270 K
(−3.15 °C); their shelf is the ice that floats in v2's output remapped from its mesh to a
regular grid.

| Simulation | Geometry | Δx | Ocean forcing | `dt` (s) | days |
|------------|----------|---:|---------------|---------:|-----:|
| [`SimulationCrossonDotsonv1`](@ref) | `lambert-2023/data/CrossDots_0.5_tanh_Tdeep0.4_ztcl-500_050.nc` | 500 m | v1 'tanh': surface freezing point (S0 = 34) to 0.4 °C, thermocline at −500 m, scale 250 m, `drho0 = 0.01` (fits the file's ambient fields to 1e−4 °C) | 120 | 50 |
| [`SimulationFRISv1`](@ref) | `lambert-2023/data/FRIS_1.0_linear_S134.8_T1-2.3_720.nc` | 1 km | v1 'linear': surface freezing point to −2.3 °C, S from 34.5 to 34.8 psu at −2000 m (fits the file to round-off) | 240 | 720 |
| [`SimulationPIGv1`](@ref) | `lambert-2023/data/PIG.nc`; no mask stored, rebuilt by flotation (ρ_ice = 917, ρ_sw = 1027) | 500 m | v1 'tanh' as Crosson–Dotson, but 1.0 °C at depth, thermocline at −450 m | 120 | 30 |
| [`SimulationASEv1`](@ref) | `lambert-2026/results/laddie1/ASE/output_000020.nc` | 1 km cells, Δx = 500 m in the derivatives | v1 'tanh', surface freezing point (S0 = 34) to 1.0 °C, thermocline at −450 m, scale 100 m | 30 | 20 |
| [`SimulationASEv1Theoretical`](@ref) | as `SimulationASEv1` | 1 km | as `SimulationASEv1` | 30 | 20 |
| [`SimulationISOMIPv2`](@ref) | `lambert-2026/results/isomip_ocean1/laddie_output_grid.nc` | 2 km | ISOMIP+ warm, sampled every 100 m as v2 does | 90 | 20 |
| [`SimulationASEv2`](@ref) | `lambert-2026/results/ASE/laddie_output_grid.nc` (the run used 1.2 km triangles) | 1 km | UFEMISM 'TANH': surface freezing point to 1.0 °C, thermocline at 450 m, scale 100 m, every 100 m | 30 | 20 |
| [`SimulationMISOMIPv2`](@ref) | last slice of `lambert-2026/results/misomip_spinup_1/main_output_ANT_grid.nc` (the coupled run used 1 km triangles on the shelf) | 2 km | UFEMISM 'TANH' to 1.5 °C | 60 | 10 |
| [`SimulationMISOMIPGapsv2`](@ref) | as `SimulationMISOMIPv2`: Jesse et al.'s spun-up geometry is on the mesh only, and this one's grounding line also sits near x = 450 km | 2 km | UFEMISM 'TANH' to −1, 0 or 1 °C (`ocean = :ocean1/2/3`), which keeps the density profile identical across the three | 100 | 20 |

**The two v1 Amundsen Sea runs.** The config reads a 500 m geometry with `coarsen = 2`, but
the v1 code sets the grid spacing before coarsening and never updates it. The run was
therefore on a 1 km grid with Δx = 500 m in every derivative, and so is
[`SimulationASEv1`](@ref) (`Grid(…, 500, 500)` on the 1 km cells).
[`SimulationASEv1Theoretical`](@ref) is the consistent 1 km run.

The MISOMIP+ simulations are the initial LADDIE spin-up of coupled UFEMISM–LADDIE runs, run
standalone as a sanity check.

## Where the papers differ

- **All v1 runs.** μ = 2.5, as in every bundled v1 config; [Lambert et al. (2023,
  Table 1)](@cite lambert_modelling_2023) state μ = 0.5. The layer starts at the ambient
  temperature (`dT_init = 0`, as in the configs); the paper starts it 0.1 °C below.
  `max_detrainment = 0.5` and `v_cut = 1.414` are code values the paper does not mention.
- **Crosson–Dotson.** The run behind the stored output lasted 50 days (averaged over days
  45–50); the paper reports a 30-day equilibration.
- **Filchner–Ronne.** The paper builds the 1 km grid by averaging 2 × 2 BedMachine cells. If
  that was done with the v1 code's `coarsen`, the run used Δx = 500 m in its derivatives (as
  the v1 Amundsen Sea run); the stored file cannot tell, so Δx = 1 km is used.
- **Pine Island, v1.** The paper gives only the deep temperature and the thermocline depth;
  the thermocline scale (250 m), `S0` and `drho0` are those of the Crosson–Dotson profile it
  says this one adapts.
- **Amundsen Sea, v1 and v2.** The configs set f = −1.37 × 10⁻⁴ s⁻¹;
  [Lambert et al. (2026)](@cite lambert_one-layer_2026) state −1.34 × 10⁻⁴ s⁻¹ for both model
  versions. The paper gives no viscosity or diffusivity for the v1 run.
- **All v2 runs.** UPSY hard-wires ρ₀ = 1027 kg m⁻³ and L = 3.335 × 10⁵ J kg⁻¹, used here;
  [Lambert et al. (2026, Table 1)](@cite lambert_one-layer_2026) state 1028 and 3.34 × 10⁵.
  The ice temperature of the standalone runs is UPSY's default 270 K, not v1's −25 °C.
- **ISOMIP+, pan-Antarctic.** The configs set the viscosity coefficient to 0.1 and `K_h = 1`;
  Table 1 of the paper lists 1 and 4. The pan-Antarctic buoyancy floor of 0.02 kg m⁻³ is four
  times that of the other runs.
- **Pine Island, 120 m.** The paper says the configuration is that of its Table 1
  (viscosity 1, `K_h = 4`); the config uses 0.1 and 0.1.
- **MISOMIP+.** The config sets 1.5 °C at depth; the paper describes the forcing of Jesse et
  al. (2025), with 1.2 °C.
- **Ice-shelf gaps.** The paper fixes the surface at −1.8 °C; UFEMISM 'TANH' puts it at the
  freezing point of 34 psu, −1.865 °C.

## Reference

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
