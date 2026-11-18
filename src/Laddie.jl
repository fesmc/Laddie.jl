module Laddie
using Adapt: Adapt
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using JLD2: jldopen, jldsave
using KernelAbstractions: KernelAbstractions, @Const, @index, @kernel, CPU
using NCDatasets: NCDatasets, NCDataset, defDim, defVar
using Printf: @sprintf
using ProgressMeter: Progress, finish!, update!
using TOML: TOML
const KA = KernelAbstractions

include("physics/entrainment.jl")
include("physics/melting.jl")
include("physics/convection.jl")
include("physics/frazil.jl")
include("physics/conduction.jl")
include("physics/eos.jl")
include("physics/max_layer_thickness.jl")
include("physics/coriolis.jl")
include("numerics/kinks.jl")
include("numerics/tracer_bounds.jl")
include("bcs/walls.jl")
include("bcs/open_ocean.jl")
include("bcs/gaps.jl")
include("bcs/boundary_conditions.jl")
include("numerics/viscosity.jl")
include("numerics/advection.jl")
include("numerics/timestepping.jl")
include("bcs/forcing.jl")
include("bcs/ocean_sources.jl")
include("utils/preprocessing.jl")
include("utils/io.jl")
include("simulation/variables.jl")
include("utils/utils.jl")
include("simulation/grid.jl")
include("simulation/state.jl")
include("simulation/cache.jl")
include("simulation/params.jl")
include("simulation/model.jl")
include("simulation/simulation.jl")

include("physics/secondary_fields.jl")
include("numerics/leapfrog.jl")
include("numerics/stencils.jl")
include("utils/backend.jl")
include("utils/diagnostics.jl")
include("simulation/run.jl")
include("simulation/autodiff.jl")
include("simulation/presets.jl")
include("utils/show.jl")

export Model, Grid, State, Cache, Params, BoundaryConditions
export CPU   # KernelAbstractions' CPU backend, the default everywhere
export Simulation, Clock, OutputConfig, DebugConfig, time_step!
export build_isomip,
    build_laddie_mask,
    ice_base_depth,
    bed_elevation,
    fill_ocean_holes!,
    fill_shelf_holes!,
    fill_small_shelf_patches!,
    fill_small_grounded_patches!,
    run!,
    meltstats,
    to_backend,
    ReactantBackend,
    integrate!,
    reactant_compile,
    trace_parameters,
    adaptive_schedule,
    DtSchedule

# Published LADDIE v1/v2 setups (reproductions.jl)
export ParamsCrossonDotsonv1, ParamsFRISv1, ParamsPIGv1, ParamsASEv1
export SimulationCrossonDotsonv1,
    SimulationFRISv1, SimulationPIGv1, SimulationASEv1, SimulationASEv1Theoretical
export ParamsISOMIPv2, ParamsAISv2, ParamsASEv2, ParamsPIGv2, ParamsMISOMIPv2, ParamsMISOMIPGapsv2
export SimulationISOMIPv2, SimulationASEv2, SimulationMISOMIPv2, SimulationMISOMIPGapsv2

export AbstractEntrainment, HollandEntrainment, GasparEntrainment
export AbstractMelting,
    FixedGamTMelting, TurbulentGamTMelting, UStarGamTMelting, TwoEquationMelting, PrescribedMelting
export AbstractConvectionScheme, ClampDensity, ResetToAmbient, RelaxToAmbient
export AbstractFrazilIce, NoFrazilIce, RelaxedFrazilIce
export AbstractEquationOfState, LinearEOS, RoquetEOS
export AbstractIceConduction, HollandJenkinsConduction, MeltOnlyConduction, NoConduction,
    LinearProfileConduction
export AbstractCoriolisParameter, CoriolisParameter0D, CoriolisParameter2D
export AbstractKinks, ExactKinks, SmoothKinks
export AbstractTracerBounds, NoTracerBounds, TracerBounds
export AbstractMaxLayerThickness,
    NoMaxLayerThickness,
    AbsoluteMaxLayerThickness,
    RelativeMaxLayerThickness,
    TopographicMaxLayerThickness
export AbstractDomainCropping, NoDomainCropping, MinRectangleDomainCropping
export AbstractPreprocess,
    FillOceanHolesPreprocess,
    FillShelfHolesPreprocess,
    FillSmallShelfPatchesPreprocess,
    FillSmallGroundedPatchesPreprocess,
    MarkGapsPreprocess

export AbstractOpenOceanBC, ZeroGradientInflow, NoInflow, OutflowOnly
export AbstractGroundingLineBC, NoSlipGL, FreeSlipGL, PartialSlipGL
export AbstractLandBC, NoSlipLand, FreeSlipLand, PartialSlipLand
export AbstractGapsBC, SinkGapsBC, ConnectedGapsBC, LeakyGapsBC
export AbstractWallAdvection, SlipScaledWallAdvection, NoWallAdvection, V1WallAdvection
export AbstractLateralViscosity, PrescribedLateralViscosity, NonlinearLateralViscosity
export AbstractMomentumAdvection, CentredMomentumAdvection, UpstreamMomentumAdvection
export AbstractTracerForm, AdvectiveTracerForm, FluxTracerForm
export AbstractMomentumForm, AdvectiveMomentumForm, FluxMomentumForm
export AbstractClosureLevel, PresentClosures, PastClosures
export AbstractLaplacianWeights, PresentLaplacianWeights, PastLaplacianWeights
export AbstractFrontPressure, FullDepthGradient, TruncatedDepthGradient, ZeroGradientFront
export AbstractIceSlopeGradient, PyGradient, JlGradient
export AbstractTimeStepper, FixedDt, AdaptiveDt
export AbstractCFL, ConservativeCFL, ExactCFL
export AbstractSimulationEnd, FixedSimulationEnd, SteadyStateEnd
export AbstractOceanForcing, OceanForcing1D, PerBasinOceanForcing1D, RampedOceanForcing, ISOMIPForcing
export front_ocean_forcing, FrontBlend, front_sources, nearest_seed, regrid_ocean, fill_ocean!
export AbstractIceForcing, PrescribedIceForcing
export CavityForcing

end # module Laddie
