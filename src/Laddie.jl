module Laddie
using KernelAbstractions
using DocStringExtensions
using ProgressMeter
const KA = KernelAbstractions

include("entrainment.jl")
include("melting.jl")
include("convection.jl")
include("frazil.jl")
include("conduction.jl")
include("eos.jl")
include("max_layer_thickness.jl")
include("coriolis.jl")
include("kinks.jl")
include("boundary_conditions.jl")
include("viscosity.jl")
include("advection.jl")
include("timestepping.jl")
include("simulationend.jl")
include("forcing.jl")
include("ocean_sources.jl")
include("geometry.jl")
include("io.jl")
include("variable.jl")
include("utils.jl")
include("grid.jl")
include("state.jl")
include("cache.jl")
include("params.jl")
include("model.jl")
include("simulation.jl")

include("physics.jl")
include("numerics.jl")
include("stencils.jl")
include("backend.jl")
include("api.jl")
include("build.jl")
include("reproductions.jl")
include("show.jl")

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

export AbstractOpenOceanBC, ZeroGradientInflow, NoInflow
export AbstractGroundingLineBC, NoSlipGL, FreeSlipGL, PartialSlipGL
export AbstractLandBC, NoSlipLand, FreeSlipLand, PartialSlipLand
export AbstractGapsBC, SinkGapsBC, ConnectedGapsBC
export AbstractWallAdvection, SlipScaledWallAdvection, NoWallAdvection
export AbstractLateralViscosity, PrescribedLateralViscosity, NonlinearLateralViscosity
export AbstractMomentumAdvection, CentredMomentumAdvection, UpstreamMomentumAdvection
export AbstractTracerForm, AdvectiveTracerForm, FluxTracerForm
export AbstractLaplacianWeights, PresentLaplacianWeights, PastLaplacianWeights
export AbstractFrontPressure, FullDepthGradient, TruncatedDepthGradient, ZeroGradientFront
export AbstractIceSlopeGradient, PyGradient, JlGradient
export AbstractTimeStepper, FixedDt, AdaptiveDt
export AbstractCFL, ConservativeCFL, ExactCFL
export AbstractSimulationEnd, FixedSimulationEnd, SteadyStateEnd
export AbstractOceanForcing, OceanForcing1D, PerBasinOceanForcing1D, ISOMIPForcing
export front_ocean_forcing, front_sources, nearest_seed, regrid_ocean, fill_ocean!
export AbstractIceForcing, PrescribedIceForcing
export CavityForcing

end # module Laddie
