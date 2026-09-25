module Laddie
using KernelAbstractions
using DocStringExtensions
using ProgressMeter
const KA = KernelAbstractions

include("entrainment.jl")
include("melting.jl")
include("convection.jl")
include("coriolis.jl")
include("boundary_conditions.jl")
include("viscosity.jl")
include("advection.jl")
include("timestepping.jl")
include("simulationend.jl")
include("forcing.jl")
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

export AbstractEntrainment, HollandEntrainment, GasparEntrainment, LambertEntrainment
export AbstractMelting,
    FixedGamTMelting, TurbulentGamTMelting, UStarGamTMelting, PrescribedMelting
export AbstractConvectionScheme, ClampDensity, ResetToAmbient, RelaxToAmbient
export AbstractCoriolisParameter, CoriolisParameter0D, CoriolisParameter2D
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
export AbstractLaplacianWeights, PresentLaplacianWeights, PastLaplacianWeights
export AbstractFrontPressure, FullDepthGradient, TruncatedDepthGradient
export AbstractIceSlopeGradient, PyGradient, JlGradient
export AbstractTimeStepper, FixedDt, AdaptiveDt
export AbstractCFL, ConservativeCFL, ExactCFL
export AbstractSimulationEnd, FixedSimulationEnd, SteadyStateEnd
export AbstractOceanForcing, OceanForcing1D, ISOMIPForcing
export AbstractIceForcing, PrescribedIceForcing
export CavityForcing

end # module Laddie
