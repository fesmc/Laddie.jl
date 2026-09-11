module Laddie
using KernelAbstractions
using DocStringExtensions
const KA = KernelAbstractions

include("entrainment.jl")
include("melting.jl")
include("convection.jl")
include("coriolis.jl")
include("boundary_conditions.jl")
include("viscosity.jl")
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

include("physics.jl")
include("numerics.jl")
include("stencils.jl")
include("backend.jl")
include("api.jl")
include("build.jl")
include("show.jl")

export Model, Grid, State, Cache, Params, RunConfig, DebugConfig
export build_isomip, build_laddie_mask, ice_base_depth, bed_elevation,
    fill_ocean_holes!, fill_shelf_holes!, fill_small_shelf_patches!,
    fill_small_grounded_patches!,
    run!, meltstats, to_backend

export AbstractEntrainment, HollandEntrainment, GasparEntrainment, LambertEntrainment
export AbstractMelting, FixedGamTMelting, TurbulentGamTMelting, PrescribedMelting
export AbstractConvectionScheme, ClampDensity, ResetToAmbient, RelaxToAmbient
export AbstractCoriolisParameter, CoriolisParameter0D, CoriolisParameter2D
export AbstractMaxLayerThickness, AbsoluteMaxLayerThickness, RelativeMaxLayerThickness, TopographicMaxLayerThickness
export AbstractDomainCropping, NoDomainCropping, MinRectangleDomainCropping
export AbstractPreprocess, FillOceanHolesPreprocess, FillShelfHolesPreprocess,
    FillSmallShelfPatchesPreprocess, FillSmallGroundedPatchesPreprocess

export AbstractOpenOceanBC, ZeroGradientInflow, NoInflow
export AbstractGroundingLineBC, FreeSlipGL, NoSlipGL
export AbstractLandBC, FreeSlipLand, NoSlipLand
export AbstractGapsBC, SinkGapsBC, ConnectedGapsBC
export AbstractLateralViscosity, PrescribedLateralViscosity, NonlinearLateralViscosity
export AbstractFrontPressure, FullDepthGradient, TruncatedDepthGradient
export AbstractIceSlopeGradient, PyGradient, JlGradient
export AbstractTimeStepper, FixedDt, AdaptiveDt
export AbstractCFL, ConservativeCFL, ExactCFL
export AbstractSimulationEnd, FixedSimulationEnd, SteadyStateEnd
export AbstractOceanForcing, OceanForcing1D, ISOMIPForcing
export AbstractIceForcing, PrescribedIceForcing
export CavityForcing

end # module Laddie
