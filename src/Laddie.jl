module Laddie
using KernelAbstractions
using DocStringExtensions
const KA = KernelAbstractions

include("entrainment.jl")
include("melting.jl")
include("convection.jl")
include("openboundary.jl")
include("groundingline.jl")
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
export build_model,
    build_isomip, build_laddie_mask, ice_base_depth, bed_elevation,
    fill_ocean_holes!, fill_shelf_holes!, fill_small_shelf_patches!,
    fill_small_grounded_patches!,
    run!, meltstats, to_backend

export AbstractEntrainment, HollandEntrainment, GasparEntrainment, LambertEntrainment
export AbstractMelting, FixedGamTMelting, TurbulentGamTMelting, PrescribedMelting
export AbstractConvectionScheme, ClampDensity, ResetToAmbient, RelaxToAmbient
export AbstractMaximumLayerThickness, AbsoluteMaxLayerThickness, RelativeMaxLayerThickness, TopographicMaxLayerThickness

export AbstractOpenBoundary, ZeroGradientInflow, NoInflow
export AbstractGroundingLineBC, FreeSlipGL, NoSlipGL
export AbstractIceSlopeGradient, PyGradient, JlGradient
export AbstractTimeStepper, FixedDt, AdaptiveDt
export AbstractCFL, ConservativeCFL, ExactCFL
export AbstractSimulationEnd, FixedSimulationEnd, SteadyStateEnd
export AbstractForcing, ISOMIPForcing, LinearForcing, Linear2Forcing
export TanhForcing, FileForcing, ProfileForcing

end # module Laddie
