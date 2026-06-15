module Laddie
using KernelAbstractions
using DocStringExtensions
const KA = KernelAbstractions

const spy = 365.25 * 24 * 3600   # seconds per year

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


export Model, Grid, State, Cache, Params, RunConfig
export build_model,
    build_isomip, build_laddie_mask, ice_base_depth, run!, meltstats, to_backend

export AbstractEntrainmentParam, HollandEntrainment, GasparEntrainment
export AbstractMeltParam, FixedGamT, TurbulentGamT
export AbstractConvectionScheme, ClampDensity, ResetToAmbient, RelaxToAmbient

export AbstractOpenBoundary, ZeroGradientInflow, NoInflow
export AbstractGroundingLineBC, FreeSlipGL, NoSlipGL
export AbstractTimeStepper, FixedDt, AdaptiveDt
export AbstractSimulationEnd, FixedSimulationEnd, SteadyStateEnd
export AbstractForcing, ISOMIPForcing, LinearForcing, Linear2Forcing
export TanhForcing, FileForcing, ProfileForcing

end # module Laddie
