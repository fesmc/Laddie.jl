# Does the raised step shard cleanly?  Moves a model onto a mesh of fake CPU devices
# (`ReactantBackend(; mesh)`: every matrix split over the mesh, everything else
# replicated), compiles one leapfrog step with raised kernels and counts the
# collectives in the partitioned XLA program: a stencil code should need only
# `collective-permute` (halo exchange) and a few `all-reduce`s; an `all-gather` of a
# field means some op fetches the whole array on every device.  Then checks a
# sharded run against KA.
#
#   julia --project=benchmark benchmark/reactant/sharding.jl
#
# NDEV (default 4) fake devices; MESH = y | x | xy (2-D mesh, NDEV = 4); FUSION =
# kernel | xla (default: what `:auto` picks on the mesh); SCEN = default | visc
# (NonlinearLateralViscosity) | adv (UpstreamMomentumAdvection) | mixing (both); NX,
# NY: ISOMIP interior (the grid adds a ring; each split axis must divide by its
# device count).  FUNC = advU | advV | lapU | lapV compiles that term alone instead
# of the step.  The partitioned module is written to sharded_<MESH>_<SCEN>.hlo in the
# working directory.
const NDEV = parse(Int, get(ENV, "NDEV", "4"))
ENV["XLA_FLAGS"] = "--xla_force_host_platform_device_count=$NDEV"

using Laddie, Reactant, CUDA, KernelAbstractions
using Reactant: Sharding, ConcreteRNumber
const KA = KernelAbstractions
Reactant.set_default_backend("cpu")
const EXT = Base.get_extension(Laddie, :LaddieReactantExt)
EXT._const_workaround!()

const MESHKIND = get(ENV, "MESH", "y")
const SCEN = get(ENV, "SCEN", "default")
const NX, NY = parse(Int, get(ENV, "NX", "158")), parse(Int, get(ENV, "NY", "78"))

const BACKEND = if MESHKIND == "y"
    ReactantBackend(; mesh = Sharding.Mesh(collect(0:(NDEV-1)), (:p,)))
elseif MESHKIND == "x"
    ReactantBackend(; mesh = Sharding.Mesh(collect(0:(NDEV-1)), (:p,)), partition = (:p, nothing))
else
    ReactantBackend(; mesh = Sharding.Mesh(reshape(collect(0:3), 2, 2), (:a, :b)))
end
EXT.TRACING_FUSION[] = haskey(ENV, "FUSION") ? Symbol(ENV["FUSION"]) : EXT._fusion(BACKEND)

const SCHEMES = Dict(
    "default" => (;),
    "visc" => (; lateral_viscosity = NonlinearLateralViscosity()),
    "adv" => (; momentum_advection = UpstreamMomentumAdvection()),
    "mixing" => (; lateral_viscosity = NonlinearLateralViscosity(),
                 momentum_advection = UpstreamMomentumAdvection()),
)
scenario_kw = (; params = Params(; FT = Float32, SCHEMES[SCEN]...))
sim = build_isomip(CPU(); nx = NX, ny = NY, FT = Float32, isomipcond = :warm, scenario_kw...)
println("grid $(size(sim.model.melt)), mesh $MESHKIND ($NDEV devices), scenario $SCEN, fusion $(EXT.TRACING_FUSION[])")

rmodel = to_backend(sim.model, BACKEND)
println("D.present shards: ", unique(rmodel.D.present.sharding.device_to_array_slices))

# FUNC: what to compile, the whole step (default) or one term, to narrow a problem down.
const FUNC = get(ENV, "FUNC", "step")
const NU = 0.8f0
const FUNCS = Dict(
    "advU" => m -> Laddie.upwind_advection_U(m),
    "advV" => m -> Laddie.upwind_advection_V(m),
    "lapU" => m -> Laddie.laplace_U(m, m.lateral_viscosity),
    "lapV" => m -> Laddie.laplace_V(m, m.lateral_viscosity),
)
step!(m, dt) = (FUNC == "step" ? Laddie._step_model!(Laddie._stepping_view(m, dt, NU)) :
                FUNCS[FUNC](m); nothing)
nsteps!(m, dt, n) = (@trace track_numbers = false for _ = 1:n
    Laddie._step_model!(Laddie._stepping_view(m, dt, NU))
end; nothing)

dt = ConcreteRNumber(Float32(Laddie._primal(sim.clock.dt)))
t0 = time()
hlo = string(Reactant.@code_xla raise = true step!(rmodel, dt))
println("compiled in $(round(time() - t0, digits = 1)) s")
write("sharded_$(MESHKIND)_$(SCEN).hlo", hlo)

println("\npartitioned step, op counts (whole module):")
for op in ("all-gather", "collective-permute", "all-to-all", "all-reduce", "reduce-scatter",
           "gather", "dynamic-update-slice", "dynamic-slice", "fusion")
    n = count(Regex("\\s$(op)(-start)?\\("), hlo)
    println("  ", rpad(op, 22), n)
end

# Where the all-gathers come from: the shapes gathered and the source op names.
lines = filter(l -> occursin(r"\sall-gather(-start)?\(", l), split(hlo, '\n'))
if !isempty(lines)
    println("\nall-gathers (result shape ← metadata op_name):")
    for l in first(lines, 20)
        shape = match(r"=\s*(\S+)", l)
        name = match(r"op_name=\"([^\"]*)\"", l)
        println("  ", shape === nothing ? "?" : shape[1], "  ← ",
                name === nothing ? "-" : first(name[1], 120))
    end
end

# Correctness: sharded run vs the KA CPU run.
FUNC == "step" || exit()
n = 50
f = Reactant.compile(nsteps!, (rmodel, dt, ConcreteRNumber(n)); raise = true)
f(rmodel, dt, ConcreteRNumber(n))
kam = sim.model
Laddie.integrate!(kam, Laddie._primal(sim.clock.dt), n; nu = NU)
err = maximum(fieldnames(Laddie.State)) do v
    a, b = Array(getfield(rmodel.state, v).present), getfield(kam.state, v).present
    maximum(abs, a .- b) / max(maximum(abs, b), eps(Float32))
end
println("\n$n steps: max relative difference vs KA CPU = $err")
