# Does the step compile under Reactant, and does it match the KA CPU path?
#
#   julia --project=benchmark benchmark/reactant/feasibility.jl
#
# RBACKEND=gpu targets the GPU (default cpu); TRACK_NUMBERS=1 traces the scalars
# in the loop too; WG=1 keeps Laddie's (32, 8) workgroup (see common.jl).
include("common.jl")
Reactant.set_default_backend(get(ENV, "RBACKEND", "cpu"))

sim = initial_sim(80, 40)
rsim = to_reactant(sim)
println("array type: ", typeof(rsim.model.D.present))

t = @elapsed (f1 = @compile raise = true rstep!(rsim))
println("compile rstep!: $(round(t, digits = 1)) s")
ref = deepcopy(sim)
for _ = 1:5
    time_step!(ref)
    f1(rsim)
end
println("5 single steps, max rel err vs KA CPU: ", maxrelerr(prognostics(rsim), prognostics(ref)))

n = ConcreteRNumber(10)
t = @elapsed (fN = @compile raise = true rstep_for!(rsim, n))
println("compile rstep_for! (track_numbers = $TRACK_NUMBERS): $(round(t, digits = 1)) s")
fN(rsim, n)
for _ = 1:10
    time_step!(ref)
end
println("+10 traced-loop steps, max rel err: ", maxrelerr(prognostics(rsim), prognostics(ref)))
