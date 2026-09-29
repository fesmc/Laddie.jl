# What does the raised step look like?  Counts the StableHLO ops that tell a
# stencil raised to shifted slices (fast) from one raised to gathers (slow), and
# measures how many cores XLA keeps busy over a 200-step traced loop.
#
#   RBACKEND=cpu taskset -c 0-7 julia --project=benchmark benchmark/reactant/hlo_ops.jl
#
# WG=1 compares against Laddie's (32, 8) workgroup (see common.jl).  The module
# is written to step_<backend>.mlir in the working directory.
include("common.jl")
rb = get(ENV, "RBACKEND", "cpu")
Reactant.set_default_backend(rb)

sim = initial_sim(640, 320)
rs = to_reactant(sim)
hlo = string(@code_hlo raise = true rstep!(rs))
write("step_$rb.mlir", hlo)
for op in ("stablehlo.slice", "stablehlo.gather", "stablehlo.transpose", "stablehlo.pad",
    "stablehlo.concatenate", "stablehlo.select", "stablehlo.dynamic_slice")
    # generic ("op") and pretty (op) printing forms
    println(rpad(op, 28), count("\"$op\"", hlo) + count("$op ", hlo))
end

cputime() = sum(parse.(Int, split(read("/proc/self/stat", String))[14:15])) / 100
n = ConcreteRNumber(200)
f = @compile raise = true rstep_for!(rs, n)
f(rs, n)
Reactant.synchronize(rs.model.D.present)
t0, c0 = time(), cputime()
f(rs, n)
Reactant.synchronize(rs.model.D.present)
t1, c1 = time(), cputime()
println("200 steps: wall $(round(t1 - t0, digits = 2)) s, cpu $(round(c1 - c0, digits = 2)) s ",
    "(~$(round((c1 - c0) / (t1 - t0), digits = 1)) cores busy)")
