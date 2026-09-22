# Which kernels raise to gathers and transposes?  Records every kernel launch of
# one CPU time step (kernel + arguments), then compiles each launch on its own on
# the Reactant CPU target and counts its StableHLO ops.
#
#   julia --project=benchmark benchmark/reactant/kernel_ops.jl
#
# DUMP=<kernel name> also writes that kernel's raised module to <name>.mlir.
include("common.jl")
using Printf

Reactant.set_default_backend("cpu")

# --- 1. record the launches of one step on the CPU ---------------------------
const LAUNCHES = Any[]
const RECORDING = Ref(false)
# CPU-only method: more specific than Laddie's `launch!(kernel!, A, args...)`.
function Laddie.launch!(kernel!, A::Matrix, args...)
    RECORDING[] && push!(LAUNCHES, (kernel!, size(A), deepcopy(args)))
    backend = Laddie._launch_backend(CPU())
    kernel!(backend, Laddie._workgroup(backend))(args...; ndrange = size(A))
    return nothing
end

sim = initial_sim(160, 80)
RECORDING[] = true
time_step!(sim)
RECORDING[] = false

# --- 2. compile each distinct kernel alone -----------------------------------
torr(a::AbstractArray{<:AbstractFloat}) = Reactant.to_rarray(a)
torr(a) = a
run_kernel(kernel!, nd, args...) = (kernel!(RB())(args...; ndrange = nd); nothing)

ops = ("stablehlo.gather", "stablehlo.transpose", "stablehlo.slice",
    "stablehlo.concatenate", "stablehlo.dynamic_update_slice")
count_op(hlo, op) = count("\"$op\"", hlo) + count("$op ", hlo)

@printf("%-36s %5s %6s %9s %6s %7s %5s\n", "kernel", "calls", "gather", "transpose",
    "slice", "concat", "dus")
seen = Dict{Any,Int}()
for (k, _, _) in LAUNCHES
    seen[k] = get(seen, k, 0) + 1
end
done = Set{Any}()
for (k, nd, args) in LAUNCHES
    k in done && continue
    push!(done, k)
    name = replace(string(k), r"\(.*" => "")
    rargs = map(torr, args)
    hlo = try
        string(@code_hlo raise = true run_kernel(k, nd, rargs...))
    catch err
        @printf("%-36s  FAILED: %s\n", name, sprint(showerror, err)[1:min(end, 80)])
        continue
    end
    c = map(op -> count_op(hlo, op), ops)
    @printf("%-36s %5d %6d %9d %6d %7d %5d\n", name, seen[k], c...)
    haskey(ENV, "DUMP") && occursin(Regex(ENV["DUMP"]), name) && write("$(name).mlir", hlo)
end
