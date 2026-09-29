# Experiment: does dropping the periodic wrap let Reactant raise a stencil to
# slices?  Negative result (2026-09-22; see laddie-roadmap/reactant.md §0).
# Superseded: `src/` now launches every time-step stencil over the interior
# (`launch_interior!`), so HALO=0 and HALO=1 run the same kernel.
# Replaces `_lapT_kernel!` (tracer Laplacian, called for T and S) on the Reactant
# path only, by a variant launched over the interior cells with plain `i ± 1`
# neighbours (the border ring is never active, so it needs no wrap).
#
#   HALO=0 julia --project=benchmark benchmark/reactant/halo_lapT.jl   # baseline
#   HALO=1 julia --project=benchmark benchmark/reactant/halo_lapT.jl   # interior variant
#
# Prints the op counts of one step, the error against KA CPU after 50 steps, and
# the Float32 GPU time per step.  The border of `out` is not written by the
# variant; it holds whatever the shared `lap` buffer held, and every consumer
# multiplies it by `tmask = 0` there.
include("common.jl")
using Statistics, Printf

const HALO = get(ENV, "HALO", "0") == "1"

@kernel function _lapT_interior_kernel!(
    out,
    @Const(var),
    @Const(D0jp),
    @Const(D0jm),
    @Const(D0ip),
    @Const(D0im),
    @Const(tmask),
    dy2,
    dx2,
)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1
    @inbounds begin
        flux_N = D0jp[i, j] * (var[i, j+1] - var[i, j]) * tmask[i, j+1] / dy2
        flux_S = D0jm[i, j] * (var[i, j-1] - var[i, j]) * tmask[i, j-1] / dy2
        flux_E = D0ip[i, j] * (var[i+1, j] - var[i, j]) * tmask[i+1, j] / dx2
        flux_W = D0im[i, j] * (var[i-1, j] - var[i, j]) * tmask[i-1, j] / dx2
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

if HALO
    function Laddie.laplace_T(out, m, var::Reactant.AnyTracedRArray)
        nx, ny = size(var)
        _lapT_interior_kernel!(RB())(
            out, var, m.D0jp, m.D0jm, m.D0ip, m.D0im, m.tmask, m.dy^2, m.dx^2;
            ndrange = (nx - 2, ny - 2),
        )
        return out
    end
end

function opcounts(rs)
    hlo = string(@code_hlo raise = true rstep!(rs))
    c(op) = count("\"$op\"", hlo) + count("$op ", hlo)
    return (; gather = c("stablehlo.gather"), slice = c("stablehlo.slice"),
        transpose = c("stablehlo.transpose"))
end

println("HALO = $HALO")

# Op counts and accuracy on the CPU target (Float64).
Reactant.set_default_backend("cpu")
sim = initial_sim(640, 320)
println("ops per step (640×320): ", opcounts(to_reactant(sim)))
ref, chk = deepcopy(sim), to_reactant(sim)
n = ConcreteRNumber(50)
@compile(raise = true, rstep_for!(chk, n))(chk, n)
for _ = 1:50
    time_step!(ref)
end
@printf("max rel err vs KA CPU after 50 steps (F64): %.2e\n", maxrelerr(prognostics(chk), prognostics(ref)))

# Float32 GPU timing: 100-step traced loop, median of 5.
Reactant.set_default_backend("gpu")
for (nx, ny) in ((640, 320), (1280, 640))
    rs = to_reactant(initial_sim(nx, ny; FT = Float32))
    k = ConcreteRNumber(100)
    f = @compile raise = true rstep_for!(rs, k)
    run() = (f(rs, k); Reactant.synchronize(rs.model.D.present))
    run()
    t = median([(@elapsed run()) for _ = 1:5]) / 100
    @printf("GPU F32 %d×%d: %.3f ms/step\n", nx, ny, 1e3t)
end
