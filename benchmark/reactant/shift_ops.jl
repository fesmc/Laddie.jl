# How does Reactant raise a single neighbour read?  One micro-kernel per access
# pattern (x-shift, y-shift, diagonal; periodic wrap or interior-only), compiled
# alone on the CPU target, with its StableHLO op counts.  Arrays are [x, y] as in
# Laddie, so x is the contiguous (column-major) axis.
#
#   julia --project=benchmark benchmark/reactant/shift_ops.jl
include("common.jl")
using Printf
Reactant.set_default_backend("cpu")

const NX, NY = 162, 82

xp(i, N) = ifelse(i == N, 1, i + 1)   # Laddie's periodic wrap (`_xp1`)
xm(i, N) = ifelse(i == 1, N, i - 1)
xpmod(i, N) = mod1(i + 1, N)          # same wrap, written with mod1

# Full-range kernels (ndrange = size(out)).
@kernel function k_x!(out, @Const(a), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = a[xp(i, Nx), j]
end
@kernel function k_y!(out, @Const(a), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = a[i, xp(j, Ny)]
end
@kernel function k_diag!(out, @Const(a), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = a[xp(i, Nx), xm(j, Ny)]
end
@kernel function k_x_mod!(out, @Const(a), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = a[xpmod(i, Nx), j]
end
@kernel function k_diag_mod!(out, @Const(a), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = a[xpmod(i, Nx), mod1(j - 1, Ny)]
end
# Full range, x-neighbour clamped at the edge instead of wrapped (the border ring
# is inactive, so the value read there is always masked out).
xpc(i, N) = min(i + 1, N)
xmc(i, N) = max(i - 1, 1)
@kernel function k_x_clamp!(out, @Const(a), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = a[xpc(i, Nx), j]
end
@kernel function k_diag_clamp!(out, @Const(a), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = a[xpc(i, Nx), xmc(j, Ny)]
end
@kernel function k_diag_clampx!(out, @Const(a), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = a[xpc(i, Nx), xm(j, Ny)]
end
# Interior kernels (ndrange = size(out) .- 2, offset by one): no wrap at all.
@kernel function k_x_int!(out, @Const(a), Nx, Ny)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1
    @inbounds out[i, j] = a[i+1, j]
end
@kernel function k_diag_int!(out, @Const(a), Nx, Ny)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1
    @inbounds out[i, j] = a[i+1, j-1]
end

run_k(k, nd, out, a) = (k(RB())(out, a, NX, NY; ndrange = nd); nothing)

ops = ("gather", "transpose", "slice", "concatenate", "dynamic_update_slice", "pad")
count_op(hlo, op) = count("\"stablehlo.$op\"", hlo) + count("stablehlo.$op ", hlo)
@printf("%-14s %-9s", "kernel", "range")
foreach(op -> @printf(" %8s", first(op, 8)), ops)
println()
out = Reactant.to_rarray(zeros(NX, NY))
a = Reactant.to_rarray(rand(NX, NY))
for (k, name, nd) in (
    (k_x!, "x-shift", (NX, NY)), (k_y!, "y-shift", (NX, NY)), (k_diag!, "diagonal", (NX, NY)),
    (k_x_mod!, "x mod1", (NX, NY)), (k_diag_mod!, "diag mod1", (NX, NY)),
    (k_x_clamp!, "x clamp", (NX, NY)), (k_diag_clamp!, "diag clamp", (NX, NY)),
    (k_diag_clampx!, "diag clampx", (NX, NY)),
    (k_x_int!, "x-shift", (NX - 2, NY - 2)), (k_diag_int!, "diagonal", (NX - 2, NY - 2)),
)
    hlo = string(@code_hlo raise = true run_k(k, nd, out, a))
    @printf("%-14s %-9s", name, nd == (NX, NY) ? "full" : "interior")
    foreach(op -> @printf(" %8d", count_op(hlo, op)), ops)
    println()
end
