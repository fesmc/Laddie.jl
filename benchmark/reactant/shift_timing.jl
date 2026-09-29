# What does each raising of a neighbour read cost?  A 5-point Laplacian
# ping-ponged between two buffers in a 200-step traced loop, Float32 on the GPU:
#   wrap      Laddie's periodic `ifelse` wrap on both axes (x reads → gathers)
#   interior  interior launch, plain `i ± 1` everywhere (slices + update-slice)
#   copy      out = a, the memory-bandwidth floor
#
#   julia --project=benchmark benchmark/reactant/shift_timing.jl
include("common.jl")
using Printf, Statistics
Reactant.set_default_backend(get(ENV, "RBACKEND", "gpu"))

xp(i, N) = ifelse(i == N, 1, i + 1)
xm(i, N) = ifelse(i == 1, N, i - 1)

@kernel function lap_wrap!(out, @Const(a), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = a[xp(i, Nx), j] + a[xm(i, Nx), j] + a[i, xp(j, Ny)] +
                          a[i, xm(j, Ny)] - 4 * a[i, j]
end
@kernel function lap_interior!(out, @Const(a), Nx, Ny)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1
    @inbounds out[i, j] = a[i+1, j] + a[i-1, j] + a[i, j+1] + a[i, j-1] - 4 * a[i, j]
end
@kernel function copy_k!(out, @Const(a), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = a[i, j]
end

function steps!(k, nd, a, b, n)
    Nx, Ny = size(a)
    @trace track_numbers = false for _ = 1:n
        k(RB())(b, a, Nx, Ny; ndrange = nd)
        k(RB())(a, b, Nx, Ny; ndrange = nd)
    end
    return nothing
end

for (nx, ny) in ((642, 322), (1282, 642))
    for (name, k, interior) in (("wrap", lap_wrap!, false), ("interior", lap_interior!, true),
        ("copy", copy_k!, false))
        a = Reactant.to_rarray(rand(Float32, nx, ny) .* 1f-3)
        b = Reactant.to_rarray(zeros(Float32, nx, ny))
        nd = interior ? (nx - 2, ny - 2) : (nx, ny)
        n = ConcreteRNumber(100)
        f = @compile raise = true steps!(k, nd, a, b, n)
        run() = (f(k, nd, a, b, n); Reactant.synchronize(a))
        run()
        t = median([(@elapsed run()) for _ = 1:7]) / 200   # 2 launches per iteration
        hlo = string(@code_hlo raise = true steps!(k, nd, a, b, n))
        g = count("\"stablehlo.gather\"", hlo) + count("stablehlo.gather ", hlo)
        d = count("stablehlo.dynamic_update_slice", hlo)
        @printf("%5d×%-4d %-9s %8.1f µs/launch   (gather %d, update_slice %d)\n",
            nx, ny, name, 1e6t, g, d)
    end
end
