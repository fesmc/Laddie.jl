using Laddie
using Test
using KernelAbstractions
import CUDA

# Access CGridProto as a submodule of Laddie
const CGridProto = Laddie.CGridProto

backend = CPU()
FT = Float64
Nx = Ny = 64
Lx = Ly = FT(Nx)   # dx = dy = 1

# Detect GPU: use CUDABackend if a functional CUDA device is present.
const gpu_backend = CUDA.functional() ? CUDA.CUDABackend() : nothing

@testset "Laddie.jl" begin

    @testset "CGridProto: location-dispatched operators" begin
        g = CGridProto.CGrid(backend, FT, Nx, Ny, Lx, Ly; topology=(CGridProto.Periodic, CGridProto.Periodic))

        c = CGridProto.Field(CGridProto.Center, CGridProto.Center, g)
        for i in axes(c.data, 1), j in axes(c.data, 2); c.data[i, j] = i; end
        @test CGridProto.∂x(10, 10, c) ≈ 1.0

        u = CGridProto.Field(CGridProto.Face, CGridProto.Center, g)
        for i in axes(u.data, 1), j in axes(u.data, 2); u.data[i, j] = i; end
        @test CGridProto.∂x(10, 10, u) ≈ 1.0

        cy = CGridProto.Field(CGridProto.Center, CGridProto.Center, g)
        for i in axes(cy.data, 1), j in axes(cy.data, 2); cy.data[i, j] = j; end
        @test CGridProto.∂y(10, 10, cy) ≈ 1.0
    end

    @testset "CGridProto: periodic halo fill" begin
        g = CGridProto.CGrid(backend, FT, Nx, Ny, Lx, Ly; topology=(CGridProto.Periodic, CGridProto.Periodic))
        f = CGridProto.Field(CGridProto.Center, CGridProto.Center, g)
        for i in 1:Nx, j in 1:Ny; f.data[g.Hx+i, g.Hy+j] = i + 100j; end
        CGridProto.fill_halo!(f)
        @test f.data[g.Hx+Nx+1, g.Hy+5] == f.data[g.Hx+1, g.Hy+5]
        @test f.data[1,          g.Hy+5] == f.data[g.Hx+Nx, g.Hy+5]
        @test f.data[g.Hx+5, g.Hy+Ny+1] == f.data[g.Hx+5, g.Hy+1]
    end

    @testset "CGridProto: Euler advection conserves mass (periodic)" begin
        g = CGridProto.CGrid(backend, FT, Nx, Ny, Lx, Ly; topology=(CGridProto.Periodic, CGridProto.Periodic))
        c = CGridProto.Field(CGridProto.Center, CGridProto.Center, g)
        CGridProto.set!(c, (x, y) -> exp(-((x - Lx/2)^2 + (y - Ly/2)^2) / (2 * (Lx/12)^2)))
        u = CGridProto.set_constant!(CGridProto.Field(CGridProto.Face,   CGridProto.Center, g), 1.0)
        v = CGridProto.set_constant!(CGridProto.Field(CGridProto.Center, CGridProto.Face,   g), 0.5)

        m0 = sum(CGridProto.interior(c))
        CGridProto.advect_euler!(c, u, v, 0.4, 200)
        m1 = sum(CGridProto.interior(c))
        @test isapprox(m1, m0; rtol=1e-10)
        @test all(isfinite, CGridProto.interior(c))
    end

    @testset "CGridProto: zero velocity leaves field unchanged" begin
        g = CGridProto.CGrid(backend, FT, Nx, Ny, Lx, Ly; topology=(CGridProto.Periodic, CGridProto.Periodic))
        c = CGridProto.set!(CGridProto.Field(CGridProto.Center, CGridProto.Center, g),
                            (x, y) -> sin(2π * x / Lx))
        saved = copy(c.data)
        uz = CGridProto.set_constant!(CGridProto.Field(CGridProto.Face,   CGridProto.Center, g), 0.0)
        vz = CGridProto.set_constant!(CGridProto.Field(CGridProto.Center, CGridProto.Face,   g), 0.0)
        CGridProto.advect_euler!(c, uz, vz, 0.4, 50)
        @test c.data == saved
    end

    @testset "CGridProto: leapfrog + RA advection runs and conserves mass" begin
        g = CGridProto.CGrid(backend, FT, Nx, Ny, Lx, Ly; topology=(CGridProto.Periodic, CGridProto.Periodic))
        c0 = CGridProto.set!(CGridProto.Field(CGridProto.Center, CGridProto.Center, g),
                             (x, y) -> exp(-((x - Lx/2)^2 + (y - Ly/2)^2) / (2 * (Lx/12)^2)))
        u = CGridProto.set_constant!(CGridProto.Field(CGridProto.Face,   CGridProto.Center, g), 1.0)
        v = CGridProto.set_constant!(CGridProto.Field(CGridProto.Center, CGridProto.Face,   g), 0.5)

        m0 = sum(CGridProto.interior(c0))
        cf = CGridProto.advect_leapfrog(c0, u, v, 0.4, 200; ν=0.05)
        @test all(isfinite, CGridProto.interior(cf))
        @test isapprox(sum(CGridProto.interior(cf)), m0; rtol=1e-8)
    end

    @testset "ISOMIP+ warm cavity: build and basic physics" begin
        # Small grid (nx=20, ny=10) for a fast smoke test
        m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:warm)

        @test size(m.tmask) == (12, 22)   # ny+2 × nx+2
        @test all(isfinite, m.melt)
        @test all(m.melt[m.tmask .> 0] .>= 0)   # melt rate non-negative under ice

        mx, mn, sp = meltstats(m)
        @test isfinite(mx) && isfinite(mn) && isfinite(sp)
        @test mx >= mn >= 0
    end

    @testset "ISOMIP+ cold cavity: build" begin
        m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:cold)
        @test all(isfinite, m.melt)
    end

    @testset "run! advances model state" begin
        m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:warm)
        D0 = copy(m.D.present)
        run!(m; days=0.5, verbose=false)
        # D should have changed
        @test m.D.present != D0
        @test all(isfinite, m.D.present)
        @test all(isfinite, m.melt)
    end

    @testset "Float32 vs Float64: mean melt within 1%" begin
        m64 = build_isomip(CPU(); FT=Float64, nx=20, ny=10, isomipcond=:warm)
        m32 = build_isomip(CPU(); FT=Float32, nx=20, ny=10, isomipcond=:warm)
        run!(m64; days=2.0, verbose=false)
        run!(m32; days=2.0, verbose=false)
        _, mn64, _ = meltstats(m64)
        _, mn32, _ = meltstats(m32)
        @test isfinite(mn64) && mn64 > 0
        @test isfinite(mn32)
        @test abs(Float64(mn32) - mn64) / mn64 < 0.01
    end

    @testset "Conservation: D equation exact over one step" begin
        m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:warm)
        Laddie.advance_leapfrog!(m)
        D_past = copy(m.D.past)
        src    = copy((m.convD .+ m.melt .+ m.nentr) .* m.tmask)
        Laddie.leapfrog_step!(m, 2)
        @test m.D.future ≈ D_past .+ src .* (2 * m.dt)
    end

    @testset "Conservation: D ≥ minD after 1-day run" begin
        m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:warm)
        run!(m; days=1.0, verbose=false)
        active = m.tmask .> 0
        @test all(m.D.present[active] .>= m.minD - 1e-10)
    end

    # -------------------------------------------------------------------------
    # GPU tests: mirror each test above using CUDABackend, then cross-compare
    # with the CPU result to verify bit-level agreement.
    # Skipped entirely when no CUDA device is present.
    # -------------------------------------------------------------------------
    if gpu_backend !== nothing

        @testset "CGridProto (GPU): location-dispatched operators match CPU" begin
            # Initialise on CPU, transfer to GPU to avoid scalar writes on device.
            g_c = CGridProto.CGrid(CPU(),        FT, Nx, Ny, Lx, Ly; topology=(CGridProto.Periodic, CGridProto.Periodic))
            g_g = CGridProto.CGrid(gpu_backend,  FT, Nx, Ny, Lx, Ly; topology=(CGridProto.Periodic, CGridProto.Periodic))

            c_c = CGridProto.Field(CGridProto.Center, CGridProto.Center, g_c)
            c_g = CGridProto.Field(CGridProto.Center, CGridProto.Center, g_g)
            for i in axes(c_c.data, 1), j in axes(c_c.data, 2); c_c.data[i, j] = i; end
            copyto!(c_g.data, c_c.data)
            @test CUDA.@allowscalar(CGridProto.∂x(10, 10, c_g)) ≈ 1.0

            u_c = CGridProto.Field(CGridProto.Face, CGridProto.Center, g_c)
            u_g = CGridProto.Field(CGridProto.Face, CGridProto.Center, g_g)
            for i in axes(u_c.data, 1), j in axes(u_c.data, 2); u_c.data[i, j] = i; end
            copyto!(u_g.data, u_c.data)
            @test CUDA.@allowscalar(CGridProto.∂x(10, 10, u_g)) ≈ 1.0

            cy_c = CGridProto.Field(CGridProto.Center, CGridProto.Center, g_c)
            cy_g = CGridProto.Field(CGridProto.Center, CGridProto.Center, g_g)
            for i in axes(cy_c.data, 1), j in axes(cy_c.data, 2); cy_c.data[i, j] = j; end
            copyto!(cy_g.data, cy_c.data)
            @test CUDA.@allowscalar(CGridProto.∂y(10, 10, cy_g)) ≈ 1.0
        end

        @testset "CGridProto (GPU): periodic halo fill matches CPU" begin
            g_c = CGridProto.CGrid(CPU(),       FT, Nx, Ny, Lx, Ly; topology=(CGridProto.Periodic, CGridProto.Periodic))
            g_g = CGridProto.CGrid(gpu_backend, FT, Nx, Ny, Lx, Ly; topology=(CGridProto.Periodic, CGridProto.Periodic))
            f_c = CGridProto.Field(CGridProto.Center, CGridProto.Center, g_c)
            f_g = CGridProto.Field(CGridProto.Center, CGridProto.Center, g_g)
            for i in 1:Nx, j in 1:Ny; f_c.data[g_c.Hx+i, g_c.Hy+j] = i + 100j; end
            copyto!(f_g.data, f_c.data)
            CGridProto.fill_halo!(f_c)
            CGridProto.fill_halo!(f_g)
            @test Array(f_g.data) == f_c.data
        end

        @testset "CGridProto (GPU): Euler advection matches CPU" begin
            g_c = CGridProto.CGrid(CPU(),       FT, Nx, Ny, Lx, Ly; topology=(CGridProto.Periodic, CGridProto.Periodic))
            g_g = CGridProto.CGrid(gpu_backend, FT, Nx, Ny, Lx, Ly; topology=(CGridProto.Periodic, CGridProto.Periodic))
            c_c = CGridProto.set!(CGridProto.Field(CGridProto.Center, CGridProto.Center, g_c),
                                  (x, y) -> exp(-((x - Lx/2)^2 + (y - Ly/2)^2) / (2 * (Lx/12)^2)))
            u_c = CGridProto.set_constant!(CGridProto.Field(CGridProto.Face,   CGridProto.Center, g_c), 1.0)
            v_c = CGridProto.set_constant!(CGridProto.Field(CGridProto.Center, CGridProto.Face,   g_c), 0.5)
            c_g = CGridProto.Field(CGridProto.Center, CGridProto.Center, g_g)
            u_g = CGridProto.Field(CGridProto.Face,   CGridProto.Center, g_g)
            v_g = CGridProto.Field(CGridProto.Center, CGridProto.Face,   g_g)
            copyto!(c_g.data, c_c.data)
            copyto!(u_g.data, u_c.data)
            copyto!(v_g.data, v_c.data)

            CGridProto.advect_euler!(c_c, u_c, v_c, 0.4, 200)
            CGridProto.advect_euler!(c_g, u_g, v_g, 0.4, 200)
            @test all(isfinite, Array(CGridProto.interior(c_g)))
            @test Array(CGridProto.interior(c_g)) ≈ CGridProto.interior(c_c)
        end

        @testset "CGridProto (GPU): leapfrog + RA advection matches CPU" begin
            g_c = CGridProto.CGrid(CPU(),       FT, Nx, Ny, Lx, Ly; topology=(CGridProto.Periodic, CGridProto.Periodic))
            g_g = CGridProto.CGrid(gpu_backend, FT, Nx, Ny, Lx, Ly; topology=(CGridProto.Periodic, CGridProto.Periodic))
            c0_c = CGridProto.set!(CGridProto.Field(CGridProto.Center, CGridProto.Center, g_c),
                                   (x, y) -> exp(-((x - Lx/2)^2 + (y - Ly/2)^2) / (2 * (Lx/12)^2)))
            u_c = CGridProto.set_constant!(CGridProto.Field(CGridProto.Face,   CGridProto.Center, g_c), 1.0)
            v_c = CGridProto.set_constant!(CGridProto.Field(CGridProto.Center, CGridProto.Face,   g_c), 0.5)
            c0_g = CGridProto.Field(CGridProto.Center, CGridProto.Center, g_g)
            u_g  = CGridProto.Field(CGridProto.Face,   CGridProto.Center, g_g)
            v_g  = CGridProto.Field(CGridProto.Center, CGridProto.Face,   g_g)
            copyto!(c0_g.data, c0_c.data)
            copyto!(u_g.data,  u_c.data)
            copyto!(v_g.data,  v_c.data)

            cf_c = CGridProto.advect_leapfrog(c0_c, u_c, v_c, 0.4, 200; ν=0.05)
            cf_g = CGridProto.advect_leapfrog(c0_g, u_g, v_g, 0.4, 200; ν=0.05)
            @test all(isfinite, Array(CGridProto.interior(cf_g)))
            @test Array(CGridProto.interior(cf_g)) ≈ CGridProto.interior(cf_c)
        end

        @testset "ISOMIP+ warm cavity (GPU): matches CPU" begin
            m_c = build_isomip(CPU();       nx=20, ny=10, isomipcond=:warm)
            m_g = build_isomip(gpu_backend; nx=20, ny=10, isomipcond=:warm)
            @test all(isfinite, Array(m_g.melt))
            @test all(Array(m_g.melt)[m_c.tmask .> 0] .>= 0)
            @test Array(m_g.melt) ≈ m_c.melt
        end

        @testset "ISOMIP+ cold cavity (GPU): matches CPU" begin
            m_c = build_isomip(CPU();       nx=20, ny=10, isomipcond=:cold)
            m_g = build_isomip(gpu_backend; nx=20, ny=10, isomipcond=:cold)
            @test all(isfinite, Array(m_g.melt))
            @test Array(m_g.melt) ≈ m_c.melt
        end

        @testset "run! GPU: advances state and matches CPU" begin
            m_c = build_isomip(CPU();       nx=20, ny=10, isomipcond=:warm)
            m_g = build_isomip(gpu_backend; nx=20, ny=10, isomipcond=:warm)
            D0_g = copy(Array(m_g.D.present))
            run!(m_c; days=0.5, verbose=false)
            run!(m_g; days=0.5, verbose=false)
            @test Array(m_g.D.present) != D0_g
            @test all(isfinite, Array(m_g.D.present))
            @test all(isfinite, Array(m_g.melt))
            @test Array(m_g.D.present) ≈ m_c.D.present
            @test Array(m_g.melt)      ≈ m_c.melt
        end

        @testset "ISOMIP+ warm (GPU Float32): matches CPU Float32" begin
            m_c = build_isomip(CPU();       FT=Float32, nx=20, ny=10, isomipcond=:warm)
            m_g = build_isomip(gpu_backend; FT=Float32, nx=20, ny=10, isomipcond=:warm)
            run!(m_c; days=0.5, verbose=false)
            run!(m_g; days=0.5, verbose=false)
            @test all(isfinite, Array(m_g.D.present))
            @test all(isfinite, Array(m_g.melt))
            _, mn_c, _ = meltstats(m_c)
            _, mn_g, _ = meltstats(m_g)
            @test abs(Float64(mn_g) - Float64(mn_c)) / Float64(mn_c) < 1e-3
        end

    end # gpu_backend !== nothing

end
