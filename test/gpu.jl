# GPU tests: each case is run with CUDABackend and cross-compared with the CPU result.
# Included only when a functional CUDA device is present.

@testset "ISOMIP+ warm cavity (GPU): matches CPU" begin
    m_c = build_isomip(CPU();       nx=20, ny=10, isomipcond=:warm)
    m_g = build_isomip(gpu_backend; nx=20, ny=10, isomipcond=:warm)
    @test all(isfinite, Array(m_g.model.melt))
    @test all(Array(m_g.model.melt)[m_c.model.tmask .> 0] .>= 0)
    @test Array(m_g.model.melt) ≈ m_c.model.melt
end

@testset "ISOMIP+ cold cavity (GPU): matches CPU" begin
    m_c = build_isomip(CPU();       nx=20, ny=10, isomipcond=:cold)
    m_g = build_isomip(gpu_backend; nx=20, ny=10, isomipcond=:cold)
    @test all(isfinite, Array(m_g.model.melt))
    @test Array(m_g.model.melt) ≈ m_c.model.melt
end

@testset "run! GPU: advances state and matches CPU" begin
    m_c = build_isomip(CPU();       nx=20, ny=10, isomipcond=:warm)
    m_g = build_isomip(gpu_backend; nx=20, ny=10, isomipcond=:warm)
    D0_g = copy(Array(m_g.model.D.present))
    run!(m_c; days=0.5, verbose=false)
    run!(m_g; days=0.5, verbose=false)
    @test Array(m_g.model.D.present) != D0_g
    @test all(isfinite, Array(m_g.model.D.present))
    @test all(isfinite, Array(m_g.model.melt))
    @test Array(m_g.model.D.present) ≈ m_c.model.D.present
    @test Array(m_g.model.melt)      ≈ m_c.model.melt
    # CFL monitor reductions run on the device and match the CPU value.
    @test Laddie._cfl_number(m_g) ≈ Laddie._cfl_number(m_c)
end

@testset "Partial/free slip (GPU): matches CPU" begin
    bgpu = BoundaryConditions(; grounding_line = PartialSlipGL(0.5), land = FreeSlipLand())
    m_c = build_isomip(CPU();       nx = 20, ny = 10, isomipcond = :warm, boundary = bgpu)
    m_g = build_isomip(gpu_backend; nx = 20, ny = 10, isomipcond = :warm, boundary = bgpu)
    run!(m_c; days = 0.5, verbose = false)
    run!(m_g; days = 0.5, verbose = false)
    @test Array(m_g.model.melt)      ≈ m_c.model.melt
    @test Array(m_g.model.V.present) ≈ m_c.model.V.present
end

@testset "ISOMIP+ warm (GPU Float32): matches CPU Float32" begin
    m_c = build_isomip(CPU();       FT=Float32, nx=20, ny=10, isomipcond=:warm)
    m_g = build_isomip(gpu_backend; FT=Float32, nx=20, ny=10, isomipcond=:warm)
    run!(m_c; days=0.5, verbose=false)
    run!(m_g; days=0.5, verbose=false)
    @test all(isfinite, Array(m_g.model.D.present))
    @test all(isfinite, Array(m_g.model.melt))
    _, mn_c, _ = meltstats(m_c)
    _, mn_g, _ = meltstats(m_g)
    @test abs(Float64(mn_g) - Float64(mn_c)) / Float64(mn_c) < 1e-3
end

@testset "AdaptiveDt (GPU): controller + re-bootstrap run on device" begin
    # The CFL reductions, worst-case startup rescue, and re-bootstrap
    # must all be GPU-safe; assert a clean completion in bounds.
    m_g = build_isomip(gpu_backend; nx = 20, ny = 10, isomipcond = :warm,
                       tstep = AdaptiveDt())
    run!(m_g; days = 1.0, verbose = false)
    @test all(isfinite, Array(m_g.model.D.present)) && all(isfinite, Array(m_g.model.melt))
    @test 1.0 <= m_g.clock.dt <= 1000.0
end
