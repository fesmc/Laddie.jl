using Laddie
using Test
using KernelAbstractions
import CUDA

FT = Float64

# Detect GPU: use CUDABackend if a functional CUDA device is present.
const gpu_backend = CUDA.functional() ? CUDA.CUDABackend() : nothing

@testset verbose=true "Laddie.jl" begin

    @testset "Utility: _safe_div and index wrap-around helpers" begin
        @test Laddie._safe_div(6.0, 2.0) === 3.0
        @test Laddie._safe_div(1.0, 0.0) === 0.0   # zero denominator → zero
        @test Laddie._safe_div(0.0, 0.0) === 0.0   # both zero → zero
        # _west and _south wrap at boundary i/j == 1
        @test Laddie._west(1, 10)  == 10
        @test Laddie._west(5, 10)  == 4
        @test Laddie._south(1, 10) == 10
        @test Laddie._south(5, 10) == 4
        # _east and _north wrap at boundary i/j == N
        @test Laddie._east(10, 10) == 1
        @test Laddie._east(5,  10) == 6
        @test Laddie._north(10, 10) == 1
        @test Laddie._north(5,  10) == 6
    end

    @testset "build_model: arbitrary mask and draft" begin
        # Minimal 4×6 interior domain (6×8 with border ring):
        #   col 1 and 8 → boundary (1); cols 2–3 → grounded (2); cols 4–7 → shelf (3)
        nx_i, ny_i = 6, 4
        mask = zeros(Int, ny_i + 2, nx_i + 2)
        mask[1, :]   .= 1;   mask[end, :] .= 1
        mask[:, 1]   .= 1;   mask[:, end] .= 1
        mask[2:end-1, 2:3]   .= 2
        mask[2:end-1, 4:end-1] .= 3
        zb_raw = fill(-400.0, ny_i + 2, nx_i + 2)

        forcing = ISOMIPForcing(FT, :warm)
        params  = Params(; FT)
        m = build_model(mask, zb_raw, 2000.0, 2000.0, forcing, params; FT)

        @test size(m.tmask) == (ny_i + 2, nx_i + 2)
        @test all(isfinite, m.melt)
        @test all(m.melt[m.tmask .> 0] .>= 0)
        run!(m; days=0.5, verbose=false)
        @test all(isfinite, m.D.present)
        @test all(isfinite, m.melt)
    end

    @testset "Geometry ingestion: build_laddie_mask classification" begin
        # 2×3 interior domain:
        #   (1,1) h=600, bed=-500 → h_af = 600*917/1028 - 500 ≈ +35 → grounded (2)
        #   (1,2) h=400, bed=-500 → h_af = 400*917/1028 - 500 ≈ -143 → floating (3)
        #   (1,3) h=0,   bed=-200 → no ice → ocean (0)
        #   row 2: same pattern
        bed       = [-500.0  -500.0  -200.0;
                     -500.0  -500.0  -200.0]
        thickness = [ 600.0   400.0     0.0;
                      600.0   400.0     0.0]

        mask = build_laddie_mask(bed, thickness)
        @test size(mask) == (4, 5)           # (ny+2, nx+2) = (4, 5)
        # Border ring is all 1
        @test all(mask[1, :]   .== 1)
        @test all(mask[end, :] .== 1)
        @test all(mask[:, 1]   .== 1)
        @test all(mask[:, end] .== 1)
        # Interior cells
        @test mask[2, 2] == 2   # grounded
        @test mask[2, 3] == 3   # floating
        @test mask[2, 4] == 0   # ocean
        @test mask[3, 2] == 2
        @test mask[3, 3] == 3
        @test mask[3, 4] == 0

        # Mismatched sizes must throw
        @test_throws ArgumentError build_laddie_mask(bed, thickness[1:1, :])
    end

    @testset "Geometry ingestion: ice_base_depth values" begin
        bed       = [-500.0  -500.0  -200.0]
        thickness = [ 600.0   400.0     0.0]

        zb = ice_base_depth(bed, thickness)
        @test size(zb) == (3, 5)    # (1+2, 3+2)
        # Border zeros
        @test all(zb[1, :] .== 0.0)
        @test all(zb[end, :] .== 0.0)
        @test all(zb[:, 1]   .== 0.0)
        @test all(zb[:, end] .== 0.0)
        # Grounded: zb = bed
        @test zb[2, 2] ≈ -500.0
        # Floating: zb = -h * rho_ice/rho_sw
        @test zb[2, 3] ≈ -400.0 * 917.0 / 1028.0
        # Ocean: zb = 0
        @test zb[2, 4] ≈ 0.0
    end

    @testset "Geometry ingestion: end-to-end build_model from synthetic BedMachine" begin
        # 4×8 interior: cols 1-2 grounded, cols 3-8 floating
        ny_bm, nx_bm = 4, 8
        bed_bm = fill(-500.0, ny_bm, nx_bm)
        h_bm   = zeros(ny_bm, nx_bm)
        h_bm[:, 1:2] .= 600.0   # grounded: h_af = 600*917/1028 - 500 > 0
        h_bm[:, 3:8] .= 400.0   # floating: h_af = 400*917/1028 - 500 < 0
        mask_bm = build_laddie_mask(bed_bm, h_bm)
        zb_bm   = ice_base_depth(bed_bm, h_bm)

        @test count(==(2), mask_bm) == ny_bm * 2
        @test count(==(3), mask_bm) == ny_bm * 6

        forcing = ISOMIPForcing(FT, :warm)
        params  = Params(; FT)
        m = build_model(mask_bm, zb_bm, 2000.0, 2000.0, forcing, params; FT)
        @test all(isfinite, m.melt)
        @test all(m.melt[m.tmask .> 0] .>= 0)
        run!(m; days = 0.5, verbose = false)
        @test all(isfinite, m.D.present)
        @test all(isfinite, m.melt)
    end

    @testset "ProfileForcing: resampling, sorting, flat extrapolation" begin
        z_c = [-1000.0, -500.0, -100.0]
        T_c = [   1.0,     0.0,   -1.0]
        S_c = [  34.7,    34.2,   33.8]
        f = ProfileForcing(T_c, S_c, z_c; FT)
        @test f.dz == 1.0
        @test f.z0 == -5000.0
        @test f.z == FT.(-5000.0:1.0:-1.0)
        # Flat extrapolation below the deepest sample (z = -5000 < -1000)
        @test f.Tz[1] ≈ 1.0
        @test f.Sz[1] ≈ 34.7
        # Flat extrapolation above the shallowest sample (z = -1 > -100)
        @test f.Tz[end] ≈ -1.0
        @test f.Sz[end] ≈ 33.8
        # Linear interpolation at z = -750 (midway between -1000 and -500)
        k = findfirst(==(FT(-750.0)), f.z)
        @test f.Tz[k] ≈ 0.5
        @test f.Sz[k] ≈ 34.45

        # Descending input (CSV convention: surface first) gives the same result
        f_rev = ProfileForcing(reverse(T_c), reverse(S_c), reverse(z_c); FT)
        @test f_rev.Tz == f.Tz
        @test f_rev.Sz == f.Sz

        # Duplicate depths are tolerated (first occurrence kept)
        f_dup = ProfileForcing([1.0, 2.0, -1.0], [34.7, 34.6, 33.8],
                               [-1000.0, -1000.0, -100.0]; FT)
        @test all(isfinite, f_dup.Tz)

        # Length mismatch must throw
        @test_throws ArgumentError ProfileForcing(T_c[1:2], S_c, z_c; FT)
    end

    @testset "ProfileForcing: reproduces ISOMIPForcing from coarse samples" begin
        # The warm ISOMIP profile is linear in z, so 3 samples recover it exactly.
        isomip = ISOMIPForcing(FT, :warm)
        T_lin(z) = -1.9 + z * (1.0 - (-1.9)) / (-720.0)
        S_lin(z) = 33.8 + z * (34.7 - 33.8) / (-720.0)
        z_c  = [-5000.0, -720.0, -1.0]
        prof = ProfileForcing(T_lin.(z_c), S_lin.(z_c), z_c; FT)
        @test prof.Tz ≈ isomip.Tz
        @test prof.Sz ≈ isomip.Sz

        # Same domain, both forcings: initial melt fields must agree
        nx_i, ny_i = 6, 4
        mask = zeros(Int, ny_i + 2, nx_i + 2)
        mask[1, :]   .= 1;   mask[end, :] .= 1
        mask[:, 1]   .= 1;   mask[:, end] .= 1
        mask[2:end-1, 2:3]   .= 2
        mask[2:end-1, 4:end-1] .= 3
        zb_raw = fill(-400.0, ny_i + 2, nx_i + 2)

        m1 = build_model(mask, zb_raw, 2000.0, 2000.0, isomip, Params(; FT); FT)
        m2 = build_model(mask, zb_raw, 2000.0, 2000.0, prof,  Params(; FT); FT)
        @test m2.melt ≈ m1.melt
        run!(m2; days = 0.5, verbose = false)
        @test all(isfinite, m2.D.present)
        @test all(isfinite, m2.melt)
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

    @testset "Physical ordering: warm mean melt exceeds cold" begin
        mw = build_isomip(CPU(); nx=20, ny=10, isomipcond=:warm)
        mc = build_isomip(CPU(); nx=20, ny=10, isomipcond=:cold)
        _, mn_warm, _ = meltstats(mw)
        _, mn_cold, _ = meltstats(mc)
        @test mn_warm > mn_cold
    end

    @testset "TurbulentGamT: build and short run" begin
        params = Params(;
            FT,
            meltpar = TurbulentGamT(FT(13.8), FT(2432.0), FT(1.95e-6)),
            entpar  = GasparEntrainment(FT(2.5)),
            convpar = ResetToAmbient(FT(0.005)),
        )
        m = build_isomip(CPU(); FT, nx=20, ny=10, isomipcond=:warm, params)
        @test all(isfinite, m.melt)
        @test all(m.melt[m.tmask .> 0] .>= 0)
        run!(m; days=0.5, verbose=false)
        @test all(isfinite, m.D.present)
        @test all(isfinite, m.melt)
        @test all(m.melt[m.tmask .> 0] .>= 0)
    end

    @testset "HollandEntrainment: build and short run" begin
        params = Params(;
            FT,
            entpar  = HollandEntrainment(FT(0.01775)),
            meltpar = FixedGamT(FT(0.00018)),
            convpar = ResetToAmbient(FT(0.005)),
        )
        m = build_isomip(CPU(); FT, nx=20, ny=10, isomipcond=:warm, params)
        @test all(isfinite, m.melt)
        @test all(m.melt[m.tmask .> 0] .>= 0)
        run!(m; days=0.5, verbose=false)
        @test all(isfinite, m.D.present)
        @test all(isfinite, m.melt)
        @test all(m.melt[m.tmask .> 0] .>= 0)
    end

    @testset "ClampDensity convection scheme: build and short run" begin
        params = Params(; FT, convpar = ClampDensity(FT(0.005)))
        m = build_isomip(CPU(); FT, nx=20, ny=10, isomipcond=:warm, params)
        @test all(isfinite, m.melt)
        run!(m; days=0.5, verbose=false)
        @test all(isfinite, m.D.present)
        @test all(isfinite, m.melt)
    end

    @testset "RelaxToAmbient convection scheme: build and short run" begin
        params = Params(; FT, convpar = RelaxToAmbient(FT(10000.0)))
        m = build_isomip(CPU(); FT, nx=20, ny=10, isomipcond=:warm, params)
        @test all(isfinite, m.melt)
        run!(m; days=0.5, verbose=false)
        @test all(isfinite, m.D.present)
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

    @testset "Conservation: D ≥ minD after 1-day run (cold)" begin
        m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:cold)
        run!(m; days=1.0, verbose=false)
        active = m.tmask .> 0
        @test all(m.D.present[active] .>= m.minD - 1e-10)
    end

    @testset "I/O: NetCDF output, log, and JLD2 restart round-trip" begin
        tmpdir = mktempdir()
        rc = RunConfig(; name = "iotest", resultdir = tmpdir,
                       saveday = 0.5, diagday = 0.5, restday = 0.5)
        m = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm, rc)
        run!(m; days = 1.0, verbose = false)

        rundir = joinpath(tmpdir, "iotest")
        @test isdir(rundir)
        @test isfile(joinpath(rundir, "log.txt"))
        @test filesize(joinpath(rundir, "log.txt")) > 0

        # NetCDF output written at the saveday cadence.  The filename carries
        # the day stamp rounded to whole days, so both sub-day writes here
        # collapse onto output_000001.nc — hence >= 1, not >= 2.
        ncs = sort(filter(endswith(".nc"), readdir(rundir)))
        @test length(ncs) >= 1
        NCD = Laddie.NCDatasets
        melt_out = NCD.Dataset(joinpath(rundir, ncs[end])) do ds
            @test haskey(ds, "melt") && haskey(ds, "D") && haskey(ds, "T")
            coalesce.(Array(ds["melt"][:, :]), NaN)
        end
        @test any(isfinite, melt_out)
        @test maximum(filter(isfinite, melt_out)) > 0   # m/yr, warm cavity melts

        # Restart written, then round-trips: a model restarted from it must
        # carry the same prognostic state.
        @test isfile(joinpath(rundir, "restart_latest.jld2"))
        rc2 = RunConfig(; name = "iotest2", resultdir = tmpdir, saveday = 0.5,
                        fromrestart = true,
                        restartfile = joinpath(rundir, "restart_latest.jld2"))
        m2 = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm, rc = rc2)
        @test m2.t_start ≈ 1.0 atol = 0.01
        @test m2.D.present ≈ m.D.present
        @test m2.T.present ≈ m.T.present
        @test m2.S.present ≈ m.S.present

        # Typed Model: unknown properties now error instead of landing in a Dict
        @test_throws ErrorException m.no_such_field
        @test_throws ErrorException (m.no_such_field = 1)
    end

    @testset "Verification vs Python LADDIE: 1-day warm ISOMIP+" begin
        # End-state comparison against the reference Python LADDIE (v1.1) restart
        # after 1 day on the identical 240×40 warm configuration (see
        # docs/src/examples/python_comparison.jl).  Tolerances are ~2× the
        # residuals measured at verification time, which are consistent with
        # NumPy-vs-Julia floating-point evaluation-order differences; a physics
        # regression exceeds them by orders of magnitude.
        # Note: output_000001.nc holds day-AVERAGED fields and is not comparable
        # to the end state — only the restart file is used here.
        py_restart = joinpath(@__DIR__, "..", "docs", "assets", "restart_000001.nc")
        if isfile(py_restart)
            m = build_isomip(; isomipcond = :warm)   # full size, Float64
            run!(m; days = 1.0, verbose = false)

            NCD = Laddie.NCDatasets
            py, py_tmask = NCD.Dataset(py_restart) do ds
                # Python stores (x, y, n) with n=2 the present leapfrog level;
                # transpose to Julia's (ny, nx) interior layout.
                get_v(v) = coalesce.(Array(ds[v][:, :, 2]), 0.0)'
                Dict(v => get_v(v) for v in ("D", "T", "S", "U", "V")),
                coalesce.(Array(ds["tmask"][:, :]), 0.0)'
            end

            inner(a) = a[2:end-1, 2:end-1]
            tm = inner(m.tmask) .> 0
            @test all((inner(m.tmask) .> 0) .== (py_tmask .> 0))

            #            field  mean|Δ|   max|Δ|       (measured: mean / max)
            tols = Dict("D" => (0.05,    6.0),     # 0.019  / 3.0   m
                        "T" => (0.004,   0.03),    # 0.0015 / 0.013 °C
                        "S" => (0.0015,  0.015),   # 0.0006 / 0.006 psu
                        "U" => (0.0003,  0.04),    # 1.1e-4 / 0.016 m/s
                        "V" => (0.0003,  0.04))    # 0.9e-4 / 0.016 m/s
            for (v, jl) in (("D", m.D.present), ("T", m.T.present),
                            ("S", m.S.present), ("U", m.U.present),
                            ("V", m.V.present))
                resid = abs.(inner(jl) .- py[v])[tm]
                mean_tol, max_tol = tols[v]
                @test sum(resid) / length(resid) < mean_tol
                @test maximum(resid) < max_tol
            end

            # Melt stats verified against the Python log diagnostics at t ≈ 1 day
            # (Python: mean 24.42, max ≈ 141 m/yr).
            mx, mn, _ = meltstats(m)
            @test isapprox(mn, 24.42; rtol = 0.01)
            @test isapprox(mx, 140.7; rtol = 0.01)
        else
            @info "Python restart not found at $py_restart — skipping verification testset."
            @test_skip false
        end
    end

    # -------------------------------------------------------------------------
    # GPU tests: mirror each test above using CUDABackend, then cross-compare
    # with the CPU result to verify bit-level agreement.
    # Skipped entirely when no CUDA device is present.
    # -------------------------------------------------------------------------
    if gpu_backend !== nothing

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
