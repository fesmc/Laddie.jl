using Laddie
using Test
using KernelAbstractions
using Aqua
import CUDA

FT = Float64

# Detect GPU: use CUDABackend if a functional CUDA device is present.
const gpu_backend = CUDA.functional() ? CUDA.CUDABackend() : nothing

# Fake forcing types for the property-forwarding collision guard tests
# (type definitions must live at top level, not inside a @testset).
struct CollidingForcing <: Laddie.AbstractForcing
    Tz::Vector{Float64}
    Sz::Vector{Float64}
    z::Vector{Float64}
    dz::Float64
    z0::Float64
    melt::Float64   # collides with Cache.melt
end

struct ReservedNameForcing <: Laddie.AbstractForcing
    Tz::Vector{Float64}
    Sz::Vector{Float64}
    z::Vector{Float64}
    dz::Float64
    z0::Float64
    nx::Int         # collides with the reserved Model property `nx`
end

@testset verbose=true "Laddie.jl" begin

    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Laddie)
    end

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

    @testset "build_model: input validation errors" begin
        nx_i, ny_i = 6, 4
        mask = zeros(Int, ny_i + 2, nx_i + 2)
        mask[1, :]   .= 1;   mask[end, :] .= 1
        mask[:, 1]   .= 1;   mask[:, end] .= 1
        mask[2:end-1, 2:3]   .= 2
        mask[2:end-1, 4:end-1] .= 3
        zb = fill(-400.0, ny_i + 2, nx_i + 2)
        forcing = ISOMIPForcing(FT, :warm)
        params  = Params(; FT)

        # zb size mismatch
        @test_throws ArgumentError build_model(mask, zb[:, 1:end-1], 2000.0, 2000.0, forcing, params; FT)
        # non-positive cell spacing
        @test_throws ArgumentError build_model(mask, zb, -2000.0, 2000.0, forcing, params; FT)
        # mask value outside 0:3
        bad = copy(mask); bad[3, 4] = 7
        @test_throws ArgumentError build_model(bad, zb, 2000.0, 2000.0, forcing, params; FT)
        # no floating-shelf cells at all
        none = copy(mask); none[none .== 3] .= 2
        @test_throws ArgumentError build_model(none, zb, 2000.0, 2000.0, forcing, params; FT)
        # shelf cell on the border ring
        edge = copy(mask); edge[1, 4] = 3
        @test_throws ArgumentError build_model(edge, zb, 2000.0, 2000.0, forcing, params; FT)
        # FT mismatch with params and with forcing
        @test_throws ArgumentError build_model(mask, zb, 2000.0, 2000.0, forcing, Params(; FT = Float32); FT)
        @test_throws ArgumentError build_model(mask, zb, 2000.0, 2000.0, ISOMIPForcing(Float32, :warm), params; FT)
        # valid inputs still build
        m = build_model(mask, zb, 2000.0, 2000.0, forcing, params; FT)
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

    @testset "Mask cleaning: fill_ocean_holes!, fill_shelf_holes!, fill_small_shelf_patches!" begin
        # 6×6 interior (8×8 with border ring).
        # Layout (interior rows 1-6, cols 1-6):
        #   cols 1-3  → grounded (2)
        #   cols 4-6  → shelf (3), except cell (3,3) interior = (4,4) in full = isolated ocean
        #   cell (2,5) interior = (3,6) in full = isolated shelf (surrounded by grounded)
        m = zeros(Int, 8, 8)
        m[1, :] .= 1; m[end, :] .= 1; m[:, 1] .= 1; m[:, end] .= 1
        m[2:end-1, 2:4] .= 2   # grounded block
        m[2:end-1, 5:end-1] .= 3  # shelf block

        # Plant an isolated ocean pocket inside the shelf
        m[4, 6] = 0   # one ocean cell surrounded by shelf — not reachable from border
        # Plant an isolated shelf cell surrounded by grounded ice
        m[3, 3] = 3   # shelf cell inside the grounded block

        # fill_ocean_holes!: the isolated ocean pocket at (4,6) is not connected
        # to the outer ocean (which is fully excluded by the border ring of 1s);
        # it should be reclassified as grounded.
        mc = copy(m)
        n = fill_ocean_holes!(mc)
        @test n == 1
        @test mc[4, 6] == 2   # reclassified
        @test mc[3, 3] == 3   # shelf cell untouched

        # fill_shelf_holes!: the isolated shelf at (3,3) has no ocean neighbour
        # → reclassified; the shelf block (cols 5-6) touches ocean → kept.
        mc2 = copy(m)
        n = fill_shelf_holes!(mc2)
        @test n == 1
        @test mc2[3, 3] == 2   # reclassified
        @test mc2[3, 6] == 3   # shelf block untouched

        # fill_small_shelf_patches!: a single-cell shelf component is below min_cells=5
        # → reclassified; the main shelf block (many cells) is kept.
        mc3 = copy(m)
        n = fill_small_shelf_patches!(mc3, 5)
        @test n == 1
        @test mc3[3, 3] == 2   # single-cell component removed
        # Main shelf block has >> 5 cells → untouched (spot-check cells away from the ocean hole)
        @test mc3[2, 5] == 3
        @test mc3[5, 7] == 3

        # min_cells=1 keeps everything including single-cell patches
        mc4 = copy(m)
        mc4[3, 3] = 3
        n = fill_small_shelf_patches!(mc4, 1)
        @test n == 0
        @test mc4[3, 3] == 3

        # Return value: cells reclassified (0 when nothing to do)
        mc5 = copy(m); mc5[3, 3] = 2   # no isolated shelf
        @test fill_small_shelf_patches!(mc5, 10) == 0
    end

    @testset "Mask cleaning: fill_small_grounded_patches!" begin
        # 6×6 interior (8×8 with border ring).
        # Left grounded block (cols 2-4) touches the border → main sheet, never removed.
        # A 2-cell isolated grounded patch (rows 3-4, col 6) sits inside the shelf.
        g = zeros(Int, 8, 8)
        g[1, :] .= 1; g[end, :] .= 1; g[:, 1] .= 1; g[:, end] .= 1
        g[2:end-1, 2:4] .= 2   # main grounded block (touches border → kept)
        g[2:end-1, 5:end-1] .= 3  # shelf block
        g[3, 6] = 2             # isolated grounded cell 1
        g[4, 6] = 2             # isolated grounded cell 2 (component size = 2)

        # min_cells=5: the 2-cell isolated patch is removed → shelf; main block untouched
        gc = copy(g)
        n = fill_small_grounded_patches!(gc, 5)
        @test n == 2
        @test gc[3, 6] == 3   # reclassified to shelf
        @test gc[4, 6] == 3
        @test gc[3, 3] == 2   # main grounded block untouched

        # min_cells=1: nothing removed (component size 2 >= 1)
        gc2 = copy(g)
        @test fill_small_grounded_patches!(gc2, 1) == 0
        @test gc2[3, 6] == 2   # kept

        # min_cells=3: 2-cell patch is below threshold → removed
        gc3 = copy(g)
        @test fill_small_grounded_patches!(gc3, 3) == 2
        @test gc3[3, 6] == 3

        # Border-connected grounded cells are never removed regardless of min_cells
        gc4 = copy(g)
        @test fill_small_grounded_patches!(gc4, 100) == 2   # only the 2-cell patch
        @test gc4[3, 3] == 2   # main block kept
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
            melting = TurbulentGamT(FT(13.8), FT(2432.0), FT(1.95e-6)),
            entrainment  = GasparEntrainment(FT(2.5)),
            convection_scheme = ResetToAmbient(FT(0.005)),
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
            entrainment  = HollandEntrainment(FT(0.01775)),
            melting = FixedGamT(FT(0.00018)),
            convection_scheme = ResetToAmbient(FT(0.005)),
        )
        m = build_isomip(CPU(); FT, nx=20, ny=10, isomipcond=:warm, params)
        @test all(isfinite, m.melt)
        @test all(m.melt[m.tmask .> 0] .>= 0)
        run!(m; days=0.5, verbose=false)
        @test all(isfinite, m.D.present)
        @test all(isfinite, m.melt)
        @test all(m.melt[m.tmask .> 0] .>= 0)
    end

    @testset "Entrainment: Lambert is default, Gaspar (literal Eq. 14) differs" begin
        # LambertEntrainment reproduces the reference LADDIE production term
        # (2μ u★³/(g D δρ)) and must be the default so the Python verification
        # holds; GasparEntrainment is the literal Eq. 14 (μ u★³/(g D² δρ)).
        @test build_isomip(CPU(); nx=20, ny=10, isomipcond=:cold).entrainment isa
              LambertEntrainment
        mk(ep) = build_isomip(CPU(); FT, nx=20, ny=10, isomipcond=:warm,
            gradient = PyGradient(),
            params = Params(; FT, entrainment = ep, melting = FixedGamT(FT(0.00018)),
                            convection_scheme = ResetToAmbient(FT(0.005))))
        ml = mk(LambertEntrainment(FT(2.5)))
        mg = mk(GasparEntrainment(FT(2.5)))
        run!(ml; days=0.5, verbose=false)
        run!(mg; days=0.5, verbose=false)
        @test all(isfinite, ml.entr) && all(isfinite, mg.entr)
        @test all(isfinite, ml.melt) && all(isfinite, mg.melt)
        # Same μ but different production term ⇒ the entrainment fields diverge.
        @test !isapprox(ml.entr, mg.entr)
    end

    @testset "Grounding-line BC: FreeSlipGL bit-identical, NoSlipGL differs" begin
        # Explicit FreeSlipGL is the default and must reproduce it bit-for-bit
        # (dslip = 0 leaves the kernel arithmetic unchanged), so the Python
        # verification remains valid for the default configuration.
        m_def  = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
        m_free = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                              params = Params(; FT, grline_bc = FreeSlipGL()))
        run!(m_def;  days = 1.0, verbose = false)
        run!(m_free; days = 1.0, verbose = false)
        @test m_free.U.present == m_def.U.present
        @test m_free.V.present == m_def.V.present
        @test m_free.melt == m_def.melt

        # GL wall indicators are a pointwise subset of the grounded ones; the
        # ISOMIP+ geometry has a meridional grounding line, so GL faces exist
        # at least for the V-walls (glEv/glWv).
        g = getfield(m_def, :grid)
        @test all(g.glNu .<= g.grdNu) && all(g.glSu .<= g.grdSu)
        @test all(g.glEv .<= g.grdEv) && all(g.glWv .<= g.grdWv)
        @test sum(g.glEv) + sum(g.glWv) > 0

        # No-slip at the grounding line changes the solution and stays physical.
        m_ns = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                            params = Params(; FT, grline_bc = NoSlipGL()))
        run!(m_ns; days = 1.0, verbose = false)
        @test all(isfinite, m_ns.D.present) && all(isfinite, m_ns.melt)
        @test all(m_ns.melt[m_ns.tmask .> 0] .>= 0)
        @test m_ns.V.present != m_def.V.present
    end

    @testset "Time stepper: FixedDt default/equivalence, AdaptiveDt threading" begin
        # FixedDt is the default; an explicit FixedDt() must reproduce it
        # bit-for-bit so the Python verification stays valid for the default.
        m_def = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
        @test getfield(m_def, :params).tstep isa FixedDt
        m_fix = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                             params = Params(; FT, tstep = FixedDt()))
        run!(m_def; days = 1.0, verbose = false)
        run!(m_fix; days = 1.0, verbose = false)
        @test m_fix.D.present == m_def.D.present
        @test m_fix.melt == m_def.melt

        # AdaptiveDt threads through Params → Model; its FT tracks Params' FT
        # (default-constructed at Float64 here, promoted to Float32).
        p = Params(; FT, tstep = AdaptiveDt(; cfl_target = 0.4, ncheck = 10))
        @test p.tstep isa AdaptiveDt{FT}
        @test p.tstep.cfl_target ≈ FT(0.4) && p.tstep.ncheck == 10
        @test Params(; FT = Float32, tstep = AdaptiveDt()).tstep isa AdaptiveDt{Float32}

        # Run metadata records the active stepper for both default and adaptive.
        tmp = mktempdir()
        build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                     config = RunConfig(; name = "tsfix", resultdir = tmp, saveday = 0.5))
        meta = Laddie.TOML.parsefile(joinpath(tmp, "tsfix", "run_metadata.toml"))
        @test meta["params"]["time_stepper"]["type"] == "FixedDt"

        build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                     params = Params(; FT, tstep = AdaptiveDt(; cfl_target = 0.4)),
                     config = RunConfig(; name = "tsadp", resultdir = tmp, saveday = 0.5))
        meta2 = Laddie.TOML.parsefile(joinpath(tmp, "tsadp", "run_metadata.toml"))
        @test meta2["params"]["time_stepper"]["type"] == "AdaptiveDt"
        @test meta2["params"]["time_stepper"]["cfl_target"] ≈ 0.4
    end

    @testset "Params: parameterizations promoted to FT" begin
        # A mixed-precision call — FT = Float32 with objects built at Float64 —
        # yields a fully Float32 parameter set (no silent Float64 leakage that
        # would crash a Float32→Float64 setfield in the physics kernels).
        p = Params(; FT = Float32,
                   entrainment  = GasparEntrainment(2.5),
                   melting = FixedGamT(0.00018),
                   convection_scheme = ResetToAmbient(0.005),
                   tstep   = AdaptiveDt(; cfl_target = 0.4))
        @test p.entrainment  isa GasparEntrainment{Float32}
        @test p.melting isa FixedGamT{Float32}
        @test p.convection_scheme isa ResetToAmbient{Float32}
        @test p.tstep   isa AdaptiveDt{Float32}
        @test p.tstep.ncheck isa Int                  # integer field not converted
        @test p.open_bc isa ZeroGradientInflow && p.grline_bc isa FreeSlipGL   # singletons pass through

        # The payoff: a Float32 build + run from an explicit Params no longer
        # errors on a Float64-typed parameterization.
        m = build_isomip(CPU(); FT = Float32, nx = 20, ny = 10, isomipcond = :warm,
                         params = Params(; FT = Float32, tstep = AdaptiveDt()))
        run!(m; days = 0.2, verbose = false)
        @test all(isfinite, m.melt) && eltype(m.melt) == Float32
    end

    @testset "ClampDensity convection scheme: build and short run" begin
        params = Params(; FT, convection_scheme = ClampDensity(FT(0.005)))
        m = build_isomip(CPU(); FT, nx=20, ny=10, isomipcond=:warm, params)
        @test all(isfinite, m.melt)
        run!(m; days=0.5, verbose=false)
        @test all(isfinite, m.D.present)
        @test all(isfinite, m.melt)
    end

    @testset "RelaxToAmbient convection scheme: build and short run" begin
        params = Params(; FT, convection_scheme = RelaxToAmbient(FT(10000.0)))
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

    @testset "run! verbose path: ProgressMeter bar" begin
        # All other run! tests use verbose = false; exercise the progress-bar
        # code path (diagnostics refresh, showvalues, finish!) with the
        # output swallowed.
        m = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm)
        ret = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                run!(m; days = 0.05, verbose = true)
            end
        end
        @test ret === m
        @test all(isfinite, m.melt)
    end

    @testset "run!: CFL warning and blow-up detection" begin
        # CFL warning fires when dt is too large for the grid; days = 0 → no
        # stepping, so only the pre-loop warning is exercised.
        m = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm,
                         params = Params(; dt = 5000.0))
        @test_logs (:warn, r"CFL") run!(m; days = 0.0, verbose = false)

        # Default ISOMIP+ setup is CFL-safe: no warning.
        m_ok = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm)
        @test_logs run!(m_ok; days = 0.0, verbose = false)

        # Non-finite prognostics abort with an informative error instead of
        # integrating NaNs to the end of the run.
        m2 = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm)
        m2.D.present[5, 5] = NaN
        @test_throws "blew up" run!(m2; days = 0.1, verbose = false)
    end

    @testset "CFL number: matches hand-built states" begin
        m = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
        g, dt, dx, dy = m.g, m.dt, m.dx, m.dy
        cfl(u, v, D, dr) = begin
            m.U.present .= u; m.V.present .= v
            m.D.present .= D; m.drho .= dr
            Laddie._cfl_number(m)
        end

        # Full advective + gravity-wave case: c = √(g·δρ·D).
        c = sqrt(g * 1e-3 * 100.0)
        expected = dt * ((0.5 + c) / dx + (0.3 + c) / dy)
        @test cfl(0.5, 0.3, 100.0, 1e-3) ≈ expected
        # Uses max|U|/max|V| — sign-independent.
        @test cfl(-0.5, -0.3, 100.0, 1e-3) ≈ expected
        # Velocity-only (δρ = 0 → c = 0).
        @test cfl(0.4, 0.2, 100.0, 0.0) ≈ dt * (0.4 / dx + 0.2 / dy)
        # Gravity-wave-only (zero velocity) — what the startup check leans on.
        c2 = sqrt(g * 2e-3 * 50.0)
        @test cfl(0.0, 0.0, 50.0, 2e-3) ≈ dt * (c2 / dx + c2 / dy)
        # CPU Float64 scalar (device reductions return to host).
        @test cfl(0.5, 0.3, 100.0, 1e-3) isa Float64
    end

    @testset "AdaptiveDt controller: bounds, rescue, logging" begin
        # Predictive, asymmetric controller arithmetic.  Defaults: cfl_target =
        # 0.5, q = 1, max_growth = 1.1, grow_hyst = 0.8, dt ∈ [1, 1000].
        ts = AdaptiveDt()
        @test Laddie._controller_dt(ts, 210.0, 1.0;  allow_grow = true)  ≈ 105.0        # above target → shrink to target (q=1)
        @test Laddie._controller_dt(ts, 210.0, 0.45; allow_grow = true)  ≈ 210.0        # hysteresis band → hold
        @test Laddie._controller_dt(ts, 210.0, 0.20; allow_grow = true)  ≈ 210.0 * 1.1  # well below → grow, capped
        @test Laddie._controller_dt(ts, 210.0, 0.20; allow_grow = false) ≈ 210.0        # startup never grows
        @test Laddie._controller_dt(ts, 9000.0, 1.0; allow_grow = true)  ≈ 1000.0       # clamp to dtmax
        @test Laddie._controller_dt(ts, 210.0, 0.0;  allow_grow = true)  ≈ 210.0        # no CFL signal → hold

        # Startup rescue (worst-case basis): leaves a safe dt0 alone, shrinks a
        # too-large one before the first step.
        msafe = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                             params = Params(; FT, tstep = AdaptiveDt()))
        Laddie._init_adaptive_dt!(msafe, msafe.tstep)
        @test msafe.dt == 210.0
        mbig = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                            params = Params(; FT, dt = 5000.0, tstep = AdaptiveDt()))
        Laddie._init_adaptive_dt!(mbig, mbig.tstep)
        @test mbig.dt < 5000.0

        # Warm run completes with dt staying in bounds.
        ma = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                          params = Params(; FT, tstep = AdaptiveDt()))
        run!(ma; days = 1.0, verbose = false)
        @test all(isfinite, ma.D.present) && all(isfinite, ma.melt)
        @test 1.0 <= ma.dt <= 1000.0

        # Stability rescue (headline): a dt0 that blows up under FixedDt is made
        # to survive by the controller.  Blow-up can surface as a thrown error
        # or as non-finite state, so check survival directly.
        survives(p, days) = try
            mm = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm, params = p)
            run!(mm; days, verbose = false)
            all(isfinite, mm.D.present) && all(isfinite, mm.melt)
        catch
            false
        end
        @test !survives(Params(; FT, dt = 5000.0),                        2.0)  # FixedDt blows up
        @test  survives(Params(; FT, dt = 5000.0, tstep = AdaptiveDt()),  2.0)  # AdaptiveDt rescues

        # dt changes are logged to log.txt.
        tmp = mktempdir()
        ml = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                          params = Params(; FT, tstep = AdaptiveDt()),
                          config = RunConfig(; name = "adlog", resultdir = tmp, saveday = 10.0))
        run!(ml; days = 1.0, verbose = false)
        @test occursin(r"dt .* → .* s \(CFL", read(joinpath(tmp, "adlog", "log.txt"), String))
    end

    @testset "AdaptiveDt: accuracy, step count, restart round-trip" begin
        # Accuracy: on 1-day warm ISOMIP+ the adaptive solution tracks the
        # fixed-dt one to within a few percent (measured ~2.2% mean, ~0.1% max).
        # PyGradient is used here so tolerances reflect the calibrated dynamics.
        mf = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                          gradient = PyGradient())
        ma = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                          gradient = PyGradient(),
                          params = Params(; FT, tstep = AdaptiveDt()))
        run!(mf; days = 1.0, verbose = false)
        run!(ma; days = 1.0, verbose = false)
        mxf, mnf, _ = meltstats(mf)
        mxa, mna, _ = meltstats(ma)
        @test abs(mna - mnf) / mnf < 0.04
        @test abs(mxa - mxf) / mxf < 0.02

        # Speedup: in a cold (slow) cavity the controller grows dt, so the same
        # 1 day is reached in measurably fewer steps (measured 266 vs 411).
        cf = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :cold)
        ca = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :cold,
                          params = Params(; FT, tstep = AdaptiveDt()))
        run!(cf; days = 1.0, verbose = false)
        run!(ca; days = 1.0, verbose = false)
        @test ca.t < 0.9 * cf.t
        @test ca.dt > cf.dt          # dt grew above the fixed step

        # Restart round-trip: the current dt is saved and restored, so an
        # adaptive run resumes at exactly the step it left off.
        tmp = mktempdir()
        m1 = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                          params = Params(; FT, tstep = AdaptiveDt()),
                          config = RunConfig(; name = "ar1", resultdir = tmp,
                                         saveday = 0.5, restday = 0.5))
        run!(m1; days = 1.0, verbose = false)
        m2 = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                          params = Params(; FT, tstep = AdaptiveDt()),
                          config = RunConfig(; name = "ar2", resultdir = tmp, saveday = 0.5,
                                         fromrestart = true,
                                         restartfile = joinpath(tmp, "ar1", "restart_latest.jld2")))
        @test m2.dt ≈ m1.dt
        @test m2.D.present ≈ m1.D.present
        run!(m2; days = 0.5, verbose = false)
        @test all(isfinite, m2.D.present) && all(isfinite, m2.melt)
    end

    @testset "Simulation end: FixedSimulationEnd / SteadyStateEnd" begin
        # Steady-state criterion: relative change in mean melt below tol.
        @test  Laddie._steady_reached(SteadyStateEnd(tol = 0.01), 100.0, 100.05)  # 5e-4 < 1e-2
        @test !Laddie._steady_reached(SteadyStateEnd(tol = 0.01), 100.0, 90.0)    # 0.11
        @test !Laddie._steady_reached(SteadyStateEnd(tol = 0.01), 100.0, NaN)     # no predecessor
        @test !Laddie._steady_reached(FixedSimulationEnd(), 100.0, 100.0)         # fixed never early-stops

        # `days` is shorthand for FixedSimulationEnd — bit-identical, same steps.
        a = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
        b = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
        run!(a; days = 1.0, verbose = false)
        run!(b; until = FixedSimulationEnd(t_end = 1.0), verbose = false)
        @test a.D.present == b.D.present && a.melt == b.melt && a.t == b.t

        # Passing both `days` and `until` is ambiguous.
        @test_throws ArgumentError run!(a; days = 1.0, until = FixedSimulationEnd())

        # SteadyStateEnd stops early once the day-over-day mean-melt change drops
        # below tol; a (near-)zero tol never triggers and runs to the t_end cap.
        cap = 20.0
        ms = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
        run!(ms; until = SteadyStateEnd(tol = 0.3, t_end = cap), verbose = false)
        @test ms.t_sim < 0.5 * cap * 86400              # stopped well before the cap
        @test all(isfinite, ms.D.present) && all(isfinite, ms.melt)

        mc = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
        run!(mc; until = SteadyStateEnd(tol = 1e-12, t_end = cap), verbose = false)
        @test mc.t_sim > 0.9 * cap * 86400              # ran essentially to the cap
        @test ms.t < mc.t                               # early stop took fewer steps
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

    @testset "Forcing structs are concretely typed" begin
        forcings = (
            ISOMIPForcing(FT, :warm),
            LinearForcing(FT, 33.8, 34.7, 1.0, -720.0, -0.0573, 0.0832),
            Linear2Forcing(FT, 33.8, 34.7, 1.0, -720.0, -0.0573, 0.0832),
            TanhForcing(FT, 33.8, 1.0, -720.0, 100.0, 0.01, 1028.0,
                        3.733e-5, 7.843e-4, -0.0573, 0.0832),
            ProfileForcing([1.0, 0.0], [34.7, 34.2], [-1000.0, -100.0]; FT),
        )
        for f in forcings
            @test all(isconcretetype, fieldtypes(typeof(f)))
            @test f.Tz isa Vector{FT}
            @test all(isfinite, f.Tz) && all(isfinite, f.Sz)
        end
    end

    @testset "Model property forwarding: collision guard" begin
        m = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm)
        parts = (getfield(m, :io), getfield(m, :config), getfield(m, :grid),
                 getfield(m, :state), getfield(m, :cache), getfield(m, :params))
        v = zeros(2)
        @test_throws "ambiguous" Model(parts..., CollidingForcing(v, v, v, 1.0, -5000.0, 0.0))
        @test_throws "reserved" Model(parts..., ReservedNameForcing(v, v, v, 1.0, -5000.0, 3))
        # The shipped struct combination is collision-free (also checked at
        # every Model construction).
        @test Model(parts..., getfield(m, :forcing)) isa Model
    end

    @testset "Compact show methods" begin
        m = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm)
        plain(x) = sprint(show, MIME("text/plain"), x)

        s = plain(m)
        @test occursin("Model{Float64} on CPU", s)
        @test occursin("20×", replace(s, "10×20" => "20×10")) || occursin("interior", s)
        @test occursin("forcing", s) && occursin("params", s)
        @test length(s) < 800   # not a field dump

        sg = plain(getfield(m, :grid))
        @test occursin("shelf", sg) && occursin("interior", sg)
        @test length(sg) < 400

        sp = plain(getfield(m, :params))
        @test occursin("Params{Float64}", sp)
        @test occursin("dt0 = 210.0", sp) && occursin("entrainment", sp)
        @test length(sp) < 1500

        @test occursin("ISOMIP+ :warm", plain(getfield(m, :forcing)))
        for x in (getfield(m, :state), getfield(m, :cache), getfield(m, :io), m.D)
            @test length(plain(x)) < 400
        end
    end

    @testset "Fused kernels match reference equation terms" begin
        # The fused step kernels in numerics.jl and the equation-term
        # functions in physics.jl implement the same governing equations.
        # Reconstruct one leapfrog step from the term functions and require
        # the kernels to reproduce it, for both the scalar-coefficient
        # (FixedGamT/ResetToAmbient) and matrix-coefficient
        # (TurbulentGamT/RelaxToAmbient) kernel variants.
        configs = (
            Params(; FT),
            Params(; FT,
                   melting = TurbulentGamT(FT(13.8), FT(2432.0), FT(1.95e-6)),
                   convection_scheme = RelaxToAmbient(FT(10000.0)),
                   entrainment  = HollandEntrainment(FT(0.01775))),
        )
        for params in configs
            m = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm, params)
            run!(m; days = 0.2, verbose = false)   # develop a non-trivial flow
            Laddie.advance_leapfrog!(m)
            dt = 2 * m.dt
            Laddie.step_thickness(m, dt)
            Laddie.precompute_integration_terms!(m)

            rhs_U = .- Laddie.u_thickness_tendency(m) .+ Laddie.u_advection(m) .-
                       Laddie.u_pressure_depth(m)     .+ Laddie.u_pressure_slope(m) .-
                       Laddie.u_pressure_density(m)   .+ Laddie.u_coriolis(m) .-
                       Laddie.u_bottom_drag(m)        .+ Laddie.u_diffusion(m) .-
                       Laddie.u_detrainment(m)
            U_ref = m.U.past .+
                Laddie.div0(rhs_U, Laddie.ip_t(m, m.D.present)) .* m.umask .* dt

            rhs_V = .- Laddie.v_thickness_tendency(m) .+ Laddie.v_advection(m) .-
                       Laddie.v_pressure_depth(m)     .+ Laddie.v_pressure_slope(m) .-
                       Laddie.v_pressure_density(m)   .- Laddie.v_coriolis(m) .-
                       Laddie.v_bottom_drag(m)        .+ Laddie.v_diffusion(m) .-
                       Laddie.v_detrainment(m)
            V_ref = m.V.past .+
                Laddie.div0(rhs_V, Laddie.jp_t(m, m.D.present)) .* m.vmask .* dt

            rhs_T = .- Laddie.tracer_thickness_tendency(m, m.T.present) .+
                       Laddie.tracer_advection(m, m.T.present) .+
                       Laddie.tracer_entrainment(m, m.Ta) .+
                       Laddie.T_ice_ocean_exchange(m) .+
                       Laddie.tracer_diffusion(m, m.T.past) .-
                       Laddie.tracer_convection(m, m.T.past, m.Ta)
            T_ref = m.T.past .+ Laddie.div0(rhs_T, m.D.present) .* m.tmask .* dt

            rhs_S = .- Laddie.tracer_thickness_tendency(m, m.S.present) .+
                       Laddie.tracer_advection(m, m.S.present) .+
                       Laddie.tracer_entrainment(m, m.Sa) .+
                       Laddie.tracer_diffusion(m, m.S.past) .-
                       Laddie.tracer_convection(m, m.S.past, m.Sa)
            S_ref = m.S.past .+ Laddie.div0(rhs_S, m.D.present) .* m.tmask .* dt

            Laddie.step_u_momentum(m, dt)
            Laddie.step_v_momentum(m, dt)
            Laddie.step_temperature(m, dt)
            Laddie.step_salinity(m, dt)

            @test m.U.future ≈ U_ref rtol = 1e-10 atol = 1e-12
            @test m.V.future ≈ V_ref rtol = 1e-10 atol = 1e-12
            @test m.T.future ≈ T_ref rtol = 1e-10 atol = 1e-12
            @test m.S.future ≈ S_ref rtol = 1e-10 atol = 1e-12
        end
    end

    @testset "Conservation: D equation exact over one step" begin
        m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:warm)
        Laddie.advance_leapfrog!(m)
        D_past = copy(m.D.past)
        src    = copy((m.convD .+ m.melt .+ m.nentr) .* m.tmask)
        Laddie.leapfrog_step!(m, 2)
        @test m.D.future ≈ D_past .+ src .* (2 * m.dt)
    end

    @testset "Conservation: D ≥ D_min after 1-day run" begin
        m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:warm)
        run!(m; days=1.0, verbose=false)
        active = m.tmask .> 0
        @test all(m.D.present[active] .>= m.D_min - 1e-10)
    end

    @testset "Conservation: D ≥ D_min after 1-day run (cold)" begin
        m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:cold)
        run!(m; days=1.0, verbose=false)
        active = m.tmask .> 0
        @test all(m.D.present[active] .>= m.D_min - 1e-10)
    end

    @testset "I/O: NetCDF output, log, and JLD2 restart round-trip" begin
        tmpdir = mktempdir()
        config = RunConfig(; name = "iotest", resultdir = tmpdir,
                       saveday = 0.5, diagday = 0.5, restday = 0.5)
        m = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm, config)
        run!(m; days = 1.0, verbose = false)

        rundir = joinpath(tmpdir, "iotest")
        @test isdir(rundir)
        @test isfile(joinpath(rundir, "log.txt"))
        @test filesize(joinpath(rundir, "log.txt")) > 0

        # Run provenance metadata: full effective configuration on disk
        meta_path = joinpath(rundir, "run_metadata.toml")
        @test isfile(meta_path)
        meta = Laddie.TOML.parsefile(meta_path)
        @test meta["run"]["float_type"] == "Float64"
        @test meta["run"]["backend"] == "CPU"
        @test meta["run"]["laddie_version"] isa String
        @test meta["grid"]["nx"] == 20 && meta["grid"]["ny"] == 10
        @test meta["params"]["dt0"] == 210.0
        @test meta["params"]["melt"]["type"] == "FixedGamT"
        @test meta["params"]["melt"]["gamTfix"] ≈ 0.00018
        @test meta["params"]["grounding_line"]["type"] == "FreeSlipGL"
        @test meta["forcing"]["type"] == "ISOMIPForcing"
        @test meta["forcing"]["isomipcond"] == "warm"
        @test meta["run_config"]["saveday"] == 0.5

        # All output is written into a single output.nc with a time dimension;
        # saveday=0.5 over 1 day produces at least 2 time slices.
        NCD = Laddie.NCDatasets
        out_path = joinpath(rundir, "output.nc")
        @test isfile(out_path)
        melt_out = NCD.Dataset(out_path) do ds
            @test haskey(ds, "melt") && haskey(ds, "D") && haskey(ds, "T")
            @test length(ds["time"]) >= 2
            coalesce.(Array(ds["melt"][:, :, end]), NaN)
        end
        @test any(isfinite, melt_out)
        @test maximum(filter(isfinite, melt_out)) > 0   # m/yr, warm cavity melts

        # Restart written, then round-trips: a model restarted from it must
        # carry the same prognostic state.
        @test isfile(joinpath(rundir, "restart_latest.jld2"))
        rc2 = RunConfig(; name = "iotest2", resultdir = tmpdir, saveday = 0.5,
                        fromrestart = true,
                        restartfile = joinpath(rundir, "restart_latest.jld2"))
        m2 = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm, config = rc2)
        @test m2.t_start ≈ 1.0 atol = 0.01
        @test m2.D.present ≈ m.D.present
        @test m2.T.present ≈ m.T.present
        @test m2.S.present ≈ m.S.present

        # Continuation timestamps: time values in output.nc must carry the
        # t_start offset, not restart from day 0.
        run!(m2; days = 0.5, verbose = false)
        rundir2 = joinpath(tmpdir, "iotest2")
        out2_path = joinpath(rundir2, "output.nc")
        @test isfile(out2_path)
        NCD.Dataset(out2_path) do ds
            @test ds["time"][end] ≈ 1.5 atol = 0.01   # end of window at t_start + 0.5
            @test ds["time"][1] >= 1.0 - 0.01           # first slice starts after restart
        end
        latest2 = joinpath(rundir2, "restart_latest.jld2")
        @test isfile(latest2)
        Laddie.JLD2.jldopen(latest2, "r") do f
            @test f["t_days"] ≈ 1.5 atol = 0.01   # chained restarts accumulate
        end

        # Continuation metadata records the restart offset
        meta2 = Laddie.TOML.parsefile(joinpath(rundir2, "run_metadata.toml"))
        @test meta2["run"]["t_start_days"] ≈ 1.0 atol = 0.01
        @test meta2["run_config"]["fromrestart"] === true

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
        #
        # PyGradient() matches the Python LADDIE v1.1 centred-difference stencil
        # (no mask awareness), so the field residuals are meaningful here.
        # For production runs use the default JlGradient() which avoids spurious
        # slopes from ocean/grounded neighbours.
        py_restart = joinpath(@__DIR__, "..", "docs", "assets", "restart_000001.nc")
        if isfile(py_restart)
            m = build_isomip(; isomipcond = :warm, gradient = PyGradient())
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

            # Melt stats verified against the Python log diagnostics at t ≈ 1 day.
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
            # CFL monitor reductions run on the device and match the CPU value.
            @test Laddie._cfl_number(m_g) ≈ Laddie._cfl_number(m_c)
        end

        @testset "NoSlipGL (GPU): matches CPU" begin
            m_c = build_isomip(CPU();       nx = 20, ny = 10, isomipcond = :warm,
                               params = Params(; grline_bc = NoSlipGL()))
            m_g = build_isomip(gpu_backend; nx = 20, ny = 10, isomipcond = :warm,
                               params = Params(; grline_bc = NoSlipGL()))
            run!(m_c; days = 0.5, verbose = false)
            run!(m_g; days = 0.5, verbose = false)
            @test Array(m_g.melt)      ≈ m_c.melt
            @test Array(m_g.V.present) ≈ m_c.V.present
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

        @testset "AdaptiveDt (GPU): controller + re-bootstrap run on device" begin
            # The CFL reductions, worst-case startup rescue, and re-bootstrap
            # must all be GPU-safe; assert a clean completion in bounds.
            m_g = build_isomip(gpu_backend; nx = 20, ny = 10, isomipcond = :warm,
                               params = Params(; tstep = AdaptiveDt()))
            run!(m_g; days = 1.0, verbose = false)
            @test all(isfinite, Array(m_g.D.present)) && all(isfinite, Array(m_g.melt))
            @test 1.0 <= m_g.dt <= 1000.0
        end

    end # gpu_backend !== nothing

end
