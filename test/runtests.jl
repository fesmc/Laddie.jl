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
struct CollidingForcing <: Laddie.AbstractOceanForcing
    Tz::Vector{Float64}
    Sz::Vector{Float64}
    z::Vector{Float64}
    dz::Float64
    z0::Float64
    melt::Float64   # collides with Cache.melt
end

struct ReservedNameForcing <: Laddie.AbstractOceanForcing
    Tz::Vector{Float64}
    Sz::Vector{Float64}
    z::Vector{Float64}
    dz::Float64
    z0::Float64
    nx::Int         # collides with the reserved Model property `nx`
end

struct CollidingIceForcing <: Laddie.AbstractIceForcing
    T_ice_base::Matrix{Float64}
    Tz::Vector{Float64}   # collides with the ocean forcing's profile
    CollidingIceForcing(T) = new(T, Float64[])
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

    @testset "Model: arbitrary mask and draft" begin
        # Minimal 4×6 interior domain (6×8 with border ring):
        #   col 1 and 8 → boundary (1); cols 2–3 → grounded (2); cols 4–7 → shelf (3)
        nx_i, ny_i = 6, 4
        mask = zeros(Int, ny_i + 2, nx_i + 2)
        mask[1, :]   .= 1;   mask[end, :] .= 1
        mask[:, 1]   .= 1;   mask[:, end] .= 1
        mask[2:end-1, 2:3]   .= 2
        mask[2:end-1, 4:end-1] .= 3
        z_draft_raw = fill(-400.0, ny_i + 2, nx_i + 2)

        forcing = ISOMIPForcing(FT, :warm)
        params  = Params(; FT)
        m = Model(mask, z_draft_raw, 2000.0, 2000.0, forcing, params; FT,
                  domain_cropping = NoDomainCropping())

        @test size(m.tmask) == (ny_i + 2, nx_i + 2)
        @test all(isfinite, m.melt)
        @test all(m.melt[m.tmask .> 0] .>= 0)
        run!(m; days=0.5, verbose=false)
        @test all(isfinite, m.D.present)
        @test all(isfinite, m.melt)

        # MinRectangleDomainCropping is the default.  Its `margin` (default 4) is the
        # padding kept around the active region; here the shelf spans cols 4-7 of 8,
        # so a 4-cell margin reaches the array edge and nothing is cropped at all.
        mc = Model(mask, z_draft_raw, 2000.0, 2000.0, forcing, params; FT)
        @test size(mc.tmask) == (ny_i + 2, nx_i + 2)
        @test sum(mc.tmask) == sum(m.tmask)     # no active cell is lost

        # margin = 1 is the tightest the solver accepts: the shelf bounding box plus
        # the one-cell ring, dropping the outer grounded column and the far border
        # column (8 columns -> 6).
        m1 = Model(mask, z_draft_raw, 2000.0, 2000.0, forcing, params; FT,
                   domain_cropping = MinRectangleDomainCropping(margin = 1))
        @test size(m1.tmask) == (ny_i + 2, 6)
        @test sum(m1.tmask) == sum(m.tmask)

        # margin = 2 keeps one more ring than that.
        m2 = Model(mask, z_draft_raw, 2000.0, 2000.0, forcing, params; FT,
                   domain_cropping = MinRectangleDomainCropping(margin = 2))
        @test size(m2.tmask) == (ny_i + 2, 7)

        # margin = 0 would put shelf cells on the wrapping border ring.
        @test_throws ArgumentError Model(mask, z_draft_raw, 2000.0, 2000.0, forcing,
            params; FT, domain_cropping = MinRectangleDomainCropping(margin = 0))
        @test MinRectangleDomainCropping().margin == 4
    end

    @testset "Model: input validation errors" begin
        nx_i, ny_i = 6, 4
        mask = zeros(Int, ny_i + 2, nx_i + 2)
        mask[1, :]   .= 1;   mask[end, :] .= 1
        mask[:, 1]   .= 1;   mask[:, end] .= 1
        mask[2:end-1, 2:3]   .= 2
        mask[2:end-1, 4:end-1] .= 3
        z_draft = fill(-400.0, ny_i + 2, nx_i + 2)
        forcing = ISOMIPForcing(FT, :warm)
        params  = Params(; FT)

        # z_draft size mismatch
        @test_throws ArgumentError Model(mask, z_draft[:, 1:end-1], 2000.0, 2000.0, forcing, params; FT)
        # non-positive cell spacing
        @test_throws ArgumentError Model(mask, z_draft, -2000.0, 2000.0, forcing, params; FT)
        # mask value outside 0:3
        bad = copy(mask); bad[3, 4] = 7
        @test_throws ArgumentError Model(bad, z_draft, 2000.0, 2000.0, forcing, params; FT)
        # no floating-shelf cells at all
        none = copy(mask); none[none .== 3] .= 2
        @test_throws ArgumentError Model(none, z_draft, 2000.0, 2000.0, forcing, params; FT)
        # shelf cell on the border ring
        edge = copy(mask); edge[1, 4] = 3
        @test_throws ArgumentError Model(edge, z_draft, 2000.0, 2000.0, forcing, params; FT)
        # FT mismatch with params and with forcing
        @test_throws ArgumentError Model(mask, z_draft, 2000.0, 2000.0, forcing, Params(; FT = Float32); FT)
        @test_throws ArgumentError Model(mask, z_draft, 2000.0, 2000.0, ISOMIPForcing(Float32, :warm), params; FT)
        # valid inputs still build
        m = Model(mask, z_draft, 2000.0, 2000.0, forcing, params; FT)
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

        # Ice-free cells are split by bed elevation: bedrock at or above sea level
        # is land (1), not ocean (0).  Exposed rock classified as ocean would act
        # as an open-boundary sink wherever it sits inside the domain.
        bed2       = [-500.0  -500.0  -200.0   40.0    0.0;
                      -500.0  -500.0  -200.0  900.0   -0.1]
        thickness2 = [ 600.0   400.0     0.0    0.0    0.0;
                       600.0   400.0     0.0    0.0    0.0]
        m2 = build_laddie_mask(bed2, thickness2)
        @test m2[2, 4] == 0   # ice-free, bed -200 m  → ocean
        @test m2[2, 5] == 1   # ice-free, bed  +40 m  → land
        @test m2[3, 5] == 1   # ice-free, bed +900 m  → land
        @test m2[2, 6] == 1   # ice-free, bed    0 m  → land (sea level counts as land)
        @test m2[3, 6] == 0   # ice-free, bed -0.1 m  → ocean

        # Mismatched sizes must throw
        @test_throws ArgumentError build_laddie_mask(bed, thickness[1:1, :])
    end

    @testset "Land vs grounded ice: rock islands are walls, not ice fronts" begin
        # Shelf filling the domain, open ocean on the last interior column, and a
        # 2x2 rock island (mask 1) punched into the middle of the shelf.
        mk = zeros(Int, 12, 22)
        mk[1, :] .= 1;  mk[end, :] .= 1;  mk[:, 1] .= 1;  mk[:, end] .= 1
        mk[2:11, 2:3]  .= 2      # grounded ice
        mk[2:11, 4:20] .= 3      # shelf
        mk[2:11, 21]   .= 0      # open ocean → real ice front at column 20
        mk[5:6, 10:11] .= 1      # rock island
        isl = CartesianIndices((5:6, 10:11))
        m = Model(mk, fill(-400.0, 12, 22), 2000.0, 2000.0, ISOMIPForcing(FT, :warm),
                  Params(; FT); FT, domain_cropping = NoDomainCropping())

        @test all(m.lnd[isl] .== 1)      # island is land
        @test all(m.ocn[isl] .== 0)      # and emphatically not ocean
        @test all(m.grd[isl] .== 1)      # it is a wall
        @test all(m.tmask[isl] .== 0)    # not simulated
        @test sum(m.lnd) == count(mk .== 1)   # border ring + island, nothing else

        # The island generates no ice front: at_isf fires on ocean neighbours only,
        # and the island has none.  The genuine front at column 20 still does.
        isf_here = (m.tmask .> 0) .&
                   (m.ocnxm1 .+ m.ocnxp1 .+ m.ocnym1 .+ m.ocnyp1 .> 0)
        for I in isl, (di, dj) in ((-1,0),(1,0),(0,-1),(0,1))
            i, j = Tuple(I) .+ (di, dj)
            (5 <= i <= 6 && 10 <= j <= 11) && continue
            @test isf_here[i, j] == false
        end
        @test any(isf_here[2:11, 20])    # the real ice front is still detected

        # The island is a free-slip wall, not a grounding line: grd?? indicators
        # fire around it while the gl?? ones (mask == 2 only) stay off — and the
        # land-only lnd?? ones (AbstractLandBC) fire there instead, mirroring how
        # gl?? does for grounded ice.
        g = getfield(m, :grid)
        @test g.grdEv[5, 9] > 0 || g.grdWv[5, 12] > 0 || g.grdNu[7, 10] > 0 || g.grdSu[4, 10] > 0
        @test all(g.glNu[isl] .== 0) && all(g.glSu[isl] .== 0)
        @test all(g.glEv[isl] .== 0) && all(g.glWv[isl] .== 0)
        @test all(g.glNu .<= g.grdNu) && all(g.glEv .<= g.grdEv)
        @test g.lndEv[5, 9] > 0 || g.lndWv[5, 12] > 0 || g.lndNu[7, 10] > 0 || g.lndSu[4, 10] > 0
        @test all(g.lndNu .<= g.grdNu) && all(g.lndEv .<= g.grdEv)

        run!(m; days = 0.5, verbose = false)
        @test all(isfinite, m.D.present) && all(isfinite, m.melt)
    end

    @testset "Geometry ingestion: ice_base_depth values" begin
        bed       = [-500.0  -500.0  -200.0]
        thickness = [ 600.0   400.0     0.0]

        z_draft = ice_base_depth(bed, thickness)
        @test size(z_draft) == (3, 5)    # (1+2, 3+2)
        # Border zeros
        @test all(z_draft[1, :] .== 0.0)
        @test all(z_draft[end, :] .== 0.0)
        @test all(z_draft[:, 1]   .== 0.0)
        @test all(z_draft[:, end] .== 0.0)
        # Grounded: z_draft = bed
        @test z_draft[2, 2] ≈ -500.0
        # Floating: z_draft = -h * rho_ice/rho_sw
        @test z_draft[2, 3] ≈ -400.0 * 917.0 / 1028.0
        # Ocean: z_draft = 0
        @test z_draft[2, 4] ≈ 0.0
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
        # to the outer ocean; it should be reclassified as land (1) — there is no
        # ice in a water pocket, so calling it grounded ice would be wrong.
        mc = copy(m)
        n = fill_ocean_holes!(mc)
        @test n == 1
        @test mc[4, 6] == 1   # reclassified as land
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

    @testset "Geometry ingestion: end-to-end Model from synthetic BedMachine" begin
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
        m = Model(mask_bm, zb_bm, 2000.0, 2000.0, forcing, params; FT)
        @test all(isfinite, m.melt)
        @test all(m.melt[m.tmask .> 0] .>= 0)
        run!(m; days = 0.5, verbose = false)
        @test all(isfinite, m.D.present)
        @test all(isfinite, m.melt)
    end

    @testset "OceanForcing1D: resampling, sorting, flat extrapolation" begin
        z_c = [-1000.0, -500.0, -100.0]
        T_c = [   1.0,     0.0,   -1.0]
        S_c = [  34.7,    34.2,   33.8]
        f = OceanForcing1D(T_c, S_c, z_c; FT)
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
        f_rev = OceanForcing1D(reverse(T_c), reverse(S_c), reverse(z_c); FT)
        @test f_rev.Tz == f.Tz
        @test f_rev.Sz == f.Sz

        # Duplicate depths are tolerated (first occurrence kept)
        f_dup = OceanForcing1D([1.0, 2.0, -1.0], [34.7, 34.6, 33.8],
                               [-1000.0, -1000.0, -100.0]; FT)
        @test all(isfinite, f_dup.Tz)

        # Length mismatch must throw
        @test_throws ArgumentError OceanForcing1D(T_c[1:2], S_c, z_c; FT)
    end

    @testset "OceanForcing1D: reproduces ISOMIPForcing from coarse samples" begin
        # The warm ISOMIP profile is linear in z, so 3 samples recover it exactly.
        isomip = ISOMIPForcing(FT, :warm)
        T_lin(z) = -1.9 + z * (1.0 - (-1.9)) / (-720.0)
        S_lin(z) = 33.8 + z * (34.7 - 33.8) / (-720.0)
        z_c  = [-5000.0, -720.0, -1.0]
        prof = OceanForcing1D(T_lin.(z_c), S_lin.(z_c), z_c; FT)
        @test prof.Tz ≈ isomip.Tz
        @test prof.Sz ≈ isomip.Sz

        # Same domain, both forcings: initial melt fields must agree
        nx_i, ny_i = 6, 4
        mask = zeros(Int, ny_i + 2, nx_i + 2)
        mask[1, :]   .= 1;   mask[end, :] .= 1
        mask[:, 1]   .= 1;   mask[:, end] .= 1
        mask[2:end-1, 2:3]   .= 2
        mask[2:end-1, 4:end-1] .= 3
        z_draft_raw = fill(-400.0, ny_i + 2, nx_i + 2)

        m1 = Model(mask, z_draft_raw, 2000.0, 2000.0, isomip, Params(; FT); FT)
        m2 = Model(mask, z_draft_raw, 2000.0, 2000.0, prof,  Params(; FT); FT)
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

    @testset "TurbulentGamTMelting: build and short run" begin
        params = Params(;
            FT,
            melting = TurbulentGamTMelting(FT(13.8), FT(2432.0), FT(1.95e-6)),
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
            melting = FixedGamTMelting(FT(0.00018)),
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
            params = Params(; FT, entrainment = ep, melting = FixedGamTMelting(FT(0.00018)),
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

    @testset "Land BC: FreeSlipLand bit-identical, NoSlipLand differs, independent of grline_bc" begin
        # Same structure as the grounding-line BC testset above: FreeSlipLand is
        # the default and must reproduce it bit-for-bit (dslip_land = 0 leaves the
        # kernel arithmetic unchanged).  ISOMIP+'s channel is narrow enough (10
        # interior rows) that its side walls (the outer border ring, mask == 1)
        # are physical land walls, not just an inert numerical buffer, so this
        # geometry already exercises the land wall stencils without needing a
        # hand-built island.
        m_def  = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
        m_free = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                              params = Params(; FT, land_bc = FreeSlipLand()))
        run!(m_def;  days = 1.0, verbose = false)
        run!(m_free; days = 1.0, verbose = false)
        @test m_free.U.present == m_def.U.present
        @test m_free.V.present == m_def.V.present
        @test m_free.melt == m_def.melt

        # Land wall indicators are a pointwise subset of the generic wall ones,
        # and fire along the channel side walls (disjoint from the meridional
        # grounding line, which lives on gl?? instead).
        g = getfield(m_def, :grid)
        @test all(g.lndNu .<= g.grdNu) && all(g.lndSu .<= g.grdSu)
        @test all(g.lndEv .<= g.grdEv) && all(g.lndWv .<= g.grdWv)
        @test sum(g.lndNu) + sum(g.lndSu) > 0

        # No-slip at land changes the solution and stays physical.
        m_ns = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                            params = Params(; FT, land_bc = NoSlipLand()))
        run!(m_ns; days = 1.0, verbose = false)
        @test all(isfinite, m_ns.D.present) && all(isfinite, m_ns.melt)
        @test all(m_ns.melt[m_ns.tmask .> 0] .>= 0)
        @test m_ns.V.present != m_def.V.present

        # grline_bc and land_bc are independent switches: engaging one alone must
        # not reproduce engaging the other, and engaging both must differ from
        # either alone (no accidental aliasing between gl?? and lnd??).
        m_gl   = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                              params = Params(; FT, grline_bc = NoSlipGL()))
        m_both = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                              params = Params(; FT, grline_bc = NoSlipGL(), land_bc = NoSlipLand()))
        run!(m_gl;   days = 1.0, verbose = false)
        run!(m_both; days = 1.0, verbose = false)
        @test m_ns.V.present != m_gl.V.present      # land-only ≠ grounding-line-only
        @test m_both.V.present != m_gl.V.present    # both ≠ grounding-line-only
        @test m_both.V.present != m_ns.V.present    # both ≠ land-only
        @test all(isfinite, m_both.D.present) && all(isfinite, m_both.melt)
    end

    @testset "Wall slip: gl/land indicators partition mixed coastline corners" begin
        # The momentum kernels compose the two wall conditions additively
        # (slip + dslip_gl*gl?? + dslip_land*lnd??) but gate on grd?? = OR(gl,lnd).
        # A face whose stencil touches BOTH grounded ice and exposed rock would
        # therefore collect both increments and land at slip 3 instead of 2 under
        # NoSlipGL + NoSlipLand — the very pairing needed to mirror LADDIE v2.
        # Grid must hand such faces to the grounding line alone.
        mk = zeros(Int, 12, 22)
        mk[1, :] .= 1;  mk[end, :] .= 1
        mk[:, 1] .= 1;  mk[:, end] .= 1
        mk[2:11, 2:3]  .= 2
        mk[2:11, 4:20] .= 3
        mk[2:11, 21]   .= 0
        mk[6:7, 6:7]   .= 1          # rock island inside the shelf
        mk[8, 6:7]     .= 2          # grounded ice abutting it -> mixed corners
        z_draft = zeros(FT, size(mk))
        z_draft[mk .== 3] .= FT(-200.0)
        m = Model(mk, z_draft, FT(2000.0), FT(2000.0), ISOMIPForcing(FT, :warm),
                  Params(; FT); domain_cropping = NoDomainCropping())
        g = getfield(m, :grid)

        for (gl, ln, gd) in ((g.glNu, g.lndNu, g.grdNu), (g.glSu, g.lndSu, g.grdSu),
                             (g.glEv, g.lndEv, g.grdEv), (g.glWv, g.lndWv, g.grdWv))
            @test !any((gl .== 1) .& (ln .== 1))   # disjoint...
            @test gl .+ ln ≈ gd                    # ...and exhaustive
        end
        # The geometry really does contain such corners, i.e. this is not vacuous:
        # without the partition, glNu and lndNu would overlap here.
        raw_lndNu = 1 .- Laddie.ym1((1 .- g.lnd) .* (1 .- Laddie.xm1(g.lnd)))
        @test count((g.glNu .== 1) .& (raw_lndNu .== 1)) > 0

        # And the composed slip factor stays at the no-slip value of 2 everywhere.
        p = Params(; FT, grline_bc = NoSlipGL(), land_bc = NoSlipLand())
        dgl = Laddie._gl_slip(p.grline_bc, p.slip) - p.slip
        dln = Laddie._land_slip(p.land_bc, p.slip) - p.slip
        slipN = p.slip .+ dgl .* g.glNu .+ dln .* g.lndNu
        @test maximum(slipN[g.grdNu .== 1]) ≈ 2
    end

    @testset "Lateral viscosity: PrescribedLateralViscosity bit-identical, NonlinearLateralViscosity differs" begin
        # PrescribedLateralViscosity is the default and must reproduce the
        # pre-AbstractLateralViscosity behaviour bit-for-bit: laplace_U/V now
        # dispatch on m.lateral_viscosity, but the Prescribed path is the exact
        # same kernel followed by the same `.*= A_h` that used to live in the
        # momentum-step kernels.
        m_def  = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
        m_presc = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                               params = Params(; FT, lateral_viscosity = PrescribedLateralViscosity()))
        run!(m_def;   days = 1.0, verbose = false)
        run!(m_presc; days = 1.0, verbose = false)
        @test m_presc.U.present == m_def.U.present
        @test m_presc.V.present == m_def.V.present
        @test m_presc.melt == m_def.melt

        # NonlinearLateralViscosity changes the solution and stays physical.
        m_nl = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                            params = Params(; FT, lateral_viscosity = NonlinearLateralViscosity(FT(10.0))))
        run!(m_nl; days = 1.0, verbose = false)
        @test all(isfinite, m_nl.D.present) && all(isfinite, m_nl.melt)
        @test all(m_nl.melt[m_nl.tmask .> 0] .>= 0)
        @test m_nl.V.present != m_def.V.present

        # Decision (a): grounding-line/land wall drag stays linear in the plain
        # A_h even under the nonlinear interior scheme, so switching grline_bc
        # to no-slip must still change the solution under NonlinearLateralViscosity
        # (i.e. the wall-drag term isn't accidentally zeroed or folded into the
        # shear-scaled interior term).
        m_nl_ns = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                               params = Params(; FT, lateral_viscosity = NonlinearLateralViscosity(FT(10.0)),
                                              grline_bc = NoSlipGL()))
        run!(m_nl_ns; days = 1.0, verbose = false)
        @test all(isfinite, m_nl_ns.D.present) && all(isfinite, m_nl_ns.melt)
        @test m_nl_ns.V.present != m_nl.V.present

        # The coefficient uses the full velocity-difference norm √(ΔU² + ΔV²),
        # as the reference's dUabs does — not just the component being diffused.
        # So changing V alone must change the U-viscosity, which a per-component
        # |ΔU| coefficient could not do.  Scale rather than offset V: the norm
        # sees only *differences*, so a uniform shift would change nothing.  For
        # the same reason the flux is coeff * ΔU, so this is only visible where
        # ΔU is itself nonzero — i.e. in the shelf interior, not at a wall.
        lU_before = copy(Laddie.laplace_U(m_nl))
        m_nl.V.past .*= 2
        @test Laddie.laplace_U(m_nl) != lU_before
        # ...and the same coupling must be absent under the plain Laplacian,
        # whose coefficient is a constant and never reads V at all.
        lU0_before = copy(Laddie.laplace_U(m_def))
        m_def.V.past .*= 2
        @test Laddie.laplace_U(m_def) == lU0_before
    end

    @testset "Gaps BC: mask plumbing, SinkGapsBC bit-identical, ConnectedGapsBC differs" begin
        # 10x20 interior domain (12x22 with border ring), no cropping so mask
        # indices map 1:1 onto grid indices:
        #   cols 2-3 grounded (2), cols 4-20 shelf (3), col 21 open ocean (0)
        # with a 3x3 ice-shelf gap punched into the middle of the shelf.
        function gappy_mask()
            mk = zeros(Int, 12, 22)
            mk[1, :] .= 1;  mk[end, :] .= 1
            mk[:, 1] .= 1;  mk[:, end] .= 1
            mk[2:11, 2:3]  .= 2
            mk[2:11, 4:20] .= 3
            mk[2:11, 21]   .= 0
            mk[5:7, 10:12] .= 4          # the gap
            return mk
        end
        gap_ix = CartesianIndices((5:7, 10:12))
        z_draft_raw = fill(-400.0, 12, 22)
        forcing = ISOMIPForcing(FT, :warm)
        build(mk, p) = Model(mk, z_draft_raw, 2000.0, 2000.0, forcing, p;
                             FT, domain_cropping = NoDomainCropping())

        # -- Bucket 1: derived masks -------------------------------------------
        mc = build(gappy_mask(), Params(; FT, gaps_bc = ConnectedGapsBC()))
        @test all(mc.tmask[gap_ix] .== 1)          # gaps are dynamically active
        @test all(mc.imask[gap_ix] .== 0)          # but carry no ice
        @test all(mc.ocn[gap_ix]   .== 0)          # and are not open ocean
        @test all(mc.z_draft[gap_ix] .== 0)        # layer sits at the sea surface
        @test mc.imask != mc.tmask
        @test all(mc.imask .<= mc.tmask)           # imask is a subset of tmask
        # An interior gap is not an ice front: no ocean neighbour anywhere near it.
        @test all(mc.isf[gap_ix] .== 0)
        # Shelf cells are untouched by the gap treatment.
        shelf = (gappy_mask() .== 3)
        @test all(mc.imask[shelf] .== 1) && all(mc.z_draft[shelf] .== -400.0)

        # -- Bucket 1: validation ----------------------------------------------
        params = Params(; FT)
        # A gap on the border ring is rejected under ConnectedGapsBC, where it stays
        # an active cell; under SinkGapsBC it is demoted to ocean first, so it is
        # legal there — the mask is normalised before it is validated.
        edge = gappy_mask(); edge[1, 10] = 4
        @test_throws ArgumentError build(edge, Params(; FT, gaps_bc = ConnectedGapsBC()))
        @test build(edge, params).mask[1, 10] == 0
        # 4 is now legal; 5 is not
        bad = gappy_mask(); bad[6, 6] = 5
        @test_throws ArgumentError build(bad, params)

        # -- Bucket 2: SinkGapsBC is exactly the pre-gap behaviour -------------
        # Demoting gaps to open ocean must reproduce a mask that never had them.
        ocean_mask = gappy_mask(); ocean_mask[ocean_mask .== 4] .= 0
        m_sink  = build(gappy_mask(), Params(; FT, gaps_bc = SinkGapsBC()))
        m_plain = build(ocean_mask,   Params(; FT))     # SinkGapsBC is the default
        @test m_sink.mask == m_plain.mask
        run!(m_sink;  days = 1.0, verbose = false)
        run!(m_plain; days = 1.0, verbose = false)
        @test m_sink.D.present == m_plain.D.present
        @test m_sink.T.present == m_plain.T.present
        @test m_sink.U.present == m_plain.U.present
        @test m_sink.melt == m_plain.melt

        # -- Bucket 2: ConnectedGapsBC keeps the gaps, and refgeo derives them --
        @test sum(mc.tmask) == sum(m_sink.tmask) + length(gap_ix)
        # Same geometry expressed as a reference ice footprint over an all-ocean gap.
        refgeo = zeros(12, 22); refgeo[2:11, 4:20] .= 500.0   # reference ice thickness
        m_ref = build(ocean_mask, Params(; FT, gaps_bc = ConnectedGapsBC(refgeo)))
        @test m_ref.mask == mc.mask
        # A footprint that does not match the mask is caught, not silently broadcast.
        @test_throws ArgumentError build(ocean_mask,
            Params(; FT, gaps_bc = ConnectedGapsBC(zeros(4, 4))))

        # -- Bucket 3: no melt in gaps, and no ice-ocean heat exchange either ---
        @test all(mc.melt[gap_ix] .== 0)
        @test all(mc.Tb[gap_ix] .== mc.T.present[gap_ix])
        # Tb = T makes  melt*Tb - gamT*(T - Tb)  vanish identically in gap cells,
        # which is what keeps heat flowing across the gap instead of draining out.
        exch = Laddie.T_ice_ocean_exchange(mc)
        @test all(exch[gap_ix] .== 0)
        @test any(exch[shelf] .!= 0)               # still active under the ice

        # -- Bucket 4: convection must never reset a gap cell ------------------
        # A gap samples ambient at the sea surface (z_draft = 0), which is the
        # coldest, freshest water in the column, so `drho < 0` there is close to
        # unconditional.  Ungated, ResetToAmbient would overwrite the T/S anomaly
        # the layer carries across the gap every single step, rebuilding the very
        # sink ConnectedGapsBC exists to remove.
        cv_params(scheme) = Params(; FT, gaps_bc = ConnectedGapsBC(),
                                   convection_scheme = scheme)
        # Cool every active cell to drive the whole domain convectively unstable.
        function destabilize(scheme)
            m = build(gappy_mask(), cv_params(scheme))
            m.T.present[m.tmask .> 0] .-= 5
            Laddie.update_density!(m)
            @test all(m.drho[gap_ix] .< 0) && all(m.drho[shelf] .< 0)
            return m, copy(m.T.present), copy(m.S.present)
        end

        m_rst, T0, S0 = destabilize(ResetToAmbient(FT(0.005)))
        Laddie.update_convection!(m_rst)
        @test m_rst.T.present[gap_ix] == T0[gap_ix]      # gaps keep their heat ...
        @test m_rst.S.present[gap_ix] == S0[gap_ix]
        @test all(m_rst.convection[gap_ix] .== 0)        # ... and are never flagged
        @test all(m_rst.T.present[shelf] .!= T0[shelf])  # ice-covered cells do reset
        @test all(m_rst.convection[shelf] .== 1)

        # RelaxToAmbient is the same sink applied gradually, so conv2 is gated too.
        m_rlx, _, _ = destabilize(RelaxToAmbient(FT(10000.0)))
        Laddie.update_convection!(m_rlx)
        Laddie.precompute_integration_terms!(m_rlx)
        @test all(m_rlx.conv2[gap_ix] .== 0)
        @test all(m_rlx.conv2[shelf] .> 0)

        # ClampDensity is deliberately *not* gated: the buoyancy floor is the one
        # convection treatment LADDIE v2 also has, and it applies over its whole
        # active domain, gaps included.
        m_cld, _, _ = destabilize(ClampDensity(FT(0.005)))
        Laddie.update_convection!(m_cld)
        @test all(m_cld.drho[gap_ix] .≈ FT(0.005) / m_cld.rho0_seawater)

        # End-to-end: gap cells really do sit below the reset threshold during a
        # run — ungated the reset would keep firing there — yet the layer still
        # arrives warmer than the surface ambient it is crossing.
        m_cv = build(gappy_mask(), Params(; FT, gaps_bc = ConnectedGapsBC()))
        @test getfield(m_cv, :params).convection_scheme isa ResetToAmbient
        run!(m_cv; days = 1.0, verbose = false)
        @test any(m_cv.drho[gap_ix] .< FT(0.005) / m_cv.rho0_seawater)
        @test all(m_cv.convection[gap_ix] .== 0)
        @test all(m_cv.T.present[gap_ix] .> m_cv.Ta[gap_ix])

        # -- Connected differs from sink, and stays physical -------------------
        run!(mc; days = 1.0, verbose = false)
        @test all(isfinite, mc.D.present) && all(isfinite, mc.melt)
        @test all(mc.melt[mc.imask .> 0] .>= 0)
        @test all(mc.melt[gap_ix] .== 0)           # still zero after integrating
        @test mc.melt != m_sink.melt
    end

    @testset "Gaps BC: meltwater crosses a gap in the ISOMIP+ boundary current" begin
        # The science test for ConnectedGapsBC.  ISOMIP+ channel, coarsened to
        # 60x20 so three runs stay cheap.  Coriolis steers the plume into a
        # boundary current against the high-y wall (rows 19-21 of 22), which is
        # where a gap does the most damage — Jesse et al. (2026), Fig. 3.
        dx, dy = 8000.0, 4000.0
        base = build_isomip(CPU(); FT, nx = 60, ny = 20, dx, dy, isomipcond = :warm)
        mask0, band = copy(base.mask), 19:21
        rows, cols = 19:21, 30:32              # the gap: 12 km across, 24 km along

        # Melt-through thins the ice it eats through, so taper the draft to zero
        # over six cells around the gap rather than leaving a 400 m cliff at its
        # edge.  A cliff is admissible — the reference accepts exactly that — but
        # its pressure slope, some 40x the shelf's own, would swamp the signal
        # being measured here.
        z_draft = copy(base.z_draft)
        for j in axes(z_draft, 1), i in axes(z_draft, 2)
            r = max(max(first(rows) - j, j - last(rows), 0),
                    max(first(cols) - i, i - last(cols), 0))
            z_draft[j, i] *= clamp(r / 6, 0, 1)
        end
        gappy = copy(mask0); gappy[rows, cols] .= 4

        function channel(mask, bc)
            m = Model(mask, z_draft, dx, dy, ISOMIPForcing(FT, :warm),
                      Params(; FT, gaps_bc = bc);
                      FT, domain_cropping = NoDomainCropping())
            run!(m; days = 20.0, verbose = false)
            return m
        end
        m_sink = channel(gappy, SinkGapsBC())
        m_conn = channel(gappy, ConnectedGapsBC())
        m_none = channel(mask0, SinkGapsBC())    # same draft, no gap: the control

        for m in (m_sink, m_conn, m_none)
            @test all(isfinite, m.melt) && all(isfinite, m.D.present)
            @test all(m.melt .>= 0)
        end
        @test all(m_conn.melt[rows, cols] .== 0)   # a gap has no ice to melt

        mn(a) = sum(a) / length(a)
        meltsum(m, c) = sum(m.melt[band, c]) * m.seconds_per_year
        up, down = 5:22, 36:58

        # Upstream of the gap the two treatments are indistinguishable, and both
        # match the no-gap control: what happens at the gap does not reach back.
        @test meltsum(m_conn, up) ≈ meltsum(m_sink, up) rtol = 1e-3
        @test meltsum(m_conn, up) ≈ meltsum(m_none, up) rtol = 1e-3

        # At the gap the two diverge completely: the sink terminates the boundary
        # current (Fig. 3r), the connected layer carries it through (Fig. 3v).
        @test maximum(abs.(m_sink.U.present[band, cols])) < 0.05
        @test minimum(maximum(abs.(m_conn.U.present[band, c])) for c in cols) > 0.2

        # Downstream the sink has drained the cavity and melt collapses, while
        # the connected layer arrives faster, thicker and warmer and melts almost
        # as much as if the gap had never opened — which is the point, given the
        # gap itself melts nothing.
        @test meltsum(m_conn, down) > 1.4 * meltsum(m_sink, down)
        @test meltsum(m_conn, down) ≈ meltsum(m_none, down) rtol = 0.05
        @test mn(abs.(m_conn.U.present[band, down])) >
              1.2 * mn(abs.(m_sink.U.present[band, down]))
        @test mn(m_conn.T.present[band, down]) >
              mn(m_sink.T.present[band, down]) + 0.01
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

    @testset "Params defaults: build_isomip matches Params()" begin
        # build_isomip fills in ISOMIP+-canonical parameters when `params` is not
        # given.  Those must agree field-for-field with `Params()`, otherwise
        # merely *passing* a params object to build_isomip silently changes the
        # physics — which is exactly what happened with max_layer_thickness
        # (build_isomip: Topographic, Params(): Absolute(100)), quietly turning
        # the AdaptiveDt accuracy test into an uncapped-vs-capped comparison.
        implicit = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm).params
        explicit = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                                params = Params(; FT)).params
        @test typeof(implicit) === typeof(explicit)
        for fn in fieldnames(typeof(implicit))
            @test getfield(implicit, fn) == getfield(explicit, fn)
        end
    end

    @testset "Params: parameterizations promoted to FT" begin
        # A mixed-precision call — FT = Float32 with objects built at Float64 —
        # yields a fully Float32 parameter set (no silent Float64 leakage that
        # would crash a Float32→Float64 setfield in the physics kernels).
        p = Params(; FT = Float32,
                   entrainment  = GasparEntrainment(2.5),
                   melting = FixedGamTMelting(0.00018),
                   convection_scheme = ResetToAmbient(0.005),
                   tstep   = AdaptiveDt(; cfl_target = 0.4))
        @test p.entrainment  isa GasparEntrainment{Float32}
        @test p.melting isa FixedGamTMelting{Float32}
        @test p.convection_scheme isa ResetToAmbient{Float32}
        @test p.tstep   isa AdaptiveDt{Float32}
        @test p.tstep.ncheck isa Int                  # integer field not converted
        @test p.open_bc isa ZeroGradientInflow && p.grline_bc isa FreeSlipGL   # singletons pass through
        @test p.land_bc isa FreeSlipLand
        @test p.lateral_viscosity isa PrescribedLateralViscosity

        p_nl = Params(; FT = Float32, lateral_viscosity = NonlinearLateralViscosity(10.0))
        @test p_nl.lateral_viscosity isa NonlinearLateralViscosity{Float32}
        @test p_nl.lateral_viscosity.C_visc isa Float32
        @test p_nl.lateral_viscosity.C_visc isa Float32

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
        # 0.3, q = 1, max_growth = 1.1, grow_hyst = 0.8, dt ∈ [1, 1000].
        # The hysteresis band is [grow_hyst*target, target] = [0.24, 0.3].
        ts = AdaptiveDt()
        @test ts.cfl_target == 0.3
        @test Laddie._controller_dt(ts, 210.0, 1.0;  allow_grow = true)  ≈ 63.0         # above target → shrink to target (q=1)
        @test Laddie._controller_dt(ts, 210.0, 0.45; allow_grow = true)  ≈ 140.0        # above target → shrink
        @test Laddie._controller_dt(ts, 210.0, 0.27; allow_grow = true)  ≈ 210.0        # hysteresis band → hold
        @test Laddie._controller_dt(ts, 210.0, 0.20; allow_grow = true)  ≈ 210.0 * 1.1  # well below → grow, capped
        @test Laddie._controller_dt(ts, 210.0, 0.20; allow_grow = false) ≈ 210.0        # startup never grows
        @test Laddie._controller_dt(ts, 9000.0, 1.0; allow_grow = true)  ≈ 1000.0       # clamp to dtmax
        @test Laddie._controller_dt(ts, 210.0, 0.0;  allow_grow = true)  ≈ 210.0        # no CFL signal → hold

        # Startup rescue (worst-case basis) is shrink-only.  With cfl_target = 0.3
        # the ISOMIP+ default dt0 = 210 s sits just above the worst-case budget
        # (worst-case CFL at t = 0 is ~0.316), so it is trimmed slightly rather
        # than left untouched — but it stays the same order of magnitude, unlike
        # a genuinely too-large dt0.
        msafe = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                             params = Params(; FT, tstep = AdaptiveDt()))
        Laddie._init_adaptive_dt!(msafe, msafe.tstep)
        @test msafe.dt <= 210.0
        @test msafe.dt ≈ 210.0 rtol = 0.1
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
        # to survive by the controller.
        #
        # Blow-up can no longer be detected with `isfinite`: the state clamps in
        # `leapfrog_step!` (v_cut on U/V, T ∈ [-5, 5], S ∈ [32, 36], D floored at
        # D_min and capped by max_layer_thickness) bound every prognostic, so an
        # unstable run stays perfectly finite while producing nonsense — at
        # dt = 5000 s the mean melt rate is ~25x the converged value.  Detect it
        # physically instead, against the small-dt reference solution.
        mref = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                            params = Params(; FT, dt = 210.0))
        run!(mref; days = 2.0, verbose = false)
        mean_ref = meltstats(mref)[2]
        survives(p, days) = try
            mm = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm, params = p)
            run!(mm; days, verbose = false)
            all(isfinite, mm.D.present) && all(isfinite, mm.melt) &&
                meltstats(mm)[2] < 3 * mean_ref
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
        # fixed-dt one to within a couple of percent (measured ~1.3% mean, ~1.2%
        # max).  PyGradient is used here so tolerances reflect the calibrated
        # dynamics.
        #
        # NOTE: both models must be built the same way.  This comparison read as a
        # 19.5% peak-melt error for a while because `mf` took build_isomip's
        # implicit defaults while `ma` passed an explicit `params`, and the two
        # disagreed on max_layer_thickness — so it was measuring an uncapped D
        # against one capped at 100 m, not fixed-dt against adaptive-dt.  The
        # defaults are unified now and the guard test below keeps them that way.
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
        # 2.0% measured.  The fixed-dt run is untouched by the velocity-limiter
        # change (nothing ever reaches v_cut there); the adaptive run clips 5
        # transients following dt re-bootstraps, and a speed cap at v_cut is
        # stricter than the old per-component clamp, which allowed |u| up to
        # sqrt(2)*v_cut.  That shifts the peak by 0.85% and the dt trajectory
        # slightly (224.75 -> 225.8 s).
        @test abs(mxa - mxf) / mxf < 0.03

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
            OceanForcing1D([1.0, 0.0], [34.7, 34.2], [-1000.0, -100.0]; FT),
        )
        for f in forcings
            @test all(isconcretetype, fieldtypes(typeof(f)))
            @test f.Tz isa Vector{FT}
            @test all(isfinite, f.Tz) && all(isfinite, f.Sz)
        end
        # ...and so is the assembled cavity forcing a model actually holds, whose
        # ice field has been materialised onto the grid.
        m = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
        cf = getfield(m, :forcing)
        @test cf isa CavityForcing
        @test all(isconcretetype, fieldtypes(typeof(cf)))
        @test all(isconcretetype, fieldtypes(typeof(cf.ice)))
        @test cf.ice.T_ice_base isa Matrix{FT}
    end

    @testset "Coriolis: 0D/2D defaults agree, latitude varies f, C-grid staggering" begin
        # The whole point of the default latitude being derived from the default f
        # rather than rounded to -70: the two options must agree exactly, so
        # switching to the geographic statement changes nothing until lat is set.
        @test Laddie.DEFAULT_LATITUDE ≈ -69.95 atol = 0.01   # ~70°S, not exactly
        @test 2 * Laddie.EARTH_ROTATION_RATE * sind(Laddie.DEFAULT_LATITUDE) ≈
              Laddie.DEFAULT_CORIOLIS_F
        iso(cp) = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                               params = Params(; FT, coriolis = cp))
        m0 = iso(CoriolisParameter0D())
        m2 = iso(CoriolisParameter2D())
        @test m0.f == m2.f
        run!(m0; days = 1.0, verbose = false)
        run!(m2; days = 1.0, verbose = false)
        @test m2.melt == m0.melt && m2.V.present == m0.V.present

        # ...and the default is what Params used to hold as the scalar `f`.
        @test all(m0.f .== -1.37e-4)
        @test !hasfield(typeof(getfield(m0, :params)), :f)

        # A scalar latitude is still an f-plane, but a different one.
        m75 = iso(CoriolisParameter2D(-75.0))
        @test all(m75.f .≈ 2 * Laddie.EARTH_ROTATION_RATE * sind(-75.0))
        run!(m75; days = 1.0, verbose = false)
        @test m75.V.present != m0.V.present          # stronger rotation, different flow
        @test all(isfinite, m75.melt)

        # Uniform f: the staggered copies equal the T-point field exactly.
        @test m0.fu == m0.f && m0.fv == m0.f

        # 2D latitude: f must be staggered onto the two velocity faces separately,
        # because on a C-grid U and V do not share a point.  A latitude varying in
        # y makes fv differ from f while fu (an x-average) does not.
        mask0 = copy(m0.mask)
        ny_t, nx_t = size(mask0)
        lat_y = [FT(-80 + 10 * (i - 1) / (ny_t - 1)) for i in 1:ny_t, _ in 1:nx_t]
        m_y = Model(mask0, copy(m0.z_draft), 2000.0, 2000.0, ISOMIPForcing(FT, :warm),
                    Params(; FT, coriolis = CoriolisParameter2D(lat_y));
                    FT, domain_cropping = NoDomainCropping())
        @test m_y.f[1, 1] ≈ 2 * Laddie.EARTH_ROTATION_RATE * sind(-80.0)
        @test m_y.fu == m_y.f                        # constant along x
        @test m_y.fv != m_y.f                        # averaged across y
        @test m_y.fv[1, 1] ≈ (m_y.f[1, 1] + m_y.f[2, 1]) / 2
        run!(m_y; days = 1.0, verbose = false)
        @test all(isfinite, m_y.melt) && all(m_y.melt[m_y.imask .> 0] .>= 0)
        @test m_y.V.present != m0.V.present

        # The equivalent x-varying field swaps which face average is trivial.
        lat_x = [FT(-80 + 10 * (j - 1) / (nx_t - 1)) for _ in 1:ny_t, j in 1:nx_t]
        m_x = Model(mask0, copy(m0.z_draft), 2000.0, 2000.0, ISOMIPForcing(FT, :warm),
                    Params(; FT, coriolis = CoriolisParameter2D(lat_x));
                    FT, domain_cropping = NoDomainCropping())
        @test m_x.fv == m_x.f && m_x.fu != m_x.f

        # A 2D latitude is cropped with the mask, not silently mismatched.
        m_crop = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                              params = Params(; FT, coriolis = CoriolisParameter2D(-70.0)),
                              domain_cropping = MinRectangleDomainCropping(margin = 2))
        @test size(m_crop.f) == size(m_crop.tmask)

        # Validation.
        @test_throws ArgumentError iso(CoriolisParameter2D(-120.0))
        @test_throws ArgumentError iso(CoriolisParameter2D(zeros(FT, 3, 3)))

        # Float32 promotion follows Params, like every other parameterization.
        @test Params(; FT = Float32, coriolis = CoriolisParameter0D()).coriolis isa
              CoriolisParameter0D{Float32}
    end

    @testset "Ice forcing: scalar and 2D T_ice_base, default is inert" begin
        # T_ice_base moved out of Params and into the forcing.  The default must
        # reproduce the old scalar Params.T_i = -25.0 exactly, or every existing
        # result shifts.
        m_def = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
        @test m_def.T_ice_base isa Matrix{FT}
        @test size(m_def.T_ice_base) == size(m_def.tmask)
        @test all(m_def.T_ice_base .== -25)
        @test !hasfield(typeof(getfield(m_def, :params)), :T_i)

        mask0 = copy(m_def.mask); zd0 = copy(m_def.z_draft)
        shelf_cols = [j for j in axes(mask0, 2) if any(==(3), @view mask0[:, j])]
        build(ice; kw...) = Model(mask0, zd0, 2000.0, 2000.0,
                                  CavityForcing(ISOMIPForcing(FT, :warm), ice),
                                  Params(; FT, entrainment = LambertEntrainment(FT(2.5)),
                                         melting = FixedGamTMelting(FT(0.00018)),
                                         open_bc = ZeroGradientInflow());
                                  FT, domain_cropping = NoDomainCropping(), kw...)

        # An explicit CavityForcing with the same uniform value is bit-identical to
        # the implicit default, so the move is provably inert.
        m_exp = build(PrescribedIceForcing(FT(-25.0)))
        run!(m_def; days = 1.0, verbose = false)
        run!(m_exp; days = 1.0, verbose = false)
        @test m_exp.melt == m_def.melt
        @test m_exp.T.present == m_def.T.present

        # Warmer ice melts more: L_eff = L - c_i*T_i shrinks from 3.84e5 J/kg at
        # -25 degC to 3.34e5 at 0 degC.  The response is damped well below that 15%
        # because the extra melt cools and freshens the layer that drives it.
        m_warm = build(PrescribedIceForcing(FT(0.0)))
        run!(m_warm; days = 1.0, verbose = false)
        @test sum(m_warm.melt) / sum(m_def.melt) ≈ 1.072 rtol = 0.02

        # A 2D field is the point of the move: temperate ice over the upstream half
        # of the shelf, cold ice over the rest, must land strictly between the two
        # uniform runs and match each of them on its own half.
        half = shelf_cols[1:(length(shelf_cols) ÷ 2)]
        Ti = fill(FT(-25.0), size(mask0)); Ti[:, half] .= 0
        m_2d = build(PrescribedIceForcing(Ti))
        run!(m_2d; days = 1.0, verbose = false)
        @test sum(m_def.melt) < sum(m_2d.melt) < sum(m_warm.melt)
        @test sum(m_2d.melt[:, half]) > sum(m_def.melt[:, half])
        @test m_2d.melt[m_2d.imask .> 0] != m_warm.melt[m_warm.imask .> 0]

        # The turbulent-gamT variant is the second melt kernel; it takes the same
        # per-cell L_eff path and must stay physical on the same 2D field.
        m_turb = Model(mask0, zd0, 2000.0, 2000.0,
                       CavityForcing(ISOMIPForcing(FT, :warm), PrescribedIceForcing(Ti)),
                       Params(; FT, melting = TurbulentGamTMelting());
                       FT, domain_cropping = NoDomainCropping())
        run!(m_turb; days = 1.0, verbose = false)
        @test all(isfinite, m_turb.melt) && all(m_turb.melt .>= 0)

        # Validation: wrong shape, ice above the melting point, NaN.
        @test_throws ArgumentError build(PrescribedIceForcing(zeros(FT, 3, 3)))
        @test_throws ArgumentError build(PrescribedIceForcing(FT(5.0)))
        bad = fill(FT(-25.0), size(mask0)); bad[5, 5] = NaN
        @test_throws ArgumentError build(PrescribedIceForcing(bad))

        # A 2D field is cropped with the mask rather than silently mismatched.
        marked = fill(FT(-25.0), size(mask0)); marked[6, shelf_cols[3]] = FT(-2.0)
        m_crop = Model(mask0, zd0, 2000.0, 2000.0,
                       CavityForcing(ISOMIPForcing(FT, :warm), PrescribedIceForcing(marked)),
                       Params(; FT); FT,
                       domain_cropping = MinRectangleDomainCropping(margin = 2))
        @test size(m_crop.T_ice_base) == size(m_crop.tmask)
        @test count(==(FT(-2.0)), m_crop.T_ice_base) == 1

        # A bare ocean forcing still works and picks up the default ice.
        m_bare = Model(mask0, zd0, 2000.0, 2000.0, ISOMIPForcing(FT, :warm),
                       Params(; FT); FT, domain_cropping = NoDomainCropping())
        @test getfield(m_bare, :forcing) isa CavityForcing
        @test all(m_bare.T_ice_base .== Laddie.DEFAULT_T_ICE_BASE)
    end

    @testset "Model property forwarding: collision guard" begin
        m = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm)
        parts = (getfield(m, :io), getfield(m, :config), getfield(m, :grid),
                 getfield(m, :state), getfield(m, :cache), getfield(m, :params))
        v = zeros(2)
        # The guard inspects the members of the CavityForcing, not the wrapper, so
        # a user-defined ocean forcing is what it has to catch.
        @test_throws "ambiguous" Model(
            parts..., CavityForcing(CollidingForcing(v, v, v, 1.0, -5000.0, 0.0)))
        @test_throws "reserved" Model(
            parts..., CavityForcing(ReservedNameForcing(v, v, v, 1.0, -5000.0, 3)))
        # An ice forcing colliding with the ocean side is caught too.
        @test_throws "ambiguous" Model(
            parts...,
            CavityForcing(getfield(m, :forcing).ocean, CollidingIceForcing(zeros(2, 2))),
        )
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

        sf = plain(getfield(m, :forcing))
        @test occursin("OceanForcing1D", sf) && occursin("5000-point profile", sf)
        @test occursin("PrescribedIceForcing(T_ice_base = -25.0 °C)", sf)
        for x in (getfield(m, :state), getfield(m, :cache), getfield(m, :io), m.D)
            @test length(plain(x)) < 400
        end
    end

    @testset "Fused kernels match reference equation terms" begin
        # The fused step kernels in numerics.jl and the equation-term
        # functions in physics.jl implement the same governing equations.
        # Reconstruct one leapfrog step from the term functions and require
        # the kernels to reproduce it, for both the scalar-coefficient
        # (FixedGamTMelting/ResetToAmbient) and matrix-coefficient
        # (TurbulentGamTMelting/RelaxToAmbient) kernel variants.
        configs = (
            Params(; FT),
            Params(; FT,
                   melting = TurbulentGamTMelting(FT(13.8), FT(2432.0), FT(1.95e-6)),
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

    @testset "Front pressure: FullDepthGradient is default/v1, TruncatedDepthGradient drops the term" begin
        # The depth-gradient part of the PGF at a one-sided face (ice front, or a
        # SinkGapsBC gap-sink edge) differs between references: Python LADDIE v1.x
        # keeps the one-sided difference toward a masked-to-zero neighbour, LADDIE
        # v2 drops the term (mask_cf_b truncation).  See f90-diffs.md §4.
        p_full = Params(; FT, front_pressure = FullDepthGradient())
        p_trunc = Params(; FT, front_pressure = TruncatedDepthGradient())
        @test Params(; FT).front_pressure isa FullDepthGradient   # v1 is the default

        m = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
        run!(m; days = 0.5, verbose = false)
        g = getfield(m, :grid)
        # Under the default the interior term is live and the gate is exactly 1.0,
        # so nothing is altered anywhere.
        up = Laddie.u_pressure_depth(m)
        @test any(!iszero, up[g.tmask_ip.==2])
        @test all(Laddie._pgf_gate(m, g.tmask_ip) .== 1)

        # Selecting the truncation is bit-identical in the interior and exactly
        # zero on one-sided faces.
        m_t = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm, params = p_trunc)
        run!(m_t; days = 0.5, verbose = false)
        g_t = getfield(m_t, :grid)
        up_t = Laddie.u_pressure_depth(m_t)
        vp_t = Laddie.v_pressure_depth(m_t)
        @test all(iszero, up_t[g_t.tmask_ip.!=2])
        @test all(iszero, vp_t[g_t.tmask_jp.!=2])

        # A SinkGapsBC gap edge is exactly such a one-sided face once the gap
        # is demoted to ocean, so it must be gated the same way as the true
        # ice front — build a hand-made domain with an interior gap to check.
        mk = zeros(Int, 12, 22)
        mk[1, :] .= 1;  mk[end, :] .= 1
        mk[:, 1] .= 1;  mk[:, end] .= 1
        mk[2:11, 2:3]  .= 2
        mk[2:11, 4:20] .= 3
        mk[2:11, 21]   .= 0
        mk[5:7, 10:12] .= 4
        z_draft = zeros(FT, size(mk))
        z_draft[mk .== 3] .= FT(-200.0)
        forcing = ISOMIPForcing(FT, :warm)
        m_gap = Model(mk, z_draft, FT(2000.0), FT(2000.0), forcing, p_trunc;
                      domain_cropping = NoDomainCropping())
        run!(m_gap; days = 0.5, verbose = false)
        g_gap = getfield(m_gap, :grid)
        up_gap = Laddie.u_pressure_depth(m_gap)
        vp_gap = Laddie.v_pressure_depth(m_gap)
        @test all(iszero, up_gap[g_gap.tmask_ip.!=2])
        @test all(iszero, vp_gap[g_gap.tmask_jp.!=2])

        # Guard against the assertions above going vacuous: the gate only has
        # teeth on faces where momentum is actually solved (umask/vmask == 1),
        # and the ISOMIP channel has no such face in y at all — the gap domain
        # must supply both, or this testset stops testing the fix.
        @test count((g_gap.tmask_ip .== 1) .& (g_gap.umask .== 1)) > 0
        @test count((g_gap.tmask_jp .== 1) .& (g_gap.vmask .== 1)) > 0

        # The choice is not cosmetic: on a domain that has an ice front, the two
        # settings must actually integrate to different states.
        m_full2 = Model(mk, z_draft, FT(2000.0), FT(2000.0), forcing, p_full;
                        domain_cropping = NoDomainCropping())
        run!(m_full2; days = 0.5, verbose = false)
        @test m_full2.U.present != m_gap.U.present
        @test all(isfinite, m_gap.melt) && all(isfinite, m_full2.melt)
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
        @test meta["params"]["melt"]["type"] == "FixedGamTMelting"
        @test meta["params"]["melt"]["gamTfix"] ≈ 0.00018
        @test meta["params"]["grounding_line"]["type"] == "FreeSlipGL"
        # The forcing entry is split ocean/ice, and records the profile ranges:
        # the arrays themselves are skipped by _scalar_fields, so without these a
        # warm run would be indistinguishable from a cold one in the metadata.
        @test meta["forcing"]["ocean"]["type"] == "OceanForcing1D"
        @test meta["forcing"]["ocean"]["T_range"][2] ≈ 18.23888888888889
        @test meta["forcing"]["ocean"]["nz"] == 5000
        @test meta["forcing"]["ice"]["type"] == "PrescribedIceForcing"
        @test meta["forcing"]["ice"]["T_ice_base_range"] == [-25.0, -25.0]
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
