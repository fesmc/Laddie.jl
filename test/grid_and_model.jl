# Grid and Model construction: arbitrary masks, input validation, axis order.

@testset "Model: arbitrary mask and draft" begin
    # Minimal 6×4 interior domain (8×6 with border ring), x along dim 1:
    #   row 1 and 8 → boundary (1); rows 2–3 → grounded (2); rows 4–7 → shelf (3)
    nx_i, ny_i = 6, 4
    mask = zeros(Int, nx_i + 2, ny_i + 2)
    mask[1, :]   .= 1;   mask[end, :] .= 1
    mask[:, 1]   .= 1;   mask[:, end] .= 1
    mask[2:3, 2:end-1]   .= 2
    mask[4:end-1, 2:end-1] .= 3
    z_draft_raw = fill(-400.0, nx_i + 2, ny_i + 2)

    forcing = ISOMIPForcing(:warm; FT)
    params  = Params(; FT)
    grid = Grid(mask, z_draft_raw, 2000.0, 2000.0; FT, domain_cropping = NoDomainCropping())
    m = Simulation(Model(grid; forcing, params))

    @test size(m.model.tmask) == (nx_i + 2, ny_i + 2)
    @test all(isfinite, m.model.melt)
    @test all(m.model.melt[m.model.tmask .> 0] .>= 0)
    run!(m; days=0.5, verbose=false)
    @test all(isfinite, m.model.D.present)
    @test all(isfinite, m.model.melt)

    # MinRectangleDomainCropping is the default.  Its `margin` (default 4) is the
    # padding kept around the active region; here the shelf spans rows 4-7 of 8,
    # so a 4-cell margin reaches the array edge and nothing is cropped at all.
    mc = Model(Grid(mask, z_draft_raw, 2000.0, 2000.0; FT); forcing, params)
    @test size(mc.tmask) == (nx_i + 2, ny_i + 2)
    @test sum(mc.tmask) == sum(m.model.tmask)     # no active cell is lost

    # margin = 1 is the tightest the solver accepts: the shelf bounding box plus
    # the one-cell ring, dropping the outer grounded row and the far border
    # row (8 rows -> 6).
    m1 = Model(Grid(mask, z_draft_raw, 2000.0, 2000.0; FT,
                    domain_cropping = MinRectangleDomainCropping(margin = 1)); forcing, params)
    @test size(m1.tmask) == (6, ny_i + 2)
    @test sum(m1.tmask) == sum(m.model.tmask)

    # margin = 2 keeps one more ring than that.
    m2 = Model(Grid(mask, z_draft_raw, 2000.0, 2000.0; FT,
                    domain_cropping = MinRectangleDomainCropping(margin = 2)); forcing, params)
    @test size(m2.tmask) == (7, ny_i + 2)

    # margin = 0 would put shelf cells on the wrapping border ring.
    @test_throws ArgumentError Grid(mask, z_draft_raw, 2000.0, 2000.0; FT,
                                    domain_cropping = MinRectangleDomainCropping(margin = 0))
    @test MinRectangleDomainCropping().margin == 4
end

@testset "Model: input validation errors" begin
    nx_i, ny_i = 6, 4
    mask = zeros(Int, nx_i + 2, ny_i + 2)
    mask[1, :]   .= 1;   mask[end, :] .= 1
    mask[:, 1]   .= 1;   mask[:, end] .= 1
    mask[2:3, 2:end-1]   .= 2
    mask[4:end-1, 2:end-1] .= 3
    z_draft = fill(-400.0, nx_i + 2, ny_i + 2)
    forcing = ISOMIPForcing(:warm; FT)
    params  = Params(; FT)

    # z_draft size mismatch
    @test_throws ArgumentError Model(Grid(mask, z_draft[1:end-1, :], 2000.0, 2000.0; FT);
                                     forcing, params)
    # non-positive cell spacing
    @test_throws ArgumentError Model(Grid(mask, z_draft, -2000.0, 2000.0; FT);
                                     forcing, params)
    # coordinates: spacing inferred, checked, and cropped with the mask
    xs = 1.0e5 .+ 500.0 .* (0:nx_i+1)
    ys = -2.0e5 .- 750.0 .* (0:ny_i+1)                  # descending is fine
    gc = Grid(mask, z_draft; x = xs, y = ys, domain_cropping = NoDomainCropping(), FT)
    @test (gc.dx, gc.dy) == (500.0, 750.0)
    @test gc.x == xs[2:end-1] && gc.y == ys[2:end-1]
    @test Grid(mask, z_draft, 500.0, 750.0; x = xs, y = ys).dx == 500
    @test_throws ArgumentError Grid(mask, z_draft, 400.0, 750.0; x = xs, y = ys)
    @test_throws ArgumentError Grid(mask, z_draft; x = xs[1:end-1], y = ys)
    @test_throws ArgumentError Grid(mask, z_draft; x = [xs[1:end-1]; 1e9], y = ys)
    @test_throws ArgumentError Grid(mask, z_draft; x = xs)   # neither y nor dy
    # the default coordinates carry the crop offset
    big = zeros(Int, nx_i + 8, ny_i + 2); big[:, [1, end]] .= 1; big[[1, end], :] .= 1
    big[5:6, 2:end-1] .= 2; big[7:end-3, 2:end-1] .= 3
    gcrop = Grid(big, zeros(size(big)), 2000.0, 2000.0;
                 domain_cropping = MinRectangleDomainCropping(margin = 1))
    r, _ = gcrop.crop
    @test gcrop.x[1] == 2000.0 * first(r)
    # mask value outside 0:3
    bad = copy(mask); bad[4, 3] = 7
    @test_throws ArgumentError Model(Grid(bad, z_draft, 2000.0, 2000.0; FT);
                                     forcing, params)
    # no floating-shelf cells at all
    none = copy(mask); none[none .== 3] .= 2
    @test_throws ArgumentError Model(Grid(none, z_draft, 2000.0, 2000.0; FT);
                                     forcing, params)
    # shelf cell on the border ring
    edge = copy(mask); edge[1, 4] = 3
    @test_throws ArgumentError Model(Grid(edge, z_draft, 2000.0, 2000.0; FT);
                                     forcing, params)
    # FT mismatch with params and with forcing
    @test_throws ArgumentError Model(Grid(mask, z_draft, 2000.0, 2000.0; FT);
                                     forcing, params = Params(; FT = Float32))
    @test_throws ArgumentError Model(Grid(mask, z_draft, 2000.0, 2000.0; FT);
                                     forcing = ISOMIPForcing(:warm; FT = Float32), params)
    # forcing is required; params and boundary default
    @test_throws UndefKeywordError Model(Grid(mask, z_draft, 2000.0, 2000.0; FT); params)
    @test Model(Grid(mask, z_draft, 2000.0, 2000.0); forcing).boundary == BoundaryConditions()
    # valid inputs still build
    m = Simulation(Model(Grid(mask, z_draft, 2000.0, 2000.0; FT); forcing, params))
    @test all(isfinite, m.model.melt)
end

@testset "Axis order: fields are [x, y], dx and dy are not interchangeable" begin
    # Arrays are stored [x, y]: first index along x, second along y.  The ISOMIP+
    # channel deepens along x and is uniform across y, which pins every axis in
    # the chain — a transposed stencil, gradient or spacing shows up here.
    nx_i, ny_i = 30, 12
    m = build_isomip(CPU(); FT, nx = nx_i, ny = ny_i, dx = 2000.0, dy = 1000.0,
                     isomipcond = :warm)   # JlGradient: reads shelf neighbours only
    g = m.model
    @test size(g.tmask) == (nx_i + 2, ny_i + 2)
    @test (g.nx, g.ny) == (nx_i, ny_i)
    @test (length(g.x), length(g.y)) == (nx_i, ny_i)
    @test (g.dx, g.dy) == (FT(2000.0), FT(1000.0))

    # The draft varies along x only, so the mask-aware slope must land entirely in
    # dzdx.  Swapped axes would put it in dzdy instead.  (PyGradient would not do
    # for this check: it differences across the land wall in y and picks up the
    # draft discontinuity there — the artefact JlGradient exists to avoid.)
    shelf = g.tmask .> 0
    @test all(iszero, g.dzdy[shelf])
    @test all(!iszero, g.dzdx[shelf])
    # ... and the draft is constant across y at fixed x.
    @test all(all(g.z_draft[i, :][shelf[i, :]] .== g.z_draft[i, :][shelf[i, :]][1])
              for i in axes(g.z_draft, 1) if any(shelf[i, :]))

    # dx and dy are used where they belong: swapping them is a different problem,
    # not a relabelling of the same one.
    run!(m; days = 0.5, verbose = false)
    m_sw = build_isomip(CPU(); FT, nx = nx_i, ny = ny_i, dx = 1000.0, dy = 2000.0,
                        isomipcond = :warm)
    run!(m_sw; days = 0.5, verbose = false)
    @test size(m_sw.model.tmask) == size(g.tmask)
    @test !isapprox(meltstats(m)[2], meltstats(m_sw)[2]; rtol = 1e-6)
    @test all(isfinite, m_sw.model.melt)
end
