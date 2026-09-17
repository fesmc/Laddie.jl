# Boundary conditions: wall slip, ice-shelf gaps, ice-front pressure gradient.

@testset "Wall slip: factors, no-slip default, v1 partial slip" begin
    # Python LADDIE convention: 0 = free slip, 2 = no slip.
    @test Laddie._gl_slip(FreeSlipGL(), FT) === zero(FT)
    @test Laddie._gl_slip(NoSlipGL(), FT) === FT(2)
    @test Laddie._gl_slip(PartialSlipGL(0.7), FT) === FT(0.7)
    @test Laddie._land_slip(FreeSlipLand(), FT) === zero(FT)
    @test Laddie._land_slip(NoSlipLand(), FT) === FT(2)
    @test Laddie._land_slip(PartialSlipLand(1), FT) === FT(1)
    @test PartialSlipGL(1).factor isa AbstractFloat
    @test_throws ArgumentError PartialSlipGL(2.5)
    @test_throws ArgumentError PartialSlipLand(-0.1)
    @test !hasfield(Params, :slip)                 # no global factor any more

    b = BoundaryConditions()
    @test b.grounding_line isa NoSlipGL && b.land isa NoSlipLand

    run1(boundary) = (s = build_isomip(CPU(); FT, nx = 20, ny = 10,
                                       isomipcond = :warm, boundary);
                      run!(s; days = 1.0, verbose = false); s.model)
    bc(gl, ln) = BoundaryConditions(; grounding_line = gl, land = ln)
    m_def  = run1(BoundaryConditions())
    m_ns   = run1(bc(NoSlipGL(), NoSlipLand()))
    m_free = run1(bc(FreeSlipGL(), FreeSlipLand()))
    m_p0   = run1(bc(PartialSlipGL(0.0), PartialSlipLand(0.0)))
    m_p2   = run1(bc(PartialSlipGL(2.0), PartialSlipLand(2.0)))
    m_v1   = run1(bc(PartialSlipGL(1.0), PartialSlipLand(1.0)))

    # The named conditions are exactly their partial-slip factors.
    @test m_ns.U.present == m_def.U.present && m_ns.melt == m_def.melt
    @test m_p2.U.present == m_ns.U.present && m_p2.V.present == m_ns.V.present
    @test m_p0.U.present == m_free.U.present && m_p0.V.present == m_free.V.present
    # ...and the factor matters, while every choice stays physical.
    @test m_free.V.present != m_ns.V.present
    @test m_v1.V.present != m_ns.V.present && m_v1.V.present != m_free.V.present
    for m in (m_def, m_free, m_v1)
        @test all(isfinite, m.D.present) && all(isfinite, m.melt)
        @test all(m.melt[m.tmask .> 0] .>= 0)
    end

    # GL wall indicators are a pointwise subset of the grounded ones; the
    # ISOMIP+ geometry has a meridional grounding line, so GL faces exist
    # at least for the V-walls (glEv/glWv).  Its channel is narrow enough
    # that the side walls (the border ring, mask == 1) are land walls too.
    g = getfield(m_def, :geometry)
    @test all(g.glNu .<= wall_faces(g).grdNu) && all(g.glSu .<= wall_faces(g).grdSu)
    @test all(g.glEv .<= wall_faces(g).grdEv) && all(g.glWv .<= wall_faces(g).grdWv)
    @test sum(g.glEv) + sum(g.glWv) > 0
    @test all(g.lndNu .<= wall_faces(g).grdNu) && all(g.lndSu .<= wall_faces(g).grdSu)
    @test sum(g.lndNu) + sum(g.lndSu) > 0

    # grounding_line and land are independent switches: engaging one alone must
    # not reproduce engaging the other, and engaging both must differ from
    # either alone (no accidental aliasing between gl?? and lnd??).
    m_gl = run1(bc(NoSlipGL(), FreeSlipLand()))
    m_ln = run1(bc(FreeSlipGL(), NoSlipLand()))
    @test m_gl.V.present != m_ln.V.present
    @test m_gl.V.present != m_ns.V.present && m_ln.V.present != m_ns.V.present
    @test m_gl.V.present != m_free.V.present && m_ln.V.present != m_free.V.present
end

@testset "Wall slip: gl/land indicators partition mixed coastline corners" begin
    # The momentum kernels compose the two wall conditions additively,
    # slip_gl*gl?? + slip_land*lnd??.  A face whose stencil touches BOTH
    # grounded ice and exposed rock would therefore collect both factors and
    # land at slip 4 instead of 2 under NoSlipGL + NoSlipLand.  Grid must hand
    # such faces to the grounding line alone.
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
    grid = Grid(mk, z_draft, FT(2000.0), FT(2000.0); domain_cropping = NoDomainCropping())
    m = Model(grid; forcing = ISOMIPForcing(:warm; FT))
    g = getfield(m, :geometry)

    for (gl, ln, gd) in ((g.glNu, g.lndNu, wall_faces(g).grdNu), (g.glSu, g.lndSu, wall_faces(g).grdSu),
                         (g.glEv, g.lndEv, wall_faces(g).grdEv), (g.glWv, g.lndWv, wall_faces(g).grdWv))
        @test !any((gl .== 1) .& (ln .== 1))   # disjoint...
        @test gl .+ ln ≈ gd                    # ...and exhaustive
    end
    # The geometry really does contain such corners, i.e. this is not vacuous:
    # without the partition, glNu and lndNu would overlap here.
    raw_lndNu = 1 .- Laddie.ym1((1 .- g.lnd) .* (1 .- Laddie.xm1(g.lnd)))
    @test count((g.glNu .== 1) .& (raw_lndNu .== 1)) > 0

    # And the composed slip factor is exactly the no-slip value on every wall
    # face under the default conditions, and zero elsewhere.
    slip_gl, slip_land = Laddie._wall_slips(m)
    slipN = slip_gl .* g.glNu .+ slip_land .* g.lndNu
    @test all(slipN[wall_faces(g).grdNu .== 1] .== 2)
    @test all(slipN[wall_faces(g).grdNu .== 0] .== 0)
end

@testset "Gaps BC: mask plumbing, SinkGapsBC bit-identical, ConnectedGapsBC differs" begin
    # 20x10 interior domain (22x12 with border ring), x along dim 1, no cropping
    # so mask indices map 1:1 onto grid indices:
    #   x 2-3 grounded (2), x 4-20 shelf (3), x 21 open ocean (0)
    # with a 3x3 ice-shelf gap punched into the middle of the shelf.
    function gappy_mask()
        mk = zeros(Int, 22, 12)
        mk[1, :] .= 1;  mk[end, :] .= 1
        mk[:, 1] .= 1;  mk[:, end] .= 1
        mk[2:3,  2:11] .= 2
        mk[4:20, 2:11] .= 3
        mk[21,   2:11] .= 0
        mk[10:12, 5:7] .= 4          # the gap
        return mk
    end
    gap_ix = CartesianIndices((10:12, 5:7))
    z_draft_raw = fill(-400.0, 22, 12)
    forcing = ISOMIPForcing(:warm; FT)
    build(mk, gaps = SinkGapsBC(); params = Params(; FT), kw...) =
        Simulation(Model(Grid(mk, z_draft_raw, 2000.0, 2000.0; FT,
                              domain_cropping = NoDomainCropping(), kw...);
                         forcing, params, boundary = BoundaryConditions(; gaps)))

    # -- Bucket 1: derived masks -------------------------------------------
    mc = build(gappy_mask(), ConnectedGapsBC())
    @test all(mc.model.tmask[gap_ix] .== 1)          # gaps are dynamically active
    @test all(mc.model.imask[gap_ix] .== 0)          # but carry no ice
    @test all(mc.model.ocn[gap_ix]   .== 0)          # and are not open ocean
    @test all(mc.model.z_draft[gap_ix] .== 0)        # layer sits at the sea surface
    @test mc.model.imask != mc.model.tmask
    @test all(mc.model.imask .<= mc.model.tmask)           # imask is a subset of tmask
    # An interior gap is not an ice front: no ocean neighbour anywhere near it.
    @test all(mc.model.isf[gap_ix] .== 0)
    # Shelf cells are untouched by the gap treatment.
    shelf = (gappy_mask() .== 3)
    @test all(mc.model.imask[shelf] .== 1) && all(mc.model.z_draft[shelf] .== -400.0)

    # The grid knows nothing of the treatment: one grid drives both, and only
    # the model's geometry differs.
    gappy_grid = Grid(gappy_mask(), z_draft_raw, 2000.0, 2000.0; FT,
                      domain_cropping = NoDomainCropping())
    @test count(==(4), gappy_grid.mask) == length(gap_ix)
    sink_model = Model(gappy_grid; forcing)
    conn_model = Model(gappy_grid; forcing,
                       boundary = BoundaryConditions(; gaps = ConnectedGapsBC()))
    @test sink_model.grid === conn_model.grid
    @test count(==(4), sink_model.resolved_mask) == 0
    @test conn_model.resolved_mask == gappy_mask()
    @test conn_model.tmask == mc.model.tmask
    @test !hasfield(Grid, :tmask) && !hasfield(Grid, :dzdx) && !hasfield(Grid, :f)
    # Building a grid never modifies the caller's mask, even when preprocessing.
    ocean_in = copy(gappy_mask()); ocean_in[ocean_in .== 4] .= 0
    before = copy(ocean_in)
    footprint = zeros(Bool, 22, 12); footprint[4:20, 2:11] .= true
    g_marked = Grid(ocean_in, z_draft_raw, 2000.0, 2000.0; FT,
                    preprocess = [MarkGapsPreprocess(footprint)],
                    domain_cropping = NoDomainCropping())
    @test ocean_in == before
    @test count(==(4), g_marked.mask) == length(gap_ix)
    @test g_marked.crop == (1:22, 1:12) && g_marked.input_size == (22, 12)
    @test g_marked.x == 2000.0 .* (1:20) && g_marked.y == 2000.0 .* (1:10)

    # -- Bucket 1: validation ----------------------------------------------
    params = Params(; FT)
    # A gap on the border ring is rejected under ConnectedGapsBC, where it stays
    # an active cell; under SinkGapsBC it is demoted to ocean first, so it is
    # legal there — the mask is normalised before it is validated.
    edge = gappy_mask(); edge[1, 10] = 4
    @test_throws ArgumentError build(edge, ConnectedGapsBC())
    @test build(edge).model.resolved_mask[1, 10] == 0
    # 4 is now legal; 5 is not
    bad = gappy_mask(); bad[6, 6] = 5
    @test_throws ArgumentError build(bad)

    # -- Bucket 2: SinkGapsBC is exactly the pre-gap behaviour -------------
    # Demoting gaps to open ocean must reproduce a mask that never had them.
    ocean_mask = gappy_mask(); ocean_mask[ocean_mask .== 4] .= 0
    m_sink  = build(gappy_mask(), SinkGapsBC())
    m_plain = build(ocean_mask)                     # SinkGapsBC is the default
    @test m_sink.model.resolved_mask == m_plain.model.resolved_mask
    run!(m_sink;  days = 1.0, verbose = false)
    run!(m_plain; days = 1.0, verbose = false)
    @test m_sink.model.D.present == m_plain.model.D.present
    @test m_sink.model.T.present == m_plain.model.T.present
    @test m_sink.model.U.present == m_plain.model.U.present
    @test m_sink.model.melt == m_plain.model.melt

    # -- Bucket 2: ConnectedGapsBC keeps the gaps, MarkGapsPreprocess derives them
    @test sum(mc.model.tmask) == sum(m_sink.model.tmask) + length(gap_ix)
    # Same geometry expressed as a reference ice footprint over an all-ocean gap.
    # Marking gaps is geometry (a preprocess step); treating them is the BC.
    refgeo = zeros(22, 12); refgeo[4:20, 2:11] .= 500.0   # reference ice thickness
    connected = ConnectedGapsBC()
    m_ref = build(ocean_mask, connected; preprocess = [MarkGapsPreprocess(refgeo)])
    @test m_ref.model.mask == mc.model.mask
    # A Bool footprint is taken as-is.
    m_bool = build(ocean_mask, connected; preprocess = [MarkGapsPreprocess(refgeo .> 0)])
    @test m_bool.model.mask == mc.model.mask
    # Under SinkGapsBC the marked gaps are demoted again: no gap, no difference.
    m_refsink = build(ocean_mask; preprocess = [MarkGapsPreprocess(refgeo)])
    @test m_refsink.model.resolved_mask == m_plain.model.resolved_mask
    # A footprint that does not match the mask is caught, not silently broadcast.
    @test_throws ArgumentError build(ocean_mask, connected;
                                     preprocess = [MarkGapsPreprocess(zeros(4, 4))])
    @test !hasfield(ConnectedGapsBC, :refgeo)

    # Marking runs before cropping, so a gap at the edge of the reference
    # footprint — outside the bounding box of today's shelf — is part of the
    # active region the crop keeps, rather than cropped away before it exists.
    edge_mask = copy(ocean_mask)
    edge_mask[10:12, 5:7] .= 3                                   # no interior hole
    edge_mask[16:20, 2:11] .= 0                                  # shelf ends at x = 15
    edge_ref = zeros(Bool, 22, 12); edge_ref[4:18, 2:11] .= true
    edge_grid = Grid(edge_mask, z_draft_raw, 2000.0, 2000.0; FT,
                     preprocess = [MarkGapsPreprocess(edge_ref)],
                     domain_cropping = MinRectangleDomainCropping(margin = 1))
    m_edge = Model(edge_grid; forcing, boundary = BoundaryConditions(; gaps = connected))
    @test count(==(4), m_edge.mask) == 10 * 3                  # x 16:18 are gaps
    @test sum(m_edge.tmask) == 10 * (15 - 4 + 1) + 10 * 3      # shelf + gaps all kept
    @test size(m_edge.mask, 1) == (18 - 4 + 1) + 2             # footprint + 1-cell ring

    # -- Bucket 3: no melt in gaps, and no ice-ocean heat exchange either ---
    @test all(mc.model.melt[gap_ix] .== 0)
    @test all(mc.model.Tb[gap_ix] .== mc.model.T.present[gap_ix])
    # Tb = T makes  melt*Tb - gamT*(T - Tb)  vanish identically in gap cells,
    # which is what keeps heat flowing across the gap instead of draining out.
    exch = Laddie.T_ice_ocean_exchange(mc.model)
    @test all(exch[gap_ix] .== 0)
    @test any(exch[shelf] .!= 0)               # still active under the ice

    # -- Bucket 4: convection must never reset a gap cell ------------------
    # A gap samples ambient at the sea surface (z_draft = 0), which is the
    # coldest, freshest water in the column, so `drho < 0` there is close to
    # unconditional.  Ungated, ResetToAmbient would overwrite the T/S anomaly
    # the layer carries across the gap every single step, rebuilding the very
    # sink ConnectedGapsBC exists to remove.
    # Cool every active cell to drive the whole domain convectively unstable.
    function destabilize(scheme)
        m = build(gappy_mask(), ConnectedGapsBC();
                  params = Params(; FT, convection_scheme = scheme))
        m.model.T.present[m.model.tmask .> 0] .-= 5
        Laddie.update_density!(m.model)
        @test all(m.model.drho[gap_ix] .< 0) && all(m.model.drho[shelf] .< 0)
        return m, copy(m.model.T.present), copy(m.model.S.present)
    end

    m_rst, T0, S0 = destabilize(ResetToAmbient(FT(0.005)))
    Laddie.update_convection!(m_rst.model)
    @test m_rst.model.T.present[gap_ix] == T0[gap_ix]      # gaps keep their heat ...
    @test m_rst.model.S.present[gap_ix] == S0[gap_ix]
    @test all(m_rst.model.convection[gap_ix] .== 0)        # ... and are never flagged
    @test all(m_rst.model.T.present[shelf] .!= T0[shelf])  # ice-covered cells do reset
    @test all(m_rst.model.convection[shelf] .== 1)

    # RelaxToAmbient is the same sink applied gradually, so conv2 is gated too.
    m_rlx, _, _ = destabilize(RelaxToAmbient(FT(10000.0)))
    Laddie.update_convection!(m_rlx.model)
    Laddie.precompute_integration_terms!(m_rlx.model, m_rlx.clock.dt)
    @test all(m_rlx.model.conv2[gap_ix] .== 0)
    @test all(m_rlx.model.conv2[shelf] .> 0)

    # ClampDensity is deliberately *not* gated: the buoyancy floor is the one
    # convection treatment LADDIE v2 also has, and it applies over its whole
    # active domain, gaps included.
    m_cld, _, _ = destabilize(ClampDensity(FT(0.005)))
    Laddie.update_convection!(m_cld.model)
    @test all(m_cld.model.drho[gap_ix] .≈ FT(0.005) / m_cld.model.rho0_seawater)

    # End-to-end: gap cells really do sit below the reset threshold during a
    # run — ungated the reset would keep firing there — yet the layer still
    # arrives warmer than the surface ambient it is crossing.
    m_cv = build(gappy_mask(), ConnectedGapsBC())
    @test getfield(m_cv.model, :params).convection_scheme isa ResetToAmbient
    run!(m_cv; days = 1.0, verbose = false)
    @test any(m_cv.model.drho[gap_ix] .< FT(0.005) / m_cv.model.rho0_seawater)
    @test all(m_cv.model.convection[gap_ix] .== 0)
    @test all(m_cv.model.T.present[gap_ix] .> m_cv.model.Ta[gap_ix])

    # -- Connected differs from sink, and stays physical -------------------
    run!(mc; days = 1.0, verbose = false)
    @test all(isfinite, mc.model.D.present) && all(isfinite, mc.model.melt)
    @test all(mc.model.melt[mc.model.imask .> 0] .>= 0)
    @test all(mc.model.melt[gap_ix] .== 0)           # still zero after integrating
    @test mc.model.melt != m_sink.model.melt
end

@testset "Gaps BC: meltwater crosses a gap in the ISOMIP+ boundary current" begin
    # The science test for ConnectedGapsBC.  ISOMIP+ channel, coarsened to
    # 60x20 so three runs stay cheap.  Coriolis steers the plume into a
    # boundary current against the high-y wall (y = 19-21 of 22), which is
    # where a gap does the most damage — Jesse et al. (2026), Fig. 3.
    # Fields are [x, y]: x runs along the channel, y across it.
    dx, dy = 8000.0, 4000.0
    base = build_isomip(CPU(); FT, nx = 60, ny = 20, dx, dy, isomipcond = :warm)
    mask0, ywall = copy(base.model.mask), 19:21
    ygap, xgap = 19:21, 30:32              # the gap: 12 km across, 24 km along

    # Melt-through thins the ice it eats through, so taper the draft to zero
    # over six cells around the gap rather than leaving a 400 m cliff at its
    # edge.  A cliff is admissible — the reference accepts exactly that — but
    # its pressure slope, some 40x the shelf's own, would swamp the signal
    # being measured here.
    z_draft = copy(base.model.z_draft)
    for i in axes(z_draft, 1), j in axes(z_draft, 2)
        r = max(max(first(ygap) - j, j - last(ygap), 0),
                max(first(xgap) - i, i - last(xgap), 0))
        z_draft[i, j] *= clamp(r / 6, 0, 1)
    end
    gappy = copy(mask0); gappy[xgap, ygap] .= 4

    function channel(mask, bc)
        grid = Grid(mask, z_draft, dx, dy; FT, domain_cropping = NoDomainCropping())
        m = Simulation(Model(grid; forcing = ISOMIPForcing(:warm; FT),
                             boundary = BoundaryConditions(; gaps = bc)))
        run!(m; days = 20.0, verbose = false)
        return m
    end
    m_sink = channel(gappy, SinkGapsBC())
    m_conn = channel(gappy, ConnectedGapsBC())
    m_none = channel(mask0, SinkGapsBC())    # same draft, no gap: the control

    for m in (m_sink, m_conn, m_none)
        @test all(isfinite, m.model.melt) && all(isfinite, m.model.D.present)
        @test all(m.model.melt .>= 0)
    end
    @test all(m_conn.model.melt[xgap, ygap] .== 0)   # a gap has no ice to melt

    mn(a) = sum(a) / length(a)
    meltsum(m, xs) = sum(m.model.melt[xs, ywall]) * m.model.seconds_per_year
    up, down = 5:22, 36:58

    # Upstream of the gap the two treatments are indistinguishable, and both
    # match the no-gap control: what happens at the gap does not reach back.
    @test meltsum(m_conn, up) ≈ meltsum(m_sink, up) rtol = 1e-3
    @test meltsum(m_conn, up) ≈ meltsum(m_none, up) rtol = 1e-3

    # At the gap the two diverge completely: the sink terminates the boundary
    # current (Fig. 3r), the connected layer carries it through (Fig. 3v).
    @test maximum(abs.(m_sink.model.U.present[xgap, ywall])) < 0.05
    @test minimum(maximum(abs.(m_conn.model.U.present[x, ywall])) for x in xgap) > 0.2

    # Downstream the sink has drained the cavity and melt collapses, while
    # the connected layer arrives faster, thicker and warmer and melts almost
    # as much as if the gap had never opened — which is the point, given the
    # gap itself melts nothing.
    @test meltsum(m_conn, down) > 1.4 * meltsum(m_sink, down)
    @test meltsum(m_conn, down) ≈ meltsum(m_none, down) rtol = 0.05
    @test mn(abs.(m_conn.model.U.present[down, ywall])) >
          1.2 * mn(abs.(m_sink.model.U.present[down, ywall]))
    @test mn(m_conn.model.T.present[down, ywall]) >
          mn(m_sink.model.T.present[down, ywall]) + 0.01
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
    g = getfield(m.model, :geometry)
    # Under the default the interior term is live and the gate is exactly 1.0,
    # so nothing is altered anywhere.
    up = Laddie.u_pressure_depth(m.model)
    @test any(!iszero, up[Laddie.ip_count(g.tmask).==2])
    @test all(Laddie._pgf_gate(m.model, Laddie.ip_count(g.tmask)) .== 1)

    # Selecting the truncation is bit-identical in the interior and exactly
    # zero on one-sided faces.
    m_t = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm, params = p_trunc)
    run!(m_t; days = 0.5, verbose = false)
    g_t = getfield(m_t.model, :geometry)
    up_t = Laddie.u_pressure_depth(m_t.model)
    vp_t = Laddie.v_pressure_depth(m_t.model)
    @test all(iszero, up_t[Laddie.ip_count(g_t.tmask).!=2])
    @test all(iszero, vp_t[Laddie.jp_count(g_t.tmask).!=2])

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
    forcing = ISOMIPForcing(:warm; FT)
    gap_grid = Grid(mk, z_draft, FT(2000.0), FT(2000.0); domain_cropping = NoDomainCropping())
    m_gap = Simulation(Model(gap_grid; forcing, params = p_trunc))
    run!(m_gap; days = 0.5, verbose = false)
    g_gap = getfield(m_gap.model, :geometry)
    up_gap = Laddie.u_pressure_depth(m_gap.model)
    vp_gap = Laddie.v_pressure_depth(m_gap.model)
    @test all(iszero, up_gap[Laddie.ip_count(g_gap.tmask).!=2])
    @test all(iszero, vp_gap[Laddie.jp_count(g_gap.tmask).!=2])

    # Guard against the assertions above going vacuous: the gate only has
    # teeth on faces where momentum is actually solved (umask/vmask == 1),
    # and the ISOMIP channel has no such face in y at all — the gap domain
    # must supply both, or this testset stops testing the fix.
    @test count((Laddie.ip_count(g_gap.tmask) .== 1) .& (g_gap.umask .== 1)) > 0
    @test count((Laddie.jp_count(g_gap.tmask) .== 1) .& (g_gap.vmask .== 1)) > 0

    # The choice is not cosmetic: on a domain that has an ice front, the two
    # settings must actually integrate to different states.
    m_full2 = Simulation(Model(gap_grid; forcing, params = p_full))
    run!(m_full2; days = 0.5, verbose = false)
    @test m_full2.model.U.present != m_gap.model.U.present
    @test all(isfinite, m_gap.model.melt) && all(isfinite, m_full2.model.melt)
end
