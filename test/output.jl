# Diagnostics and file I/O: meltstats, NetCDF output, logs, restarts.

@testset "meltstats: named fields and total melt in Gt/yr" begin
    sim = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
    run!(sim; days = 0.5, verbose = false)
    m = sim.model
    st = meltstats(sim)
    @test keys(st) == (:max_meltrate, :mean_meltrate, :max_speed, :total_melt)
    mx, mn, sp = st                                # old three-name form still works
    @test (mx, mn, sp) == (st.max_meltrate, st.mean_meltrate, st.max_speed)
    area = sum(m.imask) * m.dx * m.dy
    @test st.total_melt ≈ st.mean_meltrate * area * m.rho_freshwater / 1e12
    @test st.total_melt > 0
end

@testset "I/O: optional ustar/drho/convection fields and real coordinates" begin
    tmp = mktempdir()
    mk = zeros(Int, 22, 12)
    mk[[1, end], :] .= 1; mk[:, [1, end]] .= 1
    mk[2:3, 2:end-1] .= 2; mk[4:20, 2:end-1] .= 3; mk[21, 2:end-1] .= 0
    zd = zeros(22, 12); zd[mk .== 3] .= -300.0
    xs = -1.5e6 .+ 2000.0 .* (0:21)
    grid = Grid(mk, zd; x = xs, y = 3.0e5 .- 2000.0 .* (0:11),
                domain_cropping = NoDomainCropping())
    fields = (:Ut, :Vt, :H, :T, :S, :melt, :mask, :z_draft, :ustar, :drho, :convection,
              :supercooling)
    output = OutputConfig(; name = "extra", resultdir = tmp, saveday = 0.25, fields)
    @test_throws ArgumentError OutputConfig(; fields = (:H, :nope))
    sim = Simulation(Model(grid; forcing = ISOMIPForcing(:warm)); output)
    run!(sim; days = 0.5, verbose = false)
    Laddie.NCDatasets.Dataset(joinpath(tmp, "extra", "output.nc")) do ds
        @test ds["x"][:] == xs[2:end-1]
        @test ds["y"][1] == 3.0e5 - 2000.0
        for v in ("ustar", "drho", "convection")
            a = ds[v][:, :, end]
            @test all(isfinite, a)
        end
        @test maximum(ds["ustar"][:, :, end]) > 0
        @test all(0 .<= ds["convection"][:, :, end] .<= 1)
        # T − Tf(S, z_b) is linear in T and S, so its time mean is that of the means.
        m = sim.model
        wet = m.tmask[2:end-1, 2:end-1] .> 0
        T, S = ds["T"][:, :, end], ds["S"][:, :, end]
        Tf = m.l1 .* S .+ m.l2 .+ m.l3 .* m.z_draft[2:end-1, 2:end-1]
        @test ds["supercooling"][:, :, end][wet] ≈ (T .- Tf)[wet]
    end
end

@testset "I/O: NetCDF output, log, and JLD2 restart round-trip" begin
    tmpdir = mktempdir()
    output = OutputConfig(; name = "iotest", resultdir = tmpdir,
                          saveday = 0.5, diagday = 0.5, restday = 0.5)
    m = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm, output)
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
    @test meta["simulation"]["dt0"] == 210.0
    @test meta["simulation"]["nu"] ≈ 0.8
    @test meta["simulation"]["cfl"]["type"] == "ExactCFL"
    @test meta["simulation"]["restart"] == ""
    @test !haskey(meta["params"], "dt0") && !haskey(meta["params"], "time_stepper")
    @test meta["params"]["melting"]["type"] == "FixedGamTMelting"
    @test meta["params"]["melting"]["gamTfix"] ≈ 0.00018
    @test meta["params"]["momentum_advection"]["type"] == "CentredMomentumAdvection"
    @test meta["boundary"]["wall_advection"]["type"] == "SlipScaledWallAdvection"
    @test meta["boundary"]["grounding_line"]["type"] == "NoSlipGL"
    @test meta["boundary"]["gaps"]["type"] == "SinkGapsBC"
    @test !haskey(meta["params"], "grounding_line")
    # The forcing entry is split ocean/ice, and records the profile ranges, so that
    # a warm run is distinguishable from a cold one in the metadata.
    @test meta["forcing"]["ocean"]["type"] == "OceanForcing1D"
    @test meta["forcing"]["ocean"]["Tz_range"][2] ≈ 18.23888888888889
    @test meta["forcing"]["ocean"]["z_range"][1] < 0
    @test meta["forcing"]["ice"]["type"] == "PrescribedIceForcing"
    @test meta["forcing"]["ice"]["T_ice_base_range"] == [-25.0, -25.0]
    @test meta["output"]["saveday"] == 0.5

    # All output is written into a single output.nc with a time dimension;
    # saveday=0.5 over 1 day produces at least 2 time slices.
    NCD = Laddie.NCDatasets
    out_path = joinpath(rundir, "output.nc")
    @test isfile(out_path)
    melt_out = NCD.Dataset(out_path) do ds
        @test haskey(ds, "melt") && haskey(ds, "H") && haskey(ds, "T")
        @test length(ds["time"]) >= 2
        coalesce.(Array(ds["melt"][:, :, end]), NaN)
    end
    @test any(isfinite, melt_out)
    @test maximum(filter(isfinite, melt_out)) > 0   # m/yr, warm cavity melts

    # Restart written, then round-trips: a model restarted from it must
    # carry the same prognostic state.
    @test isfile(joinpath(rundir, "restart_latest.jld2"))
    restartfile = joinpath(rundir, "restart_latest.jld2")
    m2 = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm,
                      output = OutputConfig(; name = "iotest2", resultdir = tmpdir, saveday = 0.5),
                      restart = restartfile)
    @test m2.clock.time / 86400 ≈ 1.0 atol = 0.01
    @test m2.model.H.present ≈ m.model.H.present
    @test m2.model.T.present ≈ m.model.T.present
    @test m2.model.S.present ≈ m.model.S.present

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
    @test meta2["simulation"]["restart"] == restartfile

    # A restart loads into a model of another precision, and a file from before
    # the D → H rename (thickness stored as "D") still loads.
    m_ref = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm, restart = restartfile)
    m32 = build_isomip(CPU(); FT = Float32, nx = 20, ny = 10, isomipcond = :warm,
                       restart = restartfile)
    @test eltype(m32.model.H.present) == Float32
    @test m32.model.T.present ≈ Float32.(m_ref.model.T.present)
    legacy = joinpath(tmpdir, "legacy_D.jld2")
    Laddie.JLD2.jldopen(restartfile, "r") do f
        Laddie.JLD2.jldsave(legacy; t_days = f["t_days"], dt = f["dt"], D = f["H"],
                            U = f["U"], V = f["V"], T = f["T"], S = f["S"])
    end
    m_legacy = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm, restart = legacy)
    @test m_legacy.model.H.present == m_ref.model.H.present

    # Typed Model: unknown properties now error instead of landing in a Dict
    @test_throws ErrorException m.model.no_such_field
    @test_throws ErrorException (m.model.no_such_field = 1)
end

@testset "Restart: refine a coarse run onto a finer grid" begin
    # The same ISOMIP+ channel at 4 km and at 2 km: the full domains satisfy
    # fine = 2 (coarse − 2) + 2, and the default cropping crops them differently.
    tmp = mktempdir()
    crop = MinRectangleDomainCropping()
    coarse = build_isomip(CPU(); nx = 20, ny = 6, dx = 4000.0, dy = 4000.0, xfront = 60_000.0,
                          domain_cropping = crop,
                          output = OutputConfig(; name = "coarse", resultdir = tmp, saveday = 0.5))
    run!(coarse; days = 0.5, verbose = false)
    file = joinpath(tmp, "coarse", "restart_latest.jld2")
    Laddie.JLD2.jldopen(file, "r") do f
        @test haskey(f, "tmask") && f["input_size"] == coarse.model.grid.input_size
    end
    fine_sim() = build_isomip(CPU(); nx = 40, ny = 12, dx = 2000.0, dy = 2000.0, xfront = 60_000.0,
                              domain_cropping = crop)
    fine = fine_sim()
    Laddie.init_from_restart!(fine, file; refine = 2)
    @test fine.clock.time ≈ coarse.clock.time
    @test fine.clock.dt ≈ coarse.clock.dt / 2

    # Before the bootstrap step moved `present`, every fine cell held its coarse
    # parent's state; `past` is not touched by the bootstrap, so check it there.
    gc, gf = coarse.model.grid, fine.model.grid
    @test first(gc.crop[1]) > 1 && first(gf.crop[1]) != 2 * (first(gc.crop[1]) - 2) + 2   # crops differ
    parent(i, nf, nc) = i == 1 ? 1 : i == nf ? nc : (i - 2) ÷ 2 + 2
    Hc = Laddie.JLD2.load(file, "H").past
    tm_c = Laddie.JLD2.load(file, "tmask")
    tm_f = fine.model.tmask .> 0
    checked = 0
    for (jj, j) in enumerate(gf.crop[2]), (ii, i) in enumerate(gf.crop[1])
        pi = parent(i, gf.input_size[1], gc.input_size[1]) - first(gc.crop[1]) + 1
        pj = parent(j, gf.input_size[2], gc.input_size[2]) - first(gc.crop[2]) + 1
        (1 <= pi <= size(Hc, 1) && 1 <= pj <= size(Hc, 2) && tm_c[pi, pj] && tm_f[ii, jj]) || continue
        @test fine.model.H.past[ii, jj] == Hc[pi, pj]
        checked += 1
    end
    @test checked > 0.9 * count(tm_f)
    run!(fine; days = 0.2, verbose = false)
    @test all(isfinite, fine.model.H.present) && all(isfinite, fine.model.melt)

    # A file without the grid (written before it was stored) needs `coarse_grid`.
    legacy = joinpath(tmp, "legacy.jld2")
    Laddie.JLD2.jldopen(file, "r") do f
        Laddie.JLD2.jldsave(legacy; t_days = f["t_days"], dt = f["dt"], H = f["H"],
                            U = f["U"], V = f["V"], T = f["T"], S = f["S"])
    end
    @test_throws ArgumentError Laddie.init_from_restart!(fine_sim(), legacy; refine = 2)
    f2 = Laddie.init_from_restart!(fine_sim(), legacy; refine = 2, coarse_grid = gc)
    @test f2.model.H.past == Laddie.init_from_restart!(fine_sim(), file; refine = 2).model.H.past

    # Mismatches are caught: a plain restart onto the finer grid, and a factor that
    # does not refine the coarse domain.
    @test_throws ArgumentError Laddie.init_from_restart!(fine_sim(), file)
    @test_throws ArgumentError Laddie.init_from_restart!(fine_sim(), file; refine = 3)
end
