# Diagnostics and file I/O: meltstats, NetCDF output, logs, restarts.

@testset "meltstats: named fields and total melt in Gt/yr" begin
    sim = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
    run!(sim; days = 0.5, verbose = false)
    m = sim.model
    st = meltstats(sim)
    @test keys(st) == (:max_melt, :mean_melt, :max_speed, :total_melt, :refreezing)
    area = sum(m.imask) * m.dx * m.dy
    @test st.total_melt ≈ st.mean_melt * area * m.rho_freshwater / 1e12
    @test st.total_melt > 0
    @test st.refreezing == 0                       # warm ISOMIP+ melts everywhere
end

@testset "meltstats: net melt and refreezing close the melt budget" begin
    # Start supercooled under frazil formation, so that the base refreezes and frazil
    # forms: the net melt and the refreezing then add up to the melting alone.
    sim = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                       params = Params(; FT, frazil_ice = RelaxedFrazilIce()))
    supercool!(sim.model, FT(0.05))
    Laddie.advance_leapfrog!(sim.model, sim.clock.dt)   # the supercooled levels become past/present
    m = sim.model
    st = meltstats(sim)
    ice = m.imask .> 0
    to_gt = m.seconds_per_year * m.dx * m.dy * m.rho_freshwater / 1e12
    @test any(m.melt[ice] .< 0) && any(m.frazil[ice] .> 0)
    @test st.refreezing > 0
    @test st.total_melt + st.refreezing ≈ sum(max.(m.melt[ice], 0)) * to_gt
    @test st.total_melt ≈ sum(m.melt[ice] .- m.frazil[ice]) * to_gt
    @test st.max_melt == maximum(m.melt[ice] .- m.frazil[ice]) * m.seconds_per_year
end

@testset "I/O: optional ustar/drho/convective_instability fields and real coordinates" begin
    tmp = mktempdir()
    mk = zeros(Int, 22, 12)
    mk[[1, end], :] .= 1; mk[:, [1, end]] .= 1
    mk[2:3, 2:end-1] .= 2; mk[4:20, 2:end-1] .= 3; mk[21, 2:end-1] .= 0
    zd = zeros(22, 12); zd[mk .== 3] .= -300.0
    xs = -1.5e6 .+ 2000.0 .* (0:21)
    grid = Grid(mk, zd; x = xs, y = 3.0e5 .- 2000.0 .* (0:11),
                domain_cropping = NoDomainCropping())
    fields = (:Ut, :Vt, :H, :T, :S, :melt, :mask, :z_draft, :ustar, :drho, :convective_instability,
              :supercooling)
    # Float64 maps: the supercooling identity below holds to Float64 rounding.
    output = OutputConfig(; name = "extra", resultdir = tmp, saveday = 0.25, fields,
                          float_type = Float64)
    @test_throws ArgumentError OutputConfig(; fields = (:H, :nope))
    @test_throws ArgumentError OutputConfig(; float_type = Float16)
    sim = Simulation(Model(grid; forcing = ISOMIPForcing(:warm)); output)
    run!(sim; days = 0.5, verbose = false)
    Laddie.NCDatasets.Dataset(joinpath(tmp, "extra", "output.nc")) do ds
        @test ds["x"][:] == xs[2:end-1]
        @test ds["y"][1] == 3.0e5 - 2000.0
        m = sim.model
        wet = m.tmask[2:end-1, 2:end-1] .> 0
        for v in ("ustar", "drho", "convective_instability")
            a = ds[v][:, :, end]
            @test all(isfinite, a[wet])
            @test all(ismissing, a[.!wet])   # the NaN fill value outside the active cells
        end
        @test maximum(ds["ustar"][:, :, end][wet]) > 0
        @test all(0 .<= ds["convective_instability"][:, :, end][wet] .<= 1)
        # T − Tf(S, z_b) is linear in T and S, so its time mean is that of the means.
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
    meta = TOML.parsefile(meta_path)
    @test meta["run"]["float_type"] == "Float64"
    @test meta["run"]["backend"] == "CPU"
    @test meta["run"]["laddie_version"] isa String
    @test meta["grid"]["nx"] == 20 && meta["grid"]["ny"] == 10
    @test meta["simulation"]["dt0"] == 210.0
    @test !haskey(meta["simulation"], "nu")
    @test meta["params"]["integrator"]["type"] == "LeapfrogIntegrator"
    @test meta["params"]["integrator"]["nu"] ≈ 0.8
    @test meta["simulation"]["cfl"]["type"] == "ExactCFL"
    @test meta["simulation"]["restart"] == ""
    @test !haskey(meta["params"], "dt0") && !haskey(meta["params"], "time_stepper")
    @test meta["params"]["melting"]["type"] == "TurbulentGamTMelting"
    @test meta["params"]["melting"]["Pr"] ≈ 13.8
    @test meta["params"]["momentum_advection"]["type"] == "UpstreamMomentumAdvection"
    @test meta["params"]["integrator"]["closure_level"]["type"] == "PastClosureLevel"
    @test meta["boundary"]["wall_advection"]["type"] == "NoWallAdvection"
    @test meta["boundary"]["grounding_line_slip"]["type"] == "PartialSlipAtGL"
    @test meta["boundary"]["gaps"]["type"] == "SinkGapsBC"
    @test !haskey(meta["params"], "grounding_line_slip")
    # The forcing entry is split ocean/ice, and records the profile ranges, so that
    # a warm run is distinguishable from a cold one in the metadata.
    @test meta["forcing"]["ocean"]["type"] == "OceanForcing1D"
    @test meta["forcing"]["ocean"]["Tz_range"][2] ≈ 18.23888888888889
    @test meta["forcing"]["ocean"]["z_range"][1] < 0
    @test meta["forcing"]["ice"]["type"] == "PrescribedIceForcing"
    @test meta["forcing"]["ice"]["T_ice_farfield_range"] == [-25.0, -25.0]
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

    # Default maps: Float32, compressed losslessly, NaN outside the active cells, and
    # the Float32 rounding of the maps a Float64 output writes for the same run.
    @test meta["output"]["float_type"] == "Float32"
    output64 = OutputConfig(; name = "iotest64", resultdir = tmpdir, saveday = 0.5,
                            float_type = Float64)
    m64 = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm, output = output64)
    run!(m64; days = 1.0, verbose = false)
    wet = m.model.tmask[2:end-1, 2:end-1] .> 0
    maps(path) = NCD.Dataset(path) do ds
        Dict(v => Array(ds[v][:, :, :]) for v in ("melt", "H", "T", "S", "Ut", "Vt"))
    end
    maps32, maps64 = maps(out_path), maps(joinpath(tmpdir, "iotest64", "output.nc"))
    NCD.Dataset(out_path) do ds
        @test NCD.deflate(ds["melt"].var) == (true, true, 1)   # shuffle, deflate, level
    end
    for (v, a) in maps32
        @test eltype(a) == Union{Missing,Float32}
        @test all(ismissing, a[.!wet, :])
        @test all(!ismissing, a[wet, :])
        @test a[wet, :] == Float32.(maps64[v][wet, :])
    end

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
    JLD2.jldopen(latest2, "r") do f
        @test f["t_days"] ≈ 1.5 atol = 0.01   # chained restarts accumulate
    end

    # Continuation metadata records the restart offset
    meta2 = TOML.parsefile(joinpath(rundir2, "run_metadata.toml"))
    @test meta2["run"]["t_start_days"] ≈ 1.0 atol = 0.01
    @test meta2["simulation"]["restart"] == restartfile

    # A restart loads into a model of another precision.
    m_ref = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm, restart = restartfile)
    m32 = build_isomip(CPU(); FT = Float32, nx = 20, ny = 10, isomipcond = :warm,
                       restart = restartfile)
    @test eltype(m32.model.H.present) == Float32
    @test m32.model.T.present ≈ Float32.(m_ref.model.T.present)

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
    JLD2.jldopen(file, "r") do f
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
    Hc = JLD2.load(file, "H").past
    tm_c = JLD2.load(file, "tmask")
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

    # refine_fill = :nearest: the active fine cells without an active coarse parent take H, T
    # and S of the nearest parented cell through the active cells, so each holds the values of
    # an active neighbour; velocities stay zero there, and the parented cells are as with :keep.
    # The two ISOMIP grids nest exactly, so a copy of the restart marks one interior column of
    # active coarse cells inactive: their fine children have no active parent.
    cut = joinpath(tmp, "cut.jld2")
    restart = JLD2.load(file)
    rows = first.(Tuple.(findall(tm_c)))
    column = (minimum(rows) + maximum(rows)) ÷ 2
    restart["tmask"][column, :] .= false
    JLD2.jldsave(cut; (Symbol(k) => v for (k, v) in restart)...)
    kept = Laddie.init_from_restart!(fine_sim(), cut; refine = 2).model
    near = Laddie.init_from_restart!(fine_sim(), cut; refine = 2, refine_fill = :nearest).model
    tm_cut = restart["tmask"]
    parented = falses(size(tm_f))
    for (jj, j) in enumerate(gf.crop[2]), (ii, i) in enumerate(gf.crop[1])
        pi = parent(i, gf.input_size[1], gc.input_size[1]) - first(gc.crop[1]) + 1
        pj = parent(j, gf.input_size[2], gc.input_size[2]) - first(gc.crop[2]) + 1
        parented[ii, jj] = 1 <= pi <= size(Hc, 1) && 1 <= pj <= size(Hc, 2) && tm_cut[pi, pj] && tm_f[ii, jj]
    end
    orphans = findall(tm_f .& .!parented)
    @test !isempty(orphans)
    for q in (:H, :T, :S)
        @test getproperty(near, q).past[parented] == getproperty(kept, q).past[parented]
    end
    neighbours(I) = [I + d for d in CartesianIndex.(((1, 0), (-1, 0), (0, 1), (0, -1)))
                     if checkbounds(Bool, tm_f, I + d) && tm_f[I + d]]
    for I in orphans, q in (:H, :T, :S)
        level = getproperty(near, q).past
        @test any(K -> level[K] == level[I], neighbours(I))
    end
    @test all(near.H.past[I] != kept.H.past[I] for I in orphans)
    @test all(near.U.past[I] == 0 && near.V.past[I] == 0 for I in orphans)
    @test_throws ArgumentError Laddie.init_from_restart!(fine_sim(), file; refine = 2, refine_fill = :nope)

    run!(fine; days = 0.2, verbose = false)
    @test all(isfinite, fine.model.H.present) && all(isfinite, fine.model.melt)

    # Mismatches are caught: a plain restart onto the finer grid, and a factor that
    # does not refine the coarse domain.
    @test_throws ArgumentError Laddie.init_from_restart!(fine_sim(), file)
    @test_throws ArgumentError Laddie.init_from_restart!(fine_sim(), file; refine = 3)

    # A fine domain one cell longer at the far end (a source grid's remainder that the
    # coarse grid dropped): the extra column, open ocean, has no coarse parent.
    long = build_isomip(CPU(); nx = 41, ny = 12, dx = 2000.0, dy = 2000.0, xfront = 60_000.0,
                        domain_cropping = crop)
    @test long.model.grid.input_size[1] == gf.input_size[1] + 1
    Laddie.init_from_restart!(long, file; refine = 2)
    # Where both grids have the cell (all but the far-end column, a border ring on the
    # exact grid), the refined state is the one of the exact-size fine grid.
    gl = long.model.grid
    Hf = Laddie.init_from_restart!(fine_sim(), file; refine = 2).model.H.past
    compared = 0
    for (jj, j) in enumerate(gl.crop[2]), (ii, i) in enumerate(gl.crop[1])
        (i < gf.input_size[1] && i in gf.crop[1] && j in gf.crop[2]) || continue
        @test long.model.H.past[ii, jj] == Hf[i - first(gf.crop[1]) + 1, j - first(gf.crop[2]) + 1]
        compared += 1
    end
    @test compared > 0.9 * count(tm_f)
    run!(long; days = 0.2, verbose = false)
    @test all(isfinite, long.model.H.present)
    # More than f − 1 extra cells is not a refinement of the coarse domain.
    @test_throws ArgumentError Laddie.init_from_restart!(
        build_isomip(CPU(); nx = 42, ny = 12, dx = 2000.0, dy = 2000.0, xfront = 60_000.0,
                     domain_cropping = crop), file; refine = 2)
end

@testset "Output cadence validation and restart names" begin
    @test_throws ArgumentError OutputConfig(; saveday = 1.0, restday = 0.0)
    @test_throws ArgumentError OutputConfig(; saveday = 1.0, diagday = 0.0)
    @test_throws ArgumentError OutputConfig(; saveday = -1.0)
    @test OutputConfig().saveday == 0   # saveday = 0 still disables file I/O

    # Whole days keep their name, also when a fixed dt misses the day a little.
    @test Laddie._restart_filename(1.0) == "restart_000001.jld2"
    @test Laddie._restart_filename(1.998) == "restart_000002.jld2"

    # Restarts less than a day apart must not overwrite each other.
    tmp = mktempdir()
    sim = build_isomip(CPU(); nx = 20, ny = 10,
                       output = OutputConfig(; name = "subday", resultdir = tmp, saveday = 0.25, restday = 0.25))
    run!(sim; days = 0.75, verbose = false)
    files = filter(startswith("restart_0"), readdir(joinpath(tmp, "subday")))
    @test length(files) == 3
    @test "restart_000000.250.jld2" in files
end

@testset "I/O: time series of shelf means" begin
    NCD = Laddie.NCDatasets
    tmp = mktempdir()
    @test_throws ArgumentError OutputConfig(; saveday = 1.0, tsday = -1.0)
    @test_throws ArgumentError OutputConfig(; saveday = 1.0, tsday = 0)
    @test OutputConfig().tsday === nothing          # off by default
    @test OutputConfig(; tsday = 1).tsday === 1.0   # same float type as the other intervals

    # Same window as the maps: each series point is the mean over the ice-covered
    # cells of the time-averaged map it shares its window with.
    # Float64 maps, so that the series (reduced in Float64) match them to Float64 rounding.
    output = OutputConfig(; name = "series", resultdir = tmp, saveday = 0.25, tsday = 0.25,
                          fields = (:H, :T, :S, :melt, :supercooling), float_type = Float64)
    sim = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm, output)
    run!(sim; days = 1.0, verbose = false)
    m = sim.model
    ice = m.imask[2:end-1, 2:end-1] .> 0
    NCD.Dataset(joinpath(tmp, "series", "output.nc")) do ds
        @test ds["ts_time"][:] ≈ ds["time"][:]
        @test length(ds["ts_time"]) == 5
        # No frazil: the net melt is the map of `melt`.
        for (ts, map) in (("ts_H", "H"), ("ts_T", "T"), ("ts_S", "S"), ("ts_mean_melt", "melt"),
                          ("ts_tf", "supercooling"))
            for k in 1:5
                @test ds[ts][k] ≈ sum(ds[map][:, :, k][ice]) / count(ice)
            end
        end
        @test ds["ts_mean_melt"][end] > 0
        @test ds["ts_total_melt"][:] ≈
              ds["ts_mean_melt"][:] .* count(ice) .* m.dx .* m.dy .* m.rho_freshwater ./ 1e12
        @test all(ds["ts_refreezing"][:] .== 0)
        @test all(isfinite, ds["ts_u_norm"][:])
        @test ds["ts_u_norm"][end] > 0
        @test ds["ts_total_melt"].attrib["units"] == "Gt yr-1"
        @test ds["ts_refreezing"].attrib["units"] == "Gt yr-1"
    end

    # Finer than the maps, with the final partial window flushed at the end of run!.
    output = OutputConfig(; name = "fine", resultdir = tmp, saveday = 0.5, tsday = 0.25)
    sim = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm, output)
    run!(sim; days = 1.1, verbose = false)
    NCD.Dataset(joinpath(tmp, "fine", "output.nc")) do ds
        @test length(ds["time"]) == 4
        @test ds["ts_time"][:] ≈ [0.0, 0.25, 0.5, 0.75, 1.0, 1.1] atol = 0.01
    end

    # No series without tsday.
    output = OutputConfig(; name = "noseries", resultdir = tmp, saveday = 0.5)
    sim = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm, output)
    @test isempty(sim.io.tsacc)
    run!(sim; days = 0.5, verbose = false)
    NCD.Dataset(joinpath(tmp, "noseries", "output.nc")) do ds
        @test !haskey(ds, "ts_time") && !haskey(ds, "ts_mean_melt")
    end

    # Rejected under Reactant before anything is moved (no extension needed).
    output = OutputConfig(; name = "rx", resultdir = tmp, saveday = 0.5, tsday = 0.25)
    sim = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm, output)
    @test_throws ArgumentError to_backend(sim, ReactantBackend())
end

@testset "Restart: a continuation equals the straight run, to the last bit" begin
    # 216 s divides a quarter day, so both runs take the same 200 steps.
    tmp = mktempdir()
    levels(m) = [copy(getfield(v, l)) for v in (m.H, m.U, m.V, m.T, m.S) for l in (:past, :present, :future)]
    for (label, params) in (("default", Params()), ("reset", Params(; convection = ResetToAmbientConvection(0.005))))
        build(name; kw...) = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm, dt = 216.0, params,
                                          output = OutputConfig(; name, resultdir = tmp, saveday = 0.25), kw...)
        straight = build("straight-$label")
        run!(straight; days = 0.5, verbose = false)
        first = build("first-$label")
        run!(first; days = 0.25, verbose = false)
        file = joinpath(tmp, "first-$label", "restart_latest.jld2")
        second = build("second-$label"; restart = file)
        @test second.model.H.present == first.model.H.present
        run!(second; days = 0.25, verbose = false)
        @test second.clock.time == straight.clock.time
        @test levels(second.model) == levels(straight.model)
    end

    # A different dt under FixedTimeStepper: kept, and the leapfrog restarts (logged).
    file = joinpath(tmp, "first-default", "restart_latest.jld2")
    out = OutputConfig(; name = "other-dt", resultdir = tmp, saveday = 0.25)
    sim = @test_logs (:info, r"restarting the leapfrog at this simulation's 180.0 s") build_isomip(
        CPU(); nx = 20, ny = 10, isomipcond = :warm, dt = 180.0, output = out, restart = file)
    @test sim.clock.dt == 180.0
    @test occursin("restart written at dt = 216.0 s", read(joinpath(tmp, "other-dt", "log.txt"), String))
    # AdaptiveTimeStepper resumes at the file's dt.
    sim = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm, dt = 180.0,
                       time_stepper = AdaptiveTimeStepper(), restart = file)
    @test sim.clock.dt == 216.0

    # The stored active cells must match the model's.
    m = sim.model
    active = Array(m.tmask) .> 0
    @test Laddie._check_restart_tmask(m, active, file) === nothing
    other = copy(active)
    other[findfirst(active)] = false
    @test_throws ArgumentError Laddie._check_restart_tmask(m, other, file)
end
