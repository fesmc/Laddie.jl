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
    fields = (:Ut, :Vt, :D, :T, :S, :melt, :mask, :z_draft, :ustar, :drho, :convection)
    output = OutputConfig(; name = "extra", resultdir = tmp, saveday = 0.25, fields)
    @test_throws ArgumentError OutputConfig(; fields = (:D, :nope))
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
        @test haskey(ds, "melt") && haskey(ds, "D") && haskey(ds, "T")
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
    @test m2.model.D.present ≈ m.model.D.present
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

    # Typed Model: unknown properties now error instead of landing in a Dict
    @test_throws ErrorException m.model.no_such_field
    @test_throws ErrorException (m.model.no_such_field = 1)
end
