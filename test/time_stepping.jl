# Time integration: steppers, run!, CFL, adaptive dt, stopping criteria, the clock.

@testset "Time stepper: FixedDt default/equivalence, AdaptiveDt threading" begin
    # FixedDt is the default; an explicit FixedDt() must reproduce it
    # bit-for-bit so the Python verification stays valid for the default.
    m_def = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
    @test m_def.tstep isa FixedDt
    m_fix = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                         tstep = FixedDt())
    run!(m_def; days = 1.0, verbose = false)
    run!(m_fix; days = 1.0, verbose = false)
    @test m_fix.model.D.present == m_def.model.D.present
    @test m_fix.model.melt == m_def.model.melt

    # AdaptiveDt is a Simulation option, not a Params one; its FT tracks the
    # model's FT (default-constructed at Float64 here, promoted to Float32).
    s = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                     tstep = AdaptiveDt(; cfl_target = 0.4, ncheck = 10))
    @test s.tstep isa AdaptiveDt{FT}
    @test s.tstep.cfl_target ≈ FT(0.4) && s.tstep.ncheck == 10
    @test build_isomip(CPU(); FT = Float32, nx = 20, ny = 10,
                       tstep = AdaptiveDt()).tstep isa AdaptiveDt{Float32}
    @test !hasfield(Params, :tstep) && !hasfield(Params, :dt0) && !hasfield(Params, :nu)

    # Run metadata records the active stepper for both default and adaptive.
    tmp = mktempdir()
    build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                 output = OutputConfig(; name = "tsfix", resultdir = tmp, saveday = 0.5))
    meta = Laddie.TOML.parsefile(joinpath(tmp, "tsfix", "run_metadata.toml"))
    @test meta["simulation"]["time_stepper"]["type"] == "FixedDt"

    build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                 tstep = AdaptiveDt(; cfl_target = 0.4),
                 output = OutputConfig(; name = "tsadp", resultdir = tmp, saveday = 0.5))
    meta2 = Laddie.TOML.parsefile(joinpath(tmp, "tsadp", "run_metadata.toml"))
    @test meta2["simulation"]["time_stepper"]["type"] == "AdaptiveDt"
    @test meta2["simulation"]["time_stepper"]["cfl_target"] ≈ 0.4
end

@testset "run! advances model state" begin
    m = build_isomip(CPU(); nx=20, ny=10, isomipcond=:warm)
    D0 = copy(m.model.D.present)
    run!(m; days=0.5, verbose=false)
    # D should have changed
    @test m.model.D.present != D0
    @test all(isfinite, m.model.D.present)
    @test all(isfinite, m.model.melt)
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
    @test all(isfinite, m.model.melt)
end

@testset "run!: CFL warning and blow-up detection" begin
    # CFL warning fires when dt is too large for the grid; days = 0 → no
    # stepping, so only the pre-loop warning is exercised.
    m = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm, dt = 5000.0)
    @test_logs (:warn, r"CFL") run!(m; days = 0.0, verbose = false)

    # Default ISOMIP+ setup is CFL-safe: no warning.
    m_ok = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm)
    @test_logs run!(m_ok; days = 0.0, verbose = false)

    # Non-finite prognostics abort with an informative error instead of
    # integrating NaNs to the end of the run.
    m2 = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm)
    m2.model.D.present[5, 5] = NaN
    @test_throws "blew up" run!(m2; days = 0.1, verbose = false)
end

@testset "CFL number: matches hand-built states" begin
    m = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
    g, dt, dx, dy = m.model.g, m.clock.dt, m.model.dx, m.model.dy
    cfl(u, v, D, dr) = begin
        m.model.U.present .= u; m.model.V.present .= v
        m.model.D.present .= D; m.model.drho .= dr
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
                         tstep = AdaptiveDt())
    Laddie._init_adaptive_dt!(msafe, msafe.tstep)
    @test msafe.clock.dt <= 210.0
    @test msafe.clock.dt ≈ 210.0 rtol = 0.1
    mbig = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                        dt = 5000.0, tstep = AdaptiveDt())
    Laddie._init_adaptive_dt!(mbig, mbig.tstep)
    @test mbig.clock.dt < 5000.0

    # Warm run completes with dt staying in bounds.
    ma = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                      tstep = AdaptiveDt())
    run!(ma; days = 1.0, verbose = false)
    @test all(isfinite, ma.model.D.present) && all(isfinite, ma.model.melt)
    @test 1.0 <= ma.clock.dt <= 1000.0

    # Stability rescue (headline): a dt0 that blows up under FixedDt is made
    # to survive by the controller.
    #
    # Blow-up can no longer be detected with `isfinite`: the state clamps in
    # `leapfrog_step!` (v_cut on U/V, T ∈ [-5, 5], S ∈ [32, 36], D floored at
    # D_min and capped by max_layer_thickness) bound every prognostic, so an
    # unstable run stays perfectly finite while producing nonsense — at
    # dt = 5000 s the mean melt rate is ~25x the converged value.  Detect it
    # physically instead, against the small-dt reference solution.
    mref = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm, dt = 210.0)
    run!(mref; days = 2.0, verbose = false)
    mean_ref = meltstats(mref)[2]
    survives(days; kw...) = try
        mm = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm, kw...)
        run!(mm; days, verbose = false)
        all(isfinite, mm.model.D.present) && all(isfinite, mm.model.melt) &&
            meltstats(mm)[2] < 3 * mean_ref
    catch
        false
    end
    @test !survives(2.0; dt = 5000.0)                         # FixedDt blows up
    @test  survives(2.0; dt = 5000.0, tstep = AdaptiveDt())   # AdaptiveDt rescues

    # dt changes are logged to log.txt.
    tmp = mktempdir()
    ml = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                      tstep = AdaptiveDt(),
                      output = OutputConfig(; name = "adlog", resultdir = tmp, saveday = 10.0))
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
                      gradient = PyGradient(), tstep = AdaptiveDt())
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
                      tstep = AdaptiveDt())
    run!(cf; days = 1.0, verbose = false)
    run!(ca; days = 1.0, verbose = false)
    @test ca.clock.iteration < 0.9 * cf.clock.iteration
    @test ca.clock.dt > cf.clock.dt          # dt grew above the fixed step

    # Restart round-trip: the current dt is saved and restored, so an
    # adaptive run resumes at exactly the step it left off.
    tmp = mktempdir()
    m1 = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                      tstep = AdaptiveDt(),
                      output = OutputConfig(; name = "ar1", resultdir = tmp,
                                            saveday = 0.5, restday = 0.5))
    run!(m1; days = 1.0, verbose = false)
    m2 = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                      tstep = AdaptiveDt(),
                      output = OutputConfig(; name = "ar2", resultdir = tmp, saveday = 0.5),
                      restart = joinpath(tmp, "ar1", "restart_latest.jld2"))
    @test m2.clock.dt ≈ m1.clock.dt
    @test m2.model.D.present ≈ m1.model.D.present
    run!(m2; days = 0.5, verbose = false)
    @test all(isfinite, m2.model.D.present) && all(isfinite, m2.model.melt)
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
    @test a.model.D.present == b.model.D.present && a.model.melt == b.model.melt &&
          a.clock.iteration == b.clock.iteration

    # Passing both `days` and `until` is ambiguous.
    @test_throws ArgumentError run!(a; days = 1.0, until = FixedSimulationEnd())

    # SteadyStateEnd stops early once the day-over-day mean-melt change drops
    # below tol; a (near-)zero tol never triggers and runs to the t_end cap.
    cap = 20.0
    ms = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
    run!(ms; until = SteadyStateEnd(tol = 0.3, t_end = cap), verbose = false)
    @test ms.clock.time < 0.5 * cap * 86400              # stopped well before the cap
    @test all(isfinite, ms.model.D.present) && all(isfinite, ms.model.melt)

    mc = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
    run!(mc; until = SteadyStateEnd(tol = 1e-12, t_end = cap), verbose = false)
    @test mc.clock.time > 0.9 * cap * 86400              # ran essentially to the cap
    @test ms.clock.iteration < mc.clock.iteration        # early stop took fewer steps

    # `stop` sets the default criterion of run!; an explicit `days` overrides it.
    sd = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                      stop = FixedSimulationEnd(t_end = 0.5))
    run!(sd; verbose = false)
    @test sd.clock.time ≈ 0.5 * 86400 atol = sd.clock.dt
    run!(sd; days = 0.25, verbose = false)
    @test sd.clock.time ≈ 0.75 * 86400 atol = sd.clock.dt
end

@testset "Simulation: clock persists across run! calls" begin
    # Two calls are the same integration as one call covering both: the clock,
    # the iteration count and (under FixedDt) the state all continue.  Each call
    # rounds its own duration to whole steps, so the durations here are exact
    # multiples of dt (0.5 d at 210 s would be 206 + 206 steps against 411).
    one = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
    two = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
    @test one.clock.time == 0 && one.clock.iteration == 0
    half = 200 * 210.0 / 86400
    run!(one; days = 2half, verbose = false)
    run!(two; days = half, verbose = false)
    run!(two; days = half, verbose = false)
    @test one.clock.iteration == 400
    @test two.clock.time == one.clock.time
    @test two.clock.iteration == one.clock.iteration
    @test two.model.D.present == one.model.D.present
    @test two.model.melt == one.model.melt

    # The defect this fixes: successive calls used to restart the day count,
    # stamping output and restart files of the second call with the times of
    # the first.  Output and restarts must now carry continuing times.
    tmp = mktempdir()
    s = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm,
                     output = OutputConfig(; name = "cont", resultdir = tmp,
                                           saveday = 0.5, restday = 10.0))
    run!(s; days = 1.0, verbose = false)
    run!(s; days = 1.0, verbose = false)
    rundir = joinpath(tmp, "cont")
    times = Laddie.NCDatasets.Dataset(ds -> Array(ds["time"][:]), joinpath(rundir, "output.nc"))
    @test issorted(times; lt = <=)                  # strictly increasing, no repeats
    @test times[end] ≈ 2.0 atol = 0.01
    @test isfile(joinpath(rundir, "restart_000001.jld2"))
    @test isfile(joinpath(rundir, "restart_000002.jld2"))
    Laddie.JLD2.jldopen(joinpath(rundir, "restart_latest.jld2"), "r") do f
        @test f["t_days"] ≈ 2.0 atol = 0.01
    end

    # The model carries no time-integration state at all.
    @test_throws ErrorException s.model.dt
    @test !hasfield(typeof(s.model), :io) && !hasfield(typeof(s.model), :config)

    # time_step! is one step of what run! does, without I/O.
    t = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
    for _ in 1:10
        time_step!(t)
    end
    r = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
    run!(r; days = 10 * 210.0 / 86400, verbose = false)
    @test t.clock.iteration == r.clock.iteration == 10
    @test t.model.D.present == r.model.D.present
    @test_throws ArgumentError build_isomip(CPU(); nx = 20, ny = 10, dt = -1.0)
end
