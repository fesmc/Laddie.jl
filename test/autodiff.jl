# Forward-mode AD: ForwardDiff through a whole run, checked against central differences.

# Mean melt rate after a short ISOMIP+ run, as a function of three parameters that act
# through different kernels: bottom drag (momentum), γ_T (melt) and the entrainment
# coefficient.  The model is built at the element type of `x`, so `Dual` inputs
# propagate through the grid, the state and the parameters alike.
function ad_melt(x; tstep = FixedDt(), output = OutputConfig(), kinks = ExactKinks(),
                 schemes = (;), isomipcond = :warm, supercooling = 0.0)
    T = eltype(x)
    params = Params(; FT = T, C_d = x[1], melting = FixedGamTMelting(x[2]),
                    entrainment = GasparEntrainment(x[3]), kinks, schemes...)
    sim = build_isomip(CPU(); FT = T, nx = 20, ny = 10, isomipcond, params,
                       tstep, output)
    iszero(supercooling) || supercool!(sim.model, T(supercooling))
    run!(sim; days = 2.0, verbose = false)
    return sim.model.melt
end
ad_melt_loss(x; kw...) = (melt = ad_melt(x; kw...); sum(melt) / length(melt))

# Every closure below differentiates with this one tag, so all runs share a single
# `Dual` type and the kernels compile once.
const AD_X0 = [2.5e-3, 1.8e-4, 2.5]
const AD_CFG = ForwardDiff.GradientConfig(nothing, AD_X0, ForwardDiff.Chunk{3}(),
                                          ForwardDiff.Tag(ad_melt_loss, Float64))
ad_gradient(f) = ForwardDiff.gradient(f, AD_X0, AD_CFG, Val(false))

@testset "ForwardDiff gradient matches central differences" begin
    x0 = AD_X0
    primal = Ref{Matrix{Float64}}()
    g = ad_gradient() do x
        melt = ad_melt(x)
        primal[] = ForwardDiff.value.(melt)
        sum(melt) / length(melt)
    end
    @test all(isfinite, g)
    # The primal of the dual run is the Float64 run, bit for bit.  (Compared on
    # the field: a Float64 `sum` reassociates under SIMD, a Dual one does not.)
    @test primal[] == ad_melt(x0)
    # The loss has kinks (max/clamp floors) close to x0, so the step must stay small:
    # a relative step of 1e-3 already crosses one for γ_T.
    for k in eachindex(x0)
        h = 1e-6 * x0[k]
        e = zeros(length(x0)); e[k] = h
        fd = (ad_melt_loss(x0 .+ e) - ad_melt_loss(x0 .- e)) / 2h
        @test g[k] ≈ fd rtol = 1e-5
    end
end

# Non-default schemes, each in its own run: a gradient test per scheme (AGENTS.md).
# ISOMIP+ is never supercooled on its own, so the frazil case starts 0.05 K below the
# freezing point under the ice; without that it would test nothing.  The conduction
# schemes differ only where the base refreezes, and the convection schemes only where the
# layer is denser than the ambient water, so those cases start supercooled too.  A case
# whose scheme is the default names a `reference` scheme it must differ from instead.
const AD_SCHEME_CASES = (
    "RelaxedFrazilIce" => (; schemes = (; frazil_ice = RelaxedFrazilIce()), supercooling = 0.05),
    "AdvectiveMomentumForm" => (; schemes = (; momentum_form = AdvectiveMomentumForm())),
    "AdvectiveTracerForm" => (; schemes = (; tracer_form = AdvectiveTracerForm())),
    "PresentClosures" => (; schemes = (; closure_level = PresentClosures())),
    "ResetToAmbient" => (; schemes = (; convection_scheme = ResetToAmbient()), supercooling = 0.05),
    "RelaxToAmbient, convecting" => (; schemes = (; convection_scheme = RelaxToAmbient()), supercooling = 0.05,
                                     reference = (; convection_scheme = ClampDensity())),
    "HollandJenkinsConduction" =>
        (; schemes = (; ice_conduction = HollandJenkinsConduction()), supercooling = 0.05),
    "MeltOnlyConduction, refreezing" => (; schemes = (; ice_conduction = MeltOnlyConduction()), supercooling = 0.05,
                                         reference = (; ice_conduction = NoConduction())),
)

@testset "ForwardDiff through $name matches central differences" for (name, case) in AD_SCHEME_CASES
    loss(x) = ad_melt_loss(x; case.schemes, supercooling = get(case, :supercooling, 0.0))
    # The scheme acts: the loss differs from the reference's (by default, the default
    # schemes) under the same start.
    @test loss(AD_X0) != Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
        ad_melt_loss(AD_X0; schemes = get(case, :reference, (;)), supercooling = get(case, :supercooling, 0.0))
    end
    g = ad_gradient(loss)
    @test all(isfinite, g)
    for k in eachindex(AD_X0)
        h = 1e-6 * AD_X0[k]
        e = zeros(length(AD_X0)); e[k] = h
        fd = (loss(AD_X0 .+ e) - loss(AD_X0 .- e)) / 2h
        @test g[k] ≈ fd rtol = 1e-5
    end
end

@testset "ForwardDiff through SmoothKinks matches central differences" begin
    # With the kinks smoothed, a step a hundred times larger than above still agrees.
    g = ad_gradient(x -> ad_melt_loss(x; kinks = SmoothKinks()))
    for k in eachindex(AD_X0)
        h = 1e-4 * AD_X0[k]
        e = zeros(length(AD_X0)); e[k] = h
        fd = (ad_melt_loss(AD_X0 .+ e; kinks = SmoothKinks()) - ad_melt_loss(AD_X0 .- e; kinks = SmoothKinks())) / 2h
        @test g[k] ≈ fd rtol = 1e-5
    end
end

@testset "ForwardDiff through adaptive dt and file output" begin
    # The dt controller runs on primal values, so the derivative is taken at the dt
    # sequence the Float64 run would choose.
    @test all(isfinite, ad_gradient(x -> ad_melt_loss(x; tstep = AdaptiveDt())))
    # NetCDF output, restart files and the log get plain floats.
    tmp = mktempdir()
    output = OutputConfig(; name = "ad", resultdir = tmp, saveday = 1.0, restday = 1.0)
    @test all(isfinite, ad_gradient(x -> ad_melt_loss(x; output)))
    Laddie.NCDatasets.NCDataset(joinpath(tmp, "ad", "output.nc")) do ds
        @test eltype(ds["melt"][:, :, end]) <: Union{Missing,Float32}
    end
end

@testset "ForwardDiff through the RoquetEOS parameters" begin
    # Cb, Θ₀ and T_h, scaled so that the base point AD_X0 is the paper's values; the
    # tag and the Dual type are those of the closures above.
    function ad_eos_loss(x)
        T = eltype(x)
        eos = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            RoquetEOS(T(0.011) * x[1] / 2.5e-3, T(-4.5) * x[2] / 1.8e-4, T(2.5e-5) * x[3] / 2.5, T(0.77))
        end
        params = Params(; FT = T, equation_of_state = eos)
        sim = build_isomip(CPU(); FT = T, nx = 20, ny = 10, isomipcond = :warm, params)
        run!(sim; days = 2.0, verbose = false)
        return sum(sim.model.melt) / length(sim.model.melt)
    end
    g = ad_gradient(ad_eos_loss)
    @test all(isfinite, g)
    @test any(!iszero, g)
    for k in eachindex(AD_X0)
        h = 1e-6 * AD_X0[k]
        e = zeros(length(AD_X0)); e[k] = h
        fd = (ad_eos_loss(AD_X0 .+ e) - ad_eos_loss(AD_X0 .- e)) / 2h
        @test g[k] ≈ fd rtol = 1e-4
    end
end
