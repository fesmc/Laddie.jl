# Forward-mode AD: ForwardDiff through a whole run, checked against central differences.

# Mean melt rate after a short ISOMIP+ run, as a function of three parameters that act
# through different kernels: bottom drag (momentum), γ_T (melt) and the entrainment
# coefficient.  The model is built at the element type of `x`, so `Dual` inputs
# propagate through the grid, the state and the parameters alike.
function ad_melt(x; tstep = FixedDt(), output = OutputConfig())
    T = eltype(x)
    params = Params(; FT = T, C_d = x[1], melting = FixedGamTMelting(x[2]),
                    entrainment = LambertEntrainment(x[3]))
    sim = build_isomip(CPU(); FT = T, nx = 20, ny = 10, isomipcond = :warm, params,
                       tstep, output)
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

@testset "ForwardDiff through adaptive dt and file output" begin
    # The dt controller runs on primal values, so the derivative is taken at the dt
    # sequence the Float64 run would choose.
    @test all(isfinite, ad_gradient(x -> ad_melt_loss(x; tstep = AdaptiveDt())))
    # NetCDF output, restart files and the log get plain floats.
    tmp = mktempdir()
    output = OutputConfig(; name = "ad", resultdir = tmp, saveday = 1.0, restday = 1.0)
    @test all(isfinite, ad_gradient(x -> ad_melt_loss(x; output)))
    Laddie.NCDatasets.NCDataset(joinpath(tmp, "ad", "output.nc")) do ds
        @test eltype(ds["melt"][:, :, end]) <: Union{Missing,Float64}
    end
end
