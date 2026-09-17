# Verification against the Python LADDIE reference.

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
    # slopes from ocean/grounded neighbours.  Likewise the Python walls are one
    # partial-slip factor of 1, not the no-slip default.
    py_restart = joinpath(@__DIR__, "..", "docs", "assets", "restart_000001.nc")
    if isfile(py_restart)
        v1_walls = BoundaryConditions(; grounding_line = PartialSlipGL(1.0),
                                      land = PartialSlipLand(1.0))
        m = build_isomip(; isomipcond = :warm, gradient = PyGradient(), boundary = v1_walls)
        run!(m; days = 1.0, verbose = false)

        NCD = Laddie.NCDatasets
        py, py_tmask = NCD.Dataset(py_restart) do ds
            # NCDatasets hands back (x, y, n) with n=2 the present leapfrog
            # level — already Laddie.jl's own [x, y] interior layout.
            get_v(v) = coalesce.(Array(ds[v][:, :, 2]), 0.0)
            Dict(v => get_v(v) for v in ("D", "T", "S", "U", "V")),
            coalesce.(Array(ds["tmask"][:, :]), 0.0)
        end

        inner(a) = a[2:end-1, 2:end-1]
        tm = inner(m.model.tmask) .> 0
        @test all((inner(m.model.tmask) .> 0) .== (py_tmask .> 0))

        #            field  mean|Δ|   max|Δ|       (measured: mean / max)
        tols = Dict("D" => (0.05,    6.0),     # 0.019  / 3.0   m
                    "T" => (0.004,   0.03),    # 0.0015 / 0.013 °C
                    "S" => (0.0015,  0.015),   # 0.0006 / 0.006 psu
                    "U" => (0.0003,  0.04),    # 1.1e-4 / 0.016 m/s
                    "V" => (0.0003,  0.04))    # 0.9e-4 / 0.016 m/s
        for (v, jl) in (("D", m.model.D.present), ("T", m.model.T.present),
                        ("S", m.model.S.present), ("U", m.model.U.present),
                        ("V", m.model.V.present))
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
