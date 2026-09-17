# Docstrings and compact show methods.

@testset "Docstrings and display" begin
    d = string(@doc TurbulentGamTMelting)
    @test !occursin("\\T_b", d) && !occursin("\\\\gamma", d)
    @test occursin("\\\\", d)                     # the line breaks survive
    @test sprint(show, Laddie.OceanForcing2D()) == "OceanForcing2D() (not implemented)"
    @test !(:OceanForcing2D in names(Laddie))
end

@testset "Compact show methods" begin
    m = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm)
    plain(x) = sprint(show, MIME("text/plain"), x)

    s = plain(m.model)
    @test occursin("Model{Float64} on CPU", s)
    @test occursin("20×", replace(s, "10×20" => "20×10")) || occursin("interior", s)
    @test occursin("forcing", s) && occursin("params", s)
    @test !occursin("dt", s)   # the model knows nothing about time stepping
    @test occursin("boundary: BoundaryConditions(open ocean = ZeroGradientInflow", s)
    @test length(s) < 800   # not a field dump

    ss = plain(m)
    @test occursin("Simulation{Float64} on CPU", ss)
    @test occursin("dt = 210.0 s", ss) && occursin("FixedDt()", ss)
    @test occursin("Robert–Asselin", ss) && occursin("disabled", ss)
    @test length(ss) < 800
    @test occursin("Simulation{Float64} on CPU at day 0.0", sprint(show, m))

    sg = plain(getfield(m.model, :grid))
    @test occursin("shelf", sg) && occursin("interior", sg) && occursin("gap", sg)
    @test length(sg) < 400
    @test occursin("active cells", sprint(show, getfield(m.model, :geometry)))

    sp = plain(getfield(m.model, :params))
    @test occursin("Params{Float64}", sp)
    @test occursin("g = 9.81", sp) && occursin("entrainment", sp)
    @test !occursin("dt0", sp) && !occursin("time stepper", sp)
    @test length(sp) < 1500

    sf = plain(getfield(m.model, :forcing))
    @test occursin("OceanForcing1D", sf) && occursin("5000-point profile", sf)
    @test occursin("PrescribedIceForcing(T_ice_base = -25.0 °C)", sf)
    for x in (getfield(m.model, :state), getfield(m.model, :cache), m.io, m.clock,
              m.output, m.model.D)
        @test length(plain(x)) < 400
    end
end
