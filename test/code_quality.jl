# Code quality, internal helpers, type stability and the property-forwarding guard.

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(Laddie)
end

@testset "Utility: _safe_div and index wrap-around helpers" begin
    @test Laddie._safe_div(6.0, 2.0) === 3.0
    @test Laddie._safe_div(1.0, 0.0) === 0.0   # zero denominator → zero
    @test Laddie._safe_div(0.0, 0.0) === 0.0   # both zero → zero
    # _west and _south wrap at boundary i/j == 1
    @test Laddie._xm1(1, 10)  == 10
    @test Laddie._xm1(5, 10)  == 4
    @test Laddie._ym1(1, 10) == 10
    @test Laddie._ym1(5, 10) == 4
    # _east and _north wrap at boundary i/j == N
    @test Laddie._xp1(10, 10) == 1
    @test Laddie._xp1(5,  10) == 6
    @test Laddie._yp1(10, 10) == 1
    @test Laddie._yp1(5,  10) == 6
end

@testset "Forcing structs are concretely typed" begin
    forcings = (
        ISOMIPForcing(:warm; FT),
        OceanForcing1D([1.0, 0.0], [34.7, 34.2], [-1000.0, -100.0]; FT),
    )
    for f in forcings
        @test all(isconcretetype, fieldtypes(typeof(f)))
        @test f.Tz isa Vector{FT}
        @test all(isfinite, f.Tz) && all(isfinite, f.Sz)
    end
    # ...and so is the assembled cavity forcing a model actually holds, whose
    # ice field has been materialised onto the grid.
    m = build_isomip(CPU(); FT, nx = 20, ny = 10, isomipcond = :warm)
    cf = getfield(m.model, :forcing)
    @test cf isa CavityForcing
    @test all(isconcretetype, fieldtypes(typeof(cf)))
    @test all(isconcretetype, fieldtypes(typeof(cf.ice)))
    @test cf.ice.T_ice_base isa Matrix{FT}
end

@testset "Model property forwarding: collision guard" begin
    m = build_isomip(CPU(); nx = 20, ny = 10, isomipcond = :warm)
    parts = (getfield(m.model, :grid), getfield(m.model, :geometry), getfield(m.model, :state),
             getfield(m.model, :cache), getfield(m.model, :params),
             getfield(m.model, :boundary))
    v = zeros(2)
    # The guard inspects the members of the CavityForcing, not the wrapper, so
    # a user-defined ocean forcing is what it has to catch.
    @test_throws "ambiguous" Model(
        parts..., CavityForcing(CollidingForcing(v, v, v, 1.0, -5000.0, 0.0)))
    @test_throws "reserved" Model(
        parts..., CavityForcing(ReservedNameForcing(v, v, v, 1.0, -5000.0, 3)))
    # An ice forcing colliding with the ocean side is caught too.
    @test_throws "ambiguous" Model(
        parts...,
        CavityForcing(getfield(m.model, :forcing).ocean, CollidingIceForcing(zeros(2, 2))),
    )
    # The shipped struct combination is collision-free (also checked at
    # every Model construction).
    @test Model(parts..., getfield(m.model, :forcing)) isa Model
end
