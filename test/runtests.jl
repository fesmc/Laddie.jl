# Each file holds the testsets of one area; setup.jl defines what they share.
include("setup.jl")

@testset verbose=true "Laddie.jl" begin
    @testset "Code quality" begin include("code_quality.jl") end
    @testset "Grid and Model" begin include("grid_and_model.jl") end
    @testset "Geometry ingestion" begin include("geometry_ingestion.jl") end
    @testset "Forcing" begin include("forcing.jl") end
    @testset "Params" begin include("params.jl") end
    @testset "Parameterizations" begin include("parameterizations.jl") end
    @testset "Boundary conditions" begin include("boundary_conditions.jl") end
    @testset "Time stepping" begin include("time_stepping.jl") end
    @testset "Numerics" begin include("numerics.jl") end
    @testset "Output" begin include("output.jl") end
    @testset "Display" begin include("display.jl") end
    @testset "Verification" begin include("verification.jl") end
    # Skipped entirely when no CUDA device is present.
    if gpu_backend !== nothing
        @testset "GPU" begin include("gpu.jl") end
    end
end
