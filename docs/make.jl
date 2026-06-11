using Laddie
using Documenter
using Literate

# Run Literate on the example scripts → Documenter-flavoured markdown.
const EXAMPLES = ["forcing.jl", "isomip_run.jl", "cold_run.jl", "spinup.jl", "python_comparison.jl"]
exdir  = joinpath(@__DIR__, "src", "examples")
gendir = joinpath(@__DIR__, "src", "generated")
isdir(gendir) && rm(gendir; recursive = true)
mkpath(gendir)
for ex in EXAMPLES
    Literate.markdown(joinpath(exdir, ex), gendir; documenter = true)
end

DocMeta.setdocmeta!(Laddie, :DocTestSetup, :(using Laddie); recursive=true)

makedocs(;
    modules  = [Laddie],
    authors  = "JanJereczek <jan.jereczek@gmail.com> and contributors",
    sitename = "Laddie.jl",
    format   = Documenter.HTML(;
        canonical = "https://fesmc.github.io/Laddie.jl",
        edit_link = "main",
        assets    = String[],
    ),
    pages = [
        "Home"           => "index.md",
        "Physics"        => "physics.md",
        "Numerics"       => "numerics.md",
        # "Implementation" => "implementation.md",
        "Configuration"  => "configuration.md",
        "Examples" => [
            "ISOMIP+"   => [
                "Forcing"           => "generated/forcing.md",
                "Run"               => "generated/isomip_run.md",
                "Warm vs. cold"     => "generated/cold_run.md",
                "Spin-up"           => "generated/spinup.md",
                "Python validation" => "generated/python_comparison.md",
            ],
        ],
        "API reference" => "API_public.md",
    ],
    warnonly = true,
)

deploydocs(;
    repo      = "github.com/fesmc/Laddie.jl",
    devbranch = "main",
)
