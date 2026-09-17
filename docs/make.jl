using Laddie
using Documenter
using Literate

# Run Literate on the example scripts → Documenter-flavoured markdown.
const EXAMPLES = ["isomip.jl"]
exdir  = joinpath(@__DIR__, "src", "examples")
gendir = joinpath(@__DIR__, "src", "generated")
isdir(gendir) && rm(gendir; recursive = true)
mkpath(gendir)
for ex in EXAMPLES
    Literate.markdown(joinpath(exdir, ex), gendir; documenter = true)
end

# Crosson–Dotson needs local data and ~10 min, so its page shows the code without running
# it. Set LADDIE_DOCS_CROSSON_DOTSON=true to run it as a regression check (it asserts its
# tolerances and rewrites the committed figure).
cd_script = joinpath(exdir, "crosson-dotson.jl")
if get(ENV, "LADDIE_DOCS_CROSSON_DOTSON", "false") == "true"
    Base.include(Module(:CrossonDotson), cd_script)
end
Literate.markdown(cd_script, gendir; documenter = true, codefence = "```julia" => "```")

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
        "Examples" => [
            "ISOMIP+" => "generated/isomip.md",
            "Crosson–Dotson" => "generated/crosson-dotson.md",
        ],
        "API reference" => [
            "Setup and running" => "API_public.md",
            "Parameterizations and boundaries" => "API_physics.md",
        ],
    ],
    checkdocs = :exports,
    warnonly = true,
)

deploydocs(;
    repo      = "github.com/fesmc/Laddie.jl",
    devbranch = "main",
)
