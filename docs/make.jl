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

# The Amundsen Sea example needs BedMachine v3 and the Lambert et al. (2026) output, so it
# is static too. Set LADDIE_DOCS_ASE=true to run it (with `julia -t 8`, as the page says).
ase_script = joinpath(exdir, "lambert-ase.jl")
if get(ENV, "LADDIE_DOCS_ASE", "false") == "true"
    Base.include(Module(:LambertASE), ase_script)
end
Literate.markdown(ase_script, gendir; documenter = true, codefence = "```julia" => "```")

# The ice-shelf gaps example needs the Jesse et al. (2026) model output (7.6 GB) and ~7 min,
# so it is static in the same way. Set LADDIE_DOCS_JESSE=true to run it.
jesse_script = joinpath(exdir, "jesse-gaps.jl")
if get(ENV, "LADDIE_DOCS_JESSE", "false") == "true"
    # A bare Module has no `include` of its own, and the script includes its regridding
    # helper, so give the sandbox module one that resolves relative to the included file.
    jesse_mod = Module(:JesseGaps)
    Core.eval(jesse_mod, :(include(path) = Base.include(@__MODULE__, path)))
    Base.include(jesse_mod, jesse_script)
end
Literate.markdown(jesse_script, gendir; documenter = true, codefence = "```julia" => "```")

# The inverse-problems example needs a CUDA GPU for its Reactant half (~6 min), so it is
# static too. Set LADDIE_DOCS_INVERSE=true to run it.
inv_script = joinpath(exdir, "inverse-problems.jl")
if get(ENV, "LADDIE_DOCS_INVERSE", "false") == "true"
    Base.include(Module(:InverseProblems), inv_script)
end
Literate.markdown(inv_script, gendir; documenter = true, codefence = "```julia" => "```")

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
        "Reactant backend" => "reactant.md",
        # "Implementation" => "implementation.md",
        "Examples" => [
            "ISOMIP+" => "generated/isomip.md",
            "Crosson–Dotson" => "generated/crosson-dotson.md",
            "Amundsen Sea" => "generated/lambert-ase.md",
            "Ice-shelf gaps" => "generated/jesse-gaps.md",
            "Inverse problems" => "generated/inverse-problems.md",
        ],
        "API reference" => [
            "Setup and running" => "API_public.md",
            "Parameterizations and boundaries" => "API_physics.md",
            "Automatic differentiation" => "API_ad.md",
        ],
    ],
    checkdocs = :exports,
    warnonly = true,
)

deploydocs(;
    repo      = "github.com/fesmc/Laddie.jl",
    devbranch = "main",
)
