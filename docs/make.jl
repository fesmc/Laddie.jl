using Laddie
using Documenter

DocMeta.setdocmeta!(Laddie, :DocTestSetup, :(using Laddie); recursive=true)

makedocs(;
    modules=[Laddie],
    authors="JanJereczek <jan.jereczek@gmail.com> and contributors",
    sitename="Laddie.jl",
    format=Documenter.HTML(;
        canonical="https://fesmc.github.io/Laddie.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/fesmc/Laddie.jl",
    devbranch="main",
)
