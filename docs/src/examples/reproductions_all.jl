# Every comparison in the table of reproductions.jl: runs each simulation alias for its
# full duration and prints the numbers against the run it reproduces.  Not part of the
# documentation pages; docs/make.jl runs it when LADDIE_DOCS_REPRODUCTIONS=true.  Needs
# `papers/` in the package root and a few hours (FRIS alone is 720 model days).

using Laddie
using NCDatasets
using Printf
using Statistics

const PAPERS = joinpath(pkgdir(Laddie), "papers")

# v1 references store the computational domain without v1's border ring, melt in m/yr
# and the layer thickness as `D`.
function v1_reference(path)
    NCDataset(path) do ds
        coalesce.(Array(ds["melt"]), NaN), coalesce.(Array(ds["D"]), NaN)
    end
end

# v2 references are the last snapshot of v2's output remapped to a regular grid, melt in
# m/s; v2 has a layer where `H_lad > 0`.
function v2_reference(path, seconds_per_year)
    NCDataset(path) do ds
        H = coalesce.(Array(ds["H_lad"][:, :, end]), 0.0)
        melt = coalesce.(Array(ds["melt"][:, :, end]), NaN) .* seconds_per_year
        ifelse.(H .> 0, melt, NaN), ifelse.(H .> 0, H, NaN)
    end
end

# A field of the model on the grid's input arrays, without the border ring; NaN outside
# the active cells.
function on_input(m, A)
    out = fill(NaN, m.grid.input_size)
    r, c = m.grid.crop
    out[r, c] .= ifelse.(Array(m.tmask) .> 0, Array(A), NaN)
    return out[2:end-1, 2:end-1]
end

const CASES = [
    ("SimulationCrossonDotsonv1", SimulationCrossonDotsonv1, :v1,
     "lambert-2023/data/CrossDots_0.5_tanh_Tdeep0.4_ztcl-500_050.nc"),
    ("SimulationPIGv1", SimulationPIGv1, :v1, "lambert-2023/data/PIG.nc"),
    ("SimulationASEv1", SimulationASEv1, :v1, "lambert-2026/results/laddie1/ASE/output_000020.nc"),
    ("SimulationASEv1Theoretical", SimulationASEv1Theoretical, :v1,
     "lambert-2026/results/laddie1/ASE/output_000020.nc"),
    ("SimulationASEv2", SimulationASEv2, :v2, "lambert-2026/results/ASE/laddie_output_grid.nc"),
    ("SimulationISOMIPv2", SimulationISOMIPv2, :v2,
     "lambert-2026/results/isomip_ocean1/laddie_output_grid.nc"),
    ("SimulationFRISv1", SimulationFRISv1, :v1,
     "lambert-2023/data/FRIS_1.0_linear_S134.8_T1-2.3_720.nc"),
]

for (name, build, version, file) in CASES
    sim = build()
    t = @elapsed run!(sim; verbose = false)
    m = sim.model
    melt = on_input(m, m.melt .* m.seconds_per_year)
    H = on_input(m, m.H.present)
    path = joinpath(PAPERS, file)
    ref_melt, ref_H = version === :v1 ? v1_reference(path) : v2_reference(path, m.seconds_per_year)
    shelf = isfinite.(melt) .& isfinite.(ref_melt)
    a, b = mean(melt[shelf]), mean(ref_melt[shelf])
    @printf("| %s | %.2f / %.2f (%+.1f %%) | %.3f | %.1f / %.1f |  (%d cells, %.0f min)\n",
            name, a, b, 100 * (a / b - 1), cor(melt[shelf], ref_melt[shelf]),
            mean(H[shelf]), mean(ref_H[shelf]), count(shelf), t / 60)
end
