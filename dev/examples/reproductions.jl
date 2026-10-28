#=
# Reproducing published LADDIE runs

Laddie.jl has aliases for the published runs of the earlier LADDIE versions: the LADDIE v1
runs of [Lambert et al. (2023)](@cite lambert_modelling_2023), the LADDIE v2 runs of
[Lambert et al. (2026)](@cite lambert_one-layer_2026), and the ice-shelf gap runs of
[Jesse et al. (2026)](@cite jesse_ice-sheet_2026). Each run has two:

  * `Params<Setup>v1|v2` returns its [`Params`](@ref); any keyword overrides a field, as in
    `ParamsFRISv1(; H_min = 3.0)`;
  * `Simulation<Setup>v1|v2` builds the whole run — grid, ocean profile, ice temperature,
    boundary conditions, time step and duration — on the geometry the run left behind, ready
    for [`run!`](@ref). Any [`Simulation`](@ref) keyword overrides the setup's.

The values follow the code that produced each paper's results, i.e. the run's own
configuration file. Where the paper says otherwise, the alias's docstring says so (see
[Published setups](@ref)). LADDIE v2 runs on an unstructured mesh with forward–backward
Runge–Kutta time stepping, and Laddie.jl on a C-grid with leapfrog, so the v2 aliases share
v2's parameters but not its numerics.

The simulations read their geometry from `papers/` in the package root, which holds the
papers' bundled code and output and is not part of the repository (pass `datadir` to point
elsewhere).

## Results

Each simulation was run for its full duration and compared with the run it reproduces, on
the shelf cells the two share: mean melt rate, the correlation of the melt patterns, and
mean layer thickness ``H``. The v1 references are the outputs bundled with
[Lambert et al. (2023)](@cite lambert_modelling_2023) and the v1 run of
[Lambert et al. (2026)](@cite lambert_one-layer_2026); the v2 references are the last snapshot
of v2's own output, remapped from its mesh to a regular grid.

| Simulation | Reference | Mean melt (m/yr), Laddie.jl / reference | Pattern correlation | Mean ``H`` (m), Laddie.jl / reference |
|:-----------|:----------|:----------------------------------------|--------------------:|:--------------------------------------|
| [`SimulationCrossonDotsonv1`](@ref) | [Lambert et al. (2023)](@cite lambert_modelling_2023), 500 m, 50 days | 9.82 / 9.82 (−0.02 %) | 0.9996 | 55.7 / 55.8 |
| [`SimulationPIGv1`](@ref) | [Lambert et al. (2023)](@cite lambert_modelling_2023), App. C, 500 m, 30 days | 15.30 / 15.27 (+0.2 %) | 0.991 | 41.5 / 41.6 |
| [`SimulationASEv1`](@ref) | v1 run of [Lambert et al. (2026)](@cite lambert_one-layer_2026), Sect. 3.1, 20 days | 17.49 / 17.51 (−0.1 %) | 0.999 | 26.5 / 26.5 |
| [`SimulationASEv1Theoretical`](@ref) | same | 19.02 / 17.51 (+8.6 %) | 0.85 | 32.6 / 26.5 |
| [`SimulationASEv2`](@ref) | [Lambert et al. (2026)](@cite lambert_one-layer_2026), Sect. 3.1, 20 days | 20.46 / 19.44 (+5.2 %) | 0.906 | 40.5 / 31.7 |
| [`SimulationISOMIPv2`](@ref) | [Lambert et al. (2026)](@cite lambert_one-layer_2026), Sect. 3.2.1, 20 days | 10.54 / 8.92 (+18 %) | 0.932 | 26.0 / 31.5 |
| [`SimulationFRISv1`](@ref) | [Lambert et al. (2023)](@cite lambert_modelling_2023), 1 km, 720 days | not yet run | | |
| [`SimulationMISOMIPv2`](@ref), [`SimulationMISOMIPGapsv2`](@ref) | standalone spin-ups of coupled runs | no reference | | |

The v1 runs are reproduced to a few tenths of a percent, the v2 runs to 5–20 %:

  * **`SimulationASEv1` vs `SimulationASEv1Theoretical`.** The v1 run of
    [Lambert et al. (2026)](@cite lambert_one-layer_2026) reads a 500 m geometry and coarsens it to 1 km, but the v1 code takes the grid
    spacing from the input before coarsening and never updates it. That run was therefore on
    a 1 km grid with Δx = 500 m in every derivative. `SimulationASEv1` repeats this and
    reproduces the v1 output; `SimulationASEv1Theoretical` is the consistent 1 km run, and
    melts 9 % more.
  * **v2.** The geometry is v2's own, remapped from its mesh to a regular grid, which smears
    the floating mask along the grounding line and the domain edge: on ISOMIP+ Laddie.jl
    has 3814 shelf cells where v2 had a layer in 3406. This, and the different
    discretisation, is what remains between the two.
  * **Maximum melt.** Mean melt and the pattern agree, but single cells can differ: PIG's
    maximum is 375 m/yr against v1's 152, at the edge of the mask, which the PIG output
    does not store and the alias rebuilds by flotation.

There are no simulations for the pan-Antarctic and 120 m Pine Island runs of
[Lambert et al. (2026)](@cite lambert_one-layer_2026), whose inputs (BedMachine v3 with ISMIP6 forcing, REMA with Bedmap v3) are not
bundled; [`ParamsAISv2`](@ref) and [`ParamsPIGv2`](@ref) give their parameters.

## Running one

The code below is the Crosson–Dotson comparison; the others differ only in the alias and the
reference file. It is **not executed** by the documentation build, as it needs `papers/`
and about ten minutes. Running
`LADDIE_DOCS_REPRODUCTIONS=true julia -t 8 --project=docs docs/make.jl` runs every comparison
in the table (`docs/src/examples/reproductions_all.jl`), which takes a few hours, most of
them FRIS.
=#

using Laddie
using NCDatasets
using Printf
using Statistics

sim = SimulationCrossonDotsonv1()
run!(sim)                                   # 50 days at dt = 120 s

# The reference stores the run's computational domain without the one-cell border ring
# that LADDIE v1 adds, and its melt rate in m/yr.

m = sim.model
inner(A) = Array(A)[2:end-1, 2:end-1]
melt = inner(m.melt) .* m.seconds_per_year
H = inner(m.H.present)
active = inner(m.tmask) .> 0

reference = joinpath(pkgdir(Laddie), "papers", "lambert-2023", "data",
                     "CrossDots_0.5_tanh_Tdeep0.4_ztcl-500_050.nc")
ref_melt, ref_H = NCDataset(reference) do ds
    coalesce.(Array(ds["melt"]), NaN), coalesce.(Array(ds["D"]), NaN)
end
shelf = active .& isfinite.(ref_melt)

@printf("mean melt %.3f m/yr (reference %.3f, %+.2f %%)\n", mean(melt[shelf]),
        mean(ref_melt[shelf]), 100 * (mean(melt[shelf]) / mean(ref_melt[shelf]) - 1))
@printf("pattern correlation %.4f\n", cor(melt[shelf], ref_melt[shelf]))
@printf("mean H %.1f m (reference %.1f)\n", mean(H[shelf]), mean(ref_H[shelf]))
