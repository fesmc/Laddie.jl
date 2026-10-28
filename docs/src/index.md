# Laddie.jl

Laddie.jl is a Julia implementation of LADDIE ([Lambert et al. (2023)](@cite lambert_modelling_2023);
[Lambert et al. (2026)](@cite lambert_one-layer_2026)), a model of the **basal melt rate beneath an ice
shelf**. It treats the thin, buoyant meltwater layer in the cavity as a depth-averaged circulation,
and runs on CPUs and GPUs through [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl).
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) compiles the whole time loop with XLA to
accelerate GPU runs, and the model is differentiable: a run can be differentiated with ForwardDiff.jl
on the CPU and CUDA, and with Reactant and Enzyme for gradients at a cost that does not grow with the
number of parameters.

The physics is that of LADDIE v2 on the grid of LADDIE v1; see [LADDIE versions](versions.md) for
how the two differ and which choice each option makes.

## The model at a glance

Geometry and forcing go in, the one-layer plume model integrates them, and fields and restarts come out.

```@raw html
<div class="columns is-vcentered is-multiline">
  <div class="column">
    <div class="box">
      <p class="is-size-7 is-uppercase has-text-link has-text-weight-semibold">Inputs</p>
      <p><code>Grid</code>: domain mask, ice draft, bed and cell spacing.</p>
      <p><code>CavityForcing</code>: ambient ocean profiles and the ice temperature above the base.</p>
      <p><code>Params</code>: physics and numerical choices.</p>
      <p><code>BoundaryConditions</code>: wall, gap and open-ocean conditions.</p>
    </div>
  </div>
  <div class="column is-narrow has-text-centered has-text-link">
    <span class="icon is-hidden-mobile"><i class="fa-solid fa-arrow-right"></i></span>
    <span class="icon is-hidden-tablet"><i class="fa-solid fa-arrow-down"></i></span>
  </div>
  <div class="column">
    <div class="notification is-link is-light">
      <p class="is-size-7 is-uppercase has-text-link has-text-weight-semibold">Model</p>
      <p><strong>One layer under the ice.</strong> A single layer of thickness <code>H</code> carries depth-averaged velocities <code>U</code>, <code>V</code> and tracers <code>T</code>, <code>S</code>.</p>
      <div class="tags">
        <span class="tag is-light has-text-success-dark is-family-monospace">H</span>
        <span class="tag is-light has-text-success-dark is-family-monospace">U</span>
        <span class="tag is-light has-text-success-dark is-family-monospace">V</span>
        <span class="tag is-light has-text-success-dark is-family-monospace">T</span>
        <span class="tag is-light has-text-success-dark is-family-monospace">S</span>
      </div>
      <p>Melt, entrainment and density are set from the state at each step. The melt rate at the ice base is the main output.</p>
    </div>
  </div>
  <div class="column is-narrow has-text-centered has-text-link">
    <span class="icon is-hidden-mobile"><i class="fa-solid fa-arrow-right"></i></span>
    <span class="icon is-hidden-tablet"><i class="fa-solid fa-arrow-down"></i></span>
  </div>
  <div class="column">
    <div class="box">
      <p class="is-size-7 is-uppercase has-text-link has-text-weight-semibold">Outputs</p>
      <p><strong>Basal melt rate</strong>, and the plume state, as time-averaged NetCDF.</p>
      <p><strong>Restarts</strong> as JLD2 files, continuing the same clock.</p>
      <p><strong>Provenance</strong>: <code>run_metadata.toml</code> with the parameters, forcing, grid and versions.</p>
    </div>
  </div>
</div>
```

Each time step runs these stages in order. The later ones use the earlier ones' results.

```@raw html
<div class="content">
  <div class="media">
    <figure class="media-left"><span class="tag is-link is-light">01</span></figure>
    <div class="media-content"><p>Rotate the leapfrog levels and update the ambient fields.</p></div>
  </div>
  <div class="media">
    <figure class="media-left"><span class="tag is-link is-light">02</span></figure>
    <div class="media-content"><p>Compute density and convection, then the basal melt and frazil ice.</p></div>
  </div>
  <div class="media">
    <figure class="media-left"><span class="tag is-link is-light">03</span></figure>
    <div class="media-content"><p>Compute entrainment from the buoyancy flux and friction velocity.</p></div>
  </div>
  <div class="media">
    <figure class="media-left"><span class="tag is-link is-light">04</span></figure>
    <div class="media-content"><p>Step the thickness, then the velocities, then limit the speed.</p></div>
  </div>
  <div class="media">
    <figure class="media-left"><span class="tag is-link is-light">05</span></figure>
    <div class="media-content"><p>Step the temperature and salinity, then clamp them to their bounds.</p></div>
  </div>
  <div class="media">
    <figure class="media-left"><span class="tag is-link is-light">06</span></figure>
    <div class="media-content"><p>Filter the leapfrog levels and advance the clock.</p></div>
  </div>
</div>
```

The [physics](physics/index.md) and [numerics](numerics/index.md) overviews list each
parameterisation and numerical choice, with its default. [Time stepping](numerics/time_stepping.md)
has the leapfrog and filter details.

## Examples

Each example is a page that runs a case end to end and compares it with a reference:

- [ISOMIP+](generated/isomip.md): the idealised ISOMIP+ cavity, warm and cold, with a day-long
  comparison against the LADDIE v1 code.
- [Crosson–Dotson](generated/crosson-dotson.md): reproduction of the realistic cavity run of
  [Lambert et al. (2023)](@cite lambert_modelling_2023).
- [Amundsen Sea](generated/lambert-ase.md): the Amundsen Sea Embayment, against LADDIE v2 output
  from [Lambert et al. (2026)](@cite lambert_one-layer_2026).
- [Ice-shelf gaps](generated/jesse-gaps.md): the two defensible treatments of an ice-free gap, on
  the geometries of [Jesse et al. (2026)](@cite jesse_ice-sheet_2026).
- [Inverse problems](generated/inverse-problems.md): differentiating a run with respect to its
  parameters, with ForwardDiff.jl and Reactant.jl.
- [Reproducing published runs](generated/reproductions.md): every published LADDIE v1 and v2 setup
  that Laddie.jl has a preset for, compared with its published output.

## Quick start

CPU example:
```julia
using Laddie

# Build and run the idealised ISOMIP+ warm cavity (CPU)
sim = build_isomip(; isomipcond = :warm)
run!(sim; days = 5.0)
mx, mn, sp = meltstats(sim)
```

GPU (CUDA example):
```julia
using CUDA, Laddie
sim = build_isomip(CUDABackend(); isomipcond = :warm)
run!(sim; days = 30.0)
```

## Performance

Laddie.jl allocates its arrays on a chosen KernelAbstractions backend (CPU, CUDA, ROCm or Metal),
so the entire time step executes on-device. Every term is a fused kernel, one pass per term with no
intermediate arrays, so a time step allocates almost nothing. The same code runs multi-threaded on
the CPU (`julia -t N`).

## Documentation map

| page | contents |
|------|----------|
| [LADDIE versions](versions.md) | v1 and v2 side by side, and where Laddie.jl sits between them |
| [Physics](physics/index.md) | what the model represents, the governing balances, and one page per parameterisation |
| [Numerics](numerics/index.md) | grid, time stepping, boundaries, stability, and one page per numerical choice |
| [Examples](generated/isomip.md) | the cases above, with their figures and reference comparisons |
| [API reference](API_public.md) | setup, running, published setups and automatic differentiation |
| [References](references.md) | every paper cited |
