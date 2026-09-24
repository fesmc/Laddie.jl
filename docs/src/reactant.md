# Reactant backend

[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) compiles Julia code with XLA. With
`using Reactant, CUDA`, a simulation moved to a [`ReactantBackend`](@ref) runs its time
steps as compiled batches. It can also be differentiated with Enzyme, through
[`integrate!`](@ref) and [`reactant_compile`](@ref).

## Running a simulation

```julia
using Laddie, Reactant, CUDA
Reactant.set_default_backend("gpu")               # or "cpu"

sim = build_isomip(CPU(); FT = Float32)           # build and bootstrap on the CPU
rsim = to_backend(sim, ReactantBackend())         # move to Reactant
run!(rsim; days = 30)
```

- **CUDA.jl is needed even on the CPU target.** Reactant compiles the kernels through it.
- **`run!` works as on the other backends.** It steps in compiled batches up to the next
  host event: the check cadence, an output or log time, a steady-state sample or the
  end of the run. Output, log diagnostics, restarts and `AdaptiveDt` behave as usual.
  A change of `dt` does not trigger a recompile, because `dt` is an input of the
  compiled programs.
- **The first `run!` compiles.** It builds four programs: the step batch, the
  re-bootstrap after a `dt` change, the sync-point diagnostics, and the step batch with
  output accumulation when output is on. This takes 1.5–6 minutes, longer on larger
  grids. Later `run!` calls reuse them.
- **Accuracy.** With the default native kernels, Float64 results are bit-identical to
  the KernelAbstractions backends. The raised strategies and XLA's reductions (CFL,
  diagnostics) agree to round-off after a day: about 1e-13 relative in Float64 and
  1e-5 to 1e-4 in Float32.
- **`DebugConfig(check_nans = true)` is ignored**, because a data-dependent error cannot
  be traced. The blow-up check at the sync points still runs.

## Fusion strategies

XLA can take the kernels in two forms: as native CUDA code, or *raised* to XLA
operations, which XLA then fuses. `ReactantBackend(; fusion)` picks the form:

| `fusion` | Kernels | Where | Differentiable |
|---|---|---|---|
| `:native` | CUDA code from CUDA.jl, called from the compiled loop | GPU | no |
| `:kernel` | raised, with an optimisation barrier after every kernel | GPU, CPU | no |
| `:stencil` | raised, with a barrier before each stencil kernel | GPU, CPU | no |
| `:xla` | raised, fused by XLA's heuristics | GPU, CPU | **yes** |

`:auto`, the default, picks `:native` on the GPU and `:kernel` on the CPU.

Without barriers, XLA fuses a whole step into a few giant fusions. These recompute
intermediate fields for every output cell and use up to 255 registers at 17 %
occupancy. A barrier makes XLA write a kernel's arrays to memory, so every kernel
becomes about one fusion. Enzyme has no derivative rule for barriers, and none for
native kernel calls. That leaves `:xla` as the one strategy that can be
differentiated.

With native kernels, the traced loop takes 4 steps per iteration. Otherwise the GPU sits
idle between iterations, most likely because XLA evaluates the loop condition between
them. Unrolling 4 steps is 10 % faster than 1; 10 steps gain nothing more.

### Measured cost

ISOMIP+ warm, Float32, RTX A4000, ms per step (`benchmark/reactant/fusion.jl`; raw data
in `benchmark/reactant/results-2026-09-23.csv`):

| Grid | KA CUDA | `:native` | `:kernel` | `:xla` |
|---|---|---|---|---|
| 640×320 | 0.60 | 0.67 (1.12×) | 0.83 | 1.22 |
| 1280×640 | 2.18 | 2.41 (1.11×) | 3.78 | 4.89 |
| 2000×2000 | 10.1 | 11.1 (1.09×) | 20.6 | 24.8 |

- **`:native` runs the same kernels as KA.** The GPU kernel time per step is equal:
  0.585 ms at 640×320, with 38 launches against KA's 36. The remaining ~10 % is idle
  time between the loop iterations.
- **The raised strategies are 1.4–2× slower than KA even with a barrier per kernel.**
  This is XLA's code generation for these stencils, not the fusion choice. In a profile
  at 640×320, the largest raised stencil fusions did the same work as their KA kernels
  and took up to 3× as long. XLA's `fast_min_max`, `ftz`, unroll-tuning and
  fusion-autotuner options changed nothing measurable.

## Automatic differentiation

[`integrate!`](@ref) is the model as a plain function: `n` leapfrog steps with no
`Simulation` around them. Inside a program compiled by [`reactant_compile`](@ref), with
a traced `n`, the steps become a traced loop that Enzyme can differentiate. Forward
mode, for the derivative of the mean melt rate after 100 steps with respect to a
uniform warming of the ambient temperature profile:

```julia
using Laddie, Reactant, CUDA
using Reactant: Enzyme

sim = to_backend(build_isomip(CPU()), ReactantBackend())
loss(model, dt, n) =
    (integrate!(model, dt, n); sum(model.melt .* model.imask) / sum(model.imask))

dmodel = Enzyme.make_zero(sim.model)
dmodel.forcing.ocean.Tz .= 1                    # the direction of the derivative
fwd(m, dm, dt, n) = Enzyme.autodiff(Enzyme.Forward, loss, Enzyme.Duplicated(m, dm),
                                    Enzyme.Const(dt), Enzyme.Const(n))
dt, n = ConcreteRNumber(sim.clock.dt), ConcreteRNumber(100)
dloss = only(reactant_compile(fwd, sim.model, dmodel, dt, n)(sim.model, dmodel, dt, n))
```

This matches central differences to about 1e-8 relative (the extension's tests check
it). The limitations:

- **Only fields of the model can be inputs.** That covers the state, the forcing
  profiles `Tz` and `Sz`, and the ice temperature. Scalar parameters (`C_d`, `γ_T`, …)
  are compiled in as constants. To trace them, the kernels must stop taking their
  float type from a scalar argument (`FT = typeof(g)` in 22 kernels); taking it from an
  array's `eltype` would do.
- **Raised kernels without barriers** (`:xla`) are the slowest strategy, 2–2.5× KA.
  A derivative in forward mode costs roughly one such run per direction.
- For gradients with respect to a few scalar parameters, the ForwardDiff extension
  already works on every backend (build the model at `FT = Dual`).

## Tests

The extension's tests are not part of `Pkg.test()`, since Reactant is a large dependency
and each compile takes a minute or two:

```
julia --project=test/reactant -e 'using Pkg; Pkg.instantiate()'
julia --project=test/reactant test/reactant/runtests.jl
```
