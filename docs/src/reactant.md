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
| `:xla` | raised, fused by XLA's heuristics | GPU, CPU | **yes** |

`:auto`, the default, picks `:native` on the GPU and `:kernel` on the CPU.

Without barriers, XLA fuses a whole step into a few giant fusions. These recompute
intermediate fields for every output cell and use up to 255 registers at 17 %
occupancy. A barrier makes XLA write a kernel's arrays to memory, so every kernel
becomes about one fusion. A barrier before each stencil kernel instead, so that the fields it
reads at neighbour offsets are in memory, was slower than `:kernel` on every grid (2.2×
at 2000×2000; `benchmark/reactant/results-2026-09-24.csv`) and was removed. Enzyme has no derivative rule for barriers, and none for
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
it).

### Scalar parameters

A compiled program takes the scalar parameters of the model it was compiled with as
constants, and ignores the parameters of the model it is later called with.
[`trace_parameters`](@ref) returns the model with every float of [`Params`](@ref)
(including those of the parameterisation objects) as a Reactant number, so that they
become inputs of the program. The program then runs for other parameter values
without recompiling, and Enzyme differentiates with respect to a parameter through a
tangent model seeded with `trace_parameters`:

```julia
fresh(; kw...) = trace_parameters(to_backend(build_isomip(CPU()), ReactantBackend()).model; kw...)
model = fresh()
direction() = trace_parameters(Enzyme.make_zero(model); C_d = 1)   # direction: C_d
prog = reactant_compile(fwd, model, direction(), dt, n)
dloss = only(prog(fresh(), direction(), dt, n))

# Another drag coefficient, same program:
dloss2 = only(prog(fresh(; C_d = 3e-3), direction(), dt, n))
```

A program advances the model **and the tangent** it is called with in place: call it
with a fresh pair each time. A tangent reused after a call no longer starts from zero
state perturbations, and gives a wrong derivative without any error.

The derivatives with respect to `C_d` and `L` match central differences (Float64).
The traced parameters reach the kernels as device references, which each kernel loads
once on entry (`_val`). In Float32 on the GPU (ms/step, 2000×2000):

| strategy  | constant parameters | traced parameters |
|-----------|---------------------|-------------------|
| `:native` | 12.7                | 12.4              |
| `:xla`    | 25.2                | 30.0              |

Native kernels cost the same. Raised kernels are 19 % slower, since XLA can no longer
fold the parameters into its fusions; this is the price of differentiating with
respect to them. `run!` keeps parameters as constants; do not run
a simulation on a model from `trace_parameters`. Parameters that act only when the
model is built (`coriolis`, `D_init`, `dT_init`, `dS_init`) have no effect on a
program.

### Reverse mode

Reverse mode gives the gradient with respect to every input at once: all traced
parameters, the forcing profiles, the initial state. The shadow model returned by
`Enzyme.autodiff` holds it:

```julia
rev(m, dm, dt, n) = (Enzyme.autodiff(Enzyme.Reverse, loss, Enzyme.Active,
                                     Enzyme.Duplicated(m, dm), Enzyme.Const(dt),
                                     Enzyme.Const(n)); dm)
prog = reactant_compile(rev, model, Enzyme.make_zero(model), dt, n)
grad = prog(fresh(), Enzyme.make_zero(model), dt, n)
grad.params.C_d, grad.params.melting.gamTfix, grad.forcing.ocean.Tz
```

The reverse pass needs the state of every step, in reverse order. `integrate!` keeps
at most `checkpoints` states (default 20) and recomputes the steps between them
(binomial checkpointing, revolve). Measured on the A4000 in Float64, against the
primal run:

| | reverse / primal | peak memory |
|---|---|---|
| 40×20, 2000 steps, 5 / 10 / 20 / 50 checkpoints | 11.5× / 8.5× / 7.4× / 6.8× | 7.5 / 8.1 / 10.8 / 16.9 MB |
| 40×20, 2000 steps, no checkpointing (static `n`) | 7.4× | 4.6 GB |
| 480×240, 200 steps, 20 checkpoints | 7.1× | 0.9 GB |

- **Memory does not grow with `n`.** It is about (360 + 28 × `checkpoints`) × 8 bytes
  per grid cell in Float64: one step's intermediate values plus the checkpointed
  states. About 7 GB at 1000×1000 with the default, so grid size, not run length, is
  the limit.
- **Checkpoints cost nothing** in the primal and in forward mode.
- A traced loop whose trip count is only known at run time cannot be differentiated in
  reverse mode without checkpointing (XLA cannot compile Enzyme's buffer of dynamic
  size).

The extension's tests check the reverse gradient against forward mode (along a
parameter and along a random profile, to 1e-10) and check that every entry of the
gradient is finite, for each scheme.

### Limitations

- **Raised kernels without barriers** (`:xla`) are the slowest strategy, 2–2.5× KA.
  A derivative in forward mode costs roughly one such run per direction; a gradient in
  reverse mode about 7 such runs, whatever the number of inputs.
- **Kinks.** The speed cap, the tracer clamps, the `D_min` floor and the thickness caps
  have kinks; at one the derivative is that of the active side (a subgradient).
- The ForwardDiff extension remains the option for gradients on the CPU backends and
  on the KA GPU backends (build the model at `FT = Dual`).

## Tests

The extension's tests are not part of `Pkg.test()`, since Reactant is a large dependency
and each compile takes a minute or two:

```
julia --project=test/reactant -e 'using Pkg; Pkg.instantiate()'
julia --project=test/reactant test/reactant/runtests.jl
```
