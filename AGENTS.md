# Laddie.jl — Agent Rules

## Project Overview

Laddie.jl is a Julia package for fast, friendly, flexible, sub-shelf cavity dynamics on CPUs, GPUs and TPUs.
It solves a vertically-averaged approximation of the Navier-Stokes equations in 2D for the ocean cavity along with common parameterizations. Laddie is designed to be modular in the sense that (almost) all components should be exchangeable with alternative implementations. Laddie is also intended to be a fully differentiable.

## Language & Environment

- **Julia 1.10+** | CPU and GPU
- **Key packages**: KernelAbstractions.jl, CUDA.jl, Enzyme.jl via Reactant.jl

## Critical Rules

### Kernel Functions (GPU compatibility)

- Use `@kernel` / `@index` (KernelAbstractions.jl)
- Kernels must be **type-stable** and **allocation-free**
- Use `ifelse` — never short-circuiting `if`/`else` in kernels
- No error messages
- Mark functions called inside kernels with `@inline`
- **Never loop over grid points outside kernels** — use `launch!`

### Type Stability & Memory

- All structs must be concretely typed
- Type annotations are for **dispatch**, not documentation
- Minimize allocation; favor inline computation
- **Never hardcode Float64**: no float literal such as `0.0` or `1.0` in kernels or in code that builds
  arrays. Use `zero(FT)`, `one(FT)`, `FT(0.5)`, `model.FT(x)`, `convert(FT, 1//2)`, or rational literals.
  `FT` may be a ForwardDiff `Dual` or a Reactant traced number, not only Float32/Float64.
  Literal defaults are fine in host-side constructors whose values are converted to `FT` afterwards
  (scheme `@kwdef` defaults, `Params(; FT)`, the reproduction setups)

### Parameterizations

- Each exchangeable scheme is a concrete subtype of an abstract scheme type, stored in a slot of `Params`
  (or `BoundaryConditions`, `CavityForcing`)
- Dispatch on the scheme **at the launch site**: `process!(model, scheme::SomeScheme)` launches the kernel,
  named after the process without an `update_`/`compute_` prefix (`melt!`, `entrainment!`)
- Kernels receive arrays, scalars (or NamedTuples of scalars) and **small scheme structs** for inline
  dispatch of a local operation (`kinks`, `eos`, `tracer_form`, `momentum_form`). Never `Model`, `Params`,
  `Simulation` or another container
- Selecting a whole process (which kernel to launch) happens at the launch site, not inside a kernel
- When a scheme changes the kernel arithmetic, prefer scalar coefficients computed at the launch site,
  and keep the existing default bit-identical
- Numbers read inside kernels, including those of scheme structs, go through `_val`, so that Reactant can
  trace them

### Reactant

- Reactant-specific code lives in `ext/LaddieReactantExt.jl`; the extension needs both Reactant and CUDA
  loaded (also on CPU)
- **No reachable throw path in kernels**: no `@assert`, error, un-elided bounds check, `InexactError` from
  `convert`/`round(Int, …)` or integer division by zero. These lower to traps that fail compilation even
  if never hit. Validate arguments in host-side constructors instead
- `ifelse` evaluates both branches: select the *index*, not the loads (`x[ifelse(c, i, n)]`)
- Functions traced by Reactant outside kernels must be branch-free on values: use `ifelse`, `min`, `max`,
  `clamp` instead of `if`, `? :`, `&&`, `||`
- Closures compiled into kernels capture only `isbits` values (no `Type`s such as `FT`)
- Do not run `test/reactant` with `--check-bounds=yes`
- A plain `Array` inside a value returned by a compiled program is rebuilt by one generated statement per
  element (minutes of compilation on the first call at realistic sizes). Host-only arrays of the model
  (masks, coordinates) are therefore wrapped in the extension's `HostArray` on Reactant; a new host array
  in `Grid` or `Geometry` gets the same treatment through `_host_array`, and any other one must stay out of
  what compiled programs return

### Differentiability

- Supported AD paths: ForwardDiff through a model built at `FT = Dual`, and Enzyme through Reactant
  (`reactant_compile`, `integrate!`, `trace_parameters`). Both must keep working
- No global or hidden mutable state; every input of a computation is passed explicitly. The one exception
  is compile configuration in `ext/LaddieReactantExt.jl` (`TRACING_FUSION`, `EXTRA_COMPILE_OPTIONS`): it
  selects how kernels are lowered, never what they compute
- Guard non-differentiable points (`sqrt`, division) with `_safe_sqrt` / `_safe_div` instead of
  introducing new kinks; a new `abs`/`max`/`min` in the dynamics must respect `Params.kinks`
- A change to the dynamics or to a scheme needs a gradient test (`test/autodiff.jl`, and the scheme
  coverage in `test/reactant/runtests.jl` when it is traced)

### Physics

- Every process cites its governing equation (and its source) in the docstring or in comments
- Heat and salt in the layer are not conserved: entrainment, detrainment and melt exchange them with the
  ambient water and the ice. Test budgets instead: the change of a quantity equals its fluxes (as the
  H equation is tested to close exactly over one step). A process that does conserve a quantity has a test
  that verifies it
- Prefer continuous-time formulations (tendencies, relaxations with a time scale). Instantaneous
  adjustments (resets, clamps, caps) are exceptions: document why they are needed, and do not copy them
  as a pattern for new schemes

### Docstrings

- Use DocStringExtensions.jl with `$(TYPEDSIGNATURES)` when the signature does not include
  default values for args and/or kwargs.
- Use `jldoctest` blocks. Plain `julia` blocks are only exceptionally allowed since they are not tested by CI.
- Include `# output` with verifiable output; prefer `show` methods over boolean comparisons
- Use unicode for math (`Δt`, `η`, `ρ`), not LaTeX — LaTeX doesn't render in the REPL
- Be concise. A docstring says, in this order and nothing more: what it is or does (one or two
  sentences); the governing equation, its source, units and defaults; its fields or arguments, and
  what a function returns or mutates; for an option of a family, when to choose it (one sentence);
  `Select via …` or a `jldoctest`. Aim for ≤ 100 words of prose; only an entry point (`Grid`,
  `Model`, `Simulation`, `run!`) may list more keywords. Comparisons with sibling options, run
  numbers, history and line references into other codes go to the family's docs page; design
  rationale and dead ends to a devnote; implementation notes to a `#` comment above the docstring

## Naming Conventions

- **Files**: snake_case matching the type they define
- **Types/Constructors**: PascalCase **only for true constructors**
- **Functions**: snake_case — `time_step!`; functions that return values are not PascalCase, except named
  setups that return a configured object like a constructor (`ISOMIPForcing`, `ParamsISOMIPv2`, `SimulationPIGv1`)
- **Kernels**: underscore prefix and `_kernel!` suffix — `_accum_series_kernel!`
- **Launchers**: named after the process, without an `update_`/`compute_` prefix — `melt!`, `entrainment!`.
  A launcher that takes `dt` advances something in time and starts with `step_` — `step_entrainment!`,
  `step_secondary_fields!`
- **Scheme types**: a concrete subtype ends with the suffix of its abstract type, the abstract name without
  `Abstract` — `FluxTracerForm <: AbstractTracerForm`, `SinkGapsBC <: AbstractGapsBC`. Prefer renaming the
  abstract type when its suffix is too long to repeat. Exceptions are decided with the human, case by case:
  a dimension tag may follow the suffix (`CoriolisParameter0D`, `OceanForcing1D`, `PerBasinOceanForcing1D`),
  and the internal grid locations `Center`, `Face` (`AbstractLoc`) follow the C-grid convention
- **Variables**: English long name or readable unicode math notation — never mix abbreviated and
  full forms (e.g., `cond` vs `condition`) to imply a difference; be specific

## Module Structure

`src/` has five folders. `Laddie.jl` includes every file explicitly, in dependency order (scheme types,
then containers, then kernels, then the run loop):

- `physics/`: one file per process (`entrainment.jl`, `melting.jl`, `eos.jl`, …), each with one abstract
  scheme type, its implementations, and the `process!` that launches its kernels. `secondary_fields.jl`
  holds `step_secondary_fields!` (the update order) and the friction velocity the schemes share
- `numerics/`: the leapfrog step (`leapfrog.jl`), stencils, advection and tracer/momentum forms, lateral
  viscosity, kinks, time stepping and the CFL number
- `bcs/`: boundary conditions (`walls.jl`, `open_ocean.jl`, `gaps.jl`, the `BoundaryConditions`
  container) and the forcing at the layer's boundaries (`forcing.jl`, `ocean_sources.jl`)
- `simulation/`: the containers of a run (`Grid`, `State`, `Cache`, `Params`, `Model`, `Simulation`),
  the time loop (`run.jl`), the AD entry points (`autodiff.jl`) and ready-made setups (`presets.jl`)
- `utils/`: generic helpers and kernel launches (`utils.jl`, `backend.jl`), diagnostics, I/O, mask
  preprocessing, `show`

Names resolve at call time, so a function can live with its scheme even when its file is included before
`Model`; include order matters only for type annotations in signatures, struct field types and top-level
code. The object structure of a run, from
`fieldtree(build_isomip(); maxdepth = 2)` (snippet below, `model` expanded one level further,
scalar parameters of `Params` abbreviated):

```
Simulation
├── model :: Model
│   ├── grid :: Grid                    # Nx, Ny, dx, dy, mask, z_draft, z_bed, x, y, crop
│   ├── geometry :: Geometry            # derived masks (tmask, umask, vmask, …), wall indicators
│   │                                   # (gl…/lnd… with N/S/E/W = +y/−y/+x/−x), dzdx, dzdy, Coriolis f
│   ├── state :: State                  # prognostics H, U, V, T, S — each a Var{LX,LY} on the
│   │                                   # Arakawa C-grid with leapfrog levels past/present/future
│   ├── cache :: Cache                  # diagnostics and work arrays (melt, Tb, Ta, Sa, drho, ustar,
│   │                                   # entr, detr, convective_instability, adv, lap, Hq, diag, …)
│   ├── params :: Params                # physical constants (g, C_d, A_h, K_h, ρ, L, c_p, …) and
│   │                                   # one slot per exchangeable scheme:
│   │   ├── entrainment                 #   AbstractEntrainment
│   │   ├── melting                     #   AbstractMelting
│   │   ├── convection                  #   AbstractConvection
│   │   ├── frazil_ice                  #   AbstractFrazilIce
│   │   ├── ice_conduction              #   AbstractIceConduction
│   │   ├── equation_of_state           #   AbstractEOS
│   │   ├── max_layer_thickness
│   │   ├── tracer_bounds               #   NoTracerBounds (default) / TracerBounds
│   │   ├── lateral_viscosity
│   │   ├── laplacian_weights
│   │   ├── front_pressure
│   │   ├── momentum_advection
│   │   ├── tracer_form                 #   FluxTracerForm (default) / AdvectiveTracerForm
│   │   ├── momentum_form               #   FluxMomentumForm (default) / AdvectiveMomentumForm
│   │   ├── closure_level
│   │   ├── coriolis                    #   AbstractCoriolisParameter
│   │   ├── ice_slope                   #   MaskedIceSlope (default) / CentredIceSlope
│   │   ├── kinks                       #   AbstractKinks (ExactKinks / SmoothKinks)
│   │   └── integrator                  #   AbstractIntegrator (LeapfrogIntegrator(nu) / FBRK3Integrator)
│   ├── boundary :: BoundaryConditions  # open_ocean, grounding_line_slip, land_slip, gaps, wall_advection
│   └── forcing :: CavityForcing        # ocean (ambient T/S profiles), ice (ice-base temperature)
├── clock :: Clock                      # time, iteration, dt
├── time_stepper                        # FixedTimeStepper / AdaptiveTimeStepper
├── cfl                                 # CFL check
├── stop                                # simulation end criterion
├── output :: OutputConfig              # saveday, diagday, restday, tsday, resultdir, fields
├── io :: IOState                       # output accumulators and bookkeeping
├── debug :: DebugConfig                # check_nans
└── exec                                # execution mode (native KernelAbstractions vs Reactant)
```

To print the current structure of any object (e.g. after adding a field or scheme slot):

```julia
using Laddie
function fieldtree(x, depth = 0; maxdepth = 2)
    for f in fieldnames(typeof(x))
        v = getfield(x, f)
        println("  "^depth, f, " :: ", nameof(typeof(v)))
        depth < maxdepth - 1 && isstructtype(typeof(v)) && !(v isa AbstractArray) &&
            fieldtree(v, depth + 1; maxdepth)
    end
end
fieldtree(build_isomip(); maxdepth = 2)
```

List the implementations of a scheme with `using InteractiveUtils; subtypes(Laddie.AbstractMelting)`.

## Typical Usage

```julia
using Laddie
sim = build_isomip(; nx = 40, ny = 20)   # a ready Simulation on an ISOMIP cavity
run!(sim; days = 0.5, verbose = false)   # clock, output, CFL/dt control, blow-up checks
meltstats(sim)                           # (max_melt, mean_melt, max_speed, total_melt, refreezing)
```

Own geometry: `Simulation(Model(Grid(mask, z_draft, dx, dy); forcing))`. For a GPU, pass the backend
to `Grid` or `build_isomip`, or move an existing model with `to_backend`.

## Time-Step Flow

`run!` loops over `time_step!` (`src/simulation/run.jl`) and handles I/O, adaptive dt and blow-up checks between
batches. `time_step!` sets the forcing time, calls `_step_model!` (`src/numerics/leapfrog.jl`) and ticks the clock.
`integrate!(model, dt, n)` calls `_step_model!` directly, without a `Simulation`, for AD. `_step_model!`,
`_bootstrap!` and `_rebootstrap!` dispatch on `Params.integrator`; under the default `LeapfrogIntegrator`:

```
_step_model!(m, dt)
├── advance_leapfrog!               rotate past ← present ← future, then
│   └── step_secondary_fields!    (src/physics/) from the present/past levels:
│       ambient T/S → density → convection → melt → frazil → entrainment/detrainment
├── leapfrog_step!(m, dt, 2)        centred step over 2Δt, writing the future level:
│   ├── H   step_thickness!, clamp thickness, precompute_integration_terms!
│   ├── U,V step_u_momentum!, step_v_momentum!, clamp_velocities! (speed cap v_cut)
│   ├── T   step_temperature!, apply_tracer_bounds! (none by default)
│   └── S   step_salinity!, gap surface fluxes, apply_tracer_bounds!
└── apply_robert_asselin_filter!    filter present (coefficient ν of the integrator), then refresh density and convection
```

- `FBRK3Integrator` (`src/numerics/fbrk3.jl`) runs three stages of the same core (`_step_fields!`), each
  after `step_secondary_fields!` on the stage level, with the past level fixed at qⁿ; no filter, no bootstrap
- The `Simulation` constructor bootstraps the leapfrog with one forward step (`_bootstrap!`,
  `leapfrog_step!(m, dt, 1)`); so does a dt change under `AdaptiveTimeStepper` (`_rebootstrap!`) and a restart
- `adv`, `lap` and `Hq` in `Cache` are shared work arrays: U, V, T and S are stepped one after another and
  each consumes them before the next one writes them. Do not read them across steps
- A new process usually enters as a `process!` in `step_secondary_fields!` (ordered by its inputs) and
  as a term in the relevant `step_*!` kernel

## Common Pitfalls

1. **Type instability** in kernels ruins GPU performance
2. **Overconstraining types**: use annotations for dispatch, not documentation
3. **Implicit imports**: import every name explicitly (`using X: a, b`), in `src/Laddie.jl` for the
   package and at the top of each extension. `test/explicit_imports.jl` (ExplicitImports.jl) fails on
   implicit, stale or non-owner imports; extend its `ignore` lists only for non-public names that have
   no public alternative
4. **Subtle bugs from missing method imports**, especially in extensions
5. **Expecting unexported names**: consider exporting them rather than changing user scripts
6. **"Type is not callable" errors**: variable name shadows a function — rename or qualify
7. **Quick fixes that break correctness**: if a test fails after a change, revisit the original edit
8.  **Commented-out code**: delete it. Git is the journal — don't leave commented code, debugging
    artifacts, or stale copy-paste remnants
9.  **Hardcoded Float64**: never use `0.0`, `1.0` in kernels, use `zero(FT)`, `FT(x)` or `model.FT(x)`.
10. **Changing dependencies**: never add, remove or change `[deps]`/`[weakdeps]` in `Project.toml` unless
    the task requires it; touch `[compat]` only when asked
11. **Scope creep**: keep a change to a single concern; unrelated cleanup goes in a separate commit or PR
12. **Closures over mutable state**: they break AD — pass the values as explicit arguments

## Design Principles

- **Dispatch over conditionals**: use Julia's type system and multiple dispatch instead of
  `if`/`else` branching. Backend-specific code goes in `ext/` extensions, not `if` branches in `src/`
- **Use `to_backend` to move a model to a device** — never manual `CuArray()` calls. Copying device data
  back to the host for I/O, output or restarts with `Array(x)` is fine
- **Defaults serve the common case**: avoid `nothing` defaults when a concrete default (like `CPU()`)
  covers 80% of usage. Minimize boilerplate for the typical user.
- **Keyword argument names must be consistent** across related types and constructors
- **Always use explicit `return`** in functions longer than one expression
- **One operation per line** as default; break long expressions across lines

## Testing

- Full suite: `julia --project=. -e 'using Pkg; Pkg.test()'`. GPU tests run only when a CUDA device is
  functional
- Single area, from the package root (test-only deps such as Aqua and ForwardDiff live in `test/Project.toml`):
  ```
  julia --project=test -e 'include("test/setup.jl"); include("test/forcing.jl")'
  ```
  (`test/Manifest.toml` points at the local checkout; check `path = ".."` under `[[deps.Laddie]]`)
- Reactant tests are not part of `Pkg.test()` and have their own environment. Run them only after touching
  the Reactant extension, a kernel's arguments, or when asked:
  ```
  julia --project=test/reactant -e 'using Pkg; Pkg.instantiate()'
  julia --project=test/reactant test/reactant/runtests.jl
  ```
- Julia stack traces are long: write test output to a file and `grep` it
- A failing test is a signal about the change, not about the test. Do not loosen a tolerance to make it pass
- **Stale manifests**: `Manifest.toml` and `Manifest-v1.13.toml` are resolved per Julia version. If a
  *stdlib* fails to precompile, compare `grep -m1 julia_version <env>/Manifest*.toml` with
  `julia --version`, and re-resolve that environment
  (`julia --project=<env> -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'`) instead of debugging the code

## Documentation

- Build: `julia --project=docs docs/make.jl`. Build the docs after changing public API, docstrings or
  `docs/src/`: a stale `@docs` entry or a missing docstring fails the build
- `docs/Manifest.toml` must point to the local checkout (`path = ".."` under `[[deps.Laddie]]`); otherwise
  the build documents the registered release and every new docstring is an "undefined binding". Fix with
  `julia --project=docs -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'`
- Cite with DocumenterCitations (`[Key](@cite)`) and add the BibTeX entry to
  `docs/assets/laddie-description.bib`
- Write units in parentheses, (m/s), not brackets, which clash with Markdown links

## Benchmarks

- Benchmarks live in `benchmark/` (CPU/GPU, adaptive dt, ForwardDiff, Reactant). Run them only when asked
  or after a major change to the compute path
- Compare against the most recent stored results (e.g. `benchmark/*/results-*.csv`): deviations within
  ±20% are noise; report larger regressions

## Reproductions are a regression contract

`src/simulation/presets.jl` defines `Params*v1/v2` and `Simulation*v1/v2`, which reproduce published LADDIE
setups (tested in `test/reproductions.jl`). Each `Params*` alias is pinned by its mean melt on a small
ISOMIP cavity (`REPRODUCTION_MEAN_MELT`, runs on CI); the `Simulation*` aliases need `papers/` and are
tested only where it exists.

- Never change what these aliases produce as a side effect of another change
- A new scheme or option keeps the existing default; a reproduction pins any setting it relies on explicitly
- Changing a model-wide default requires checking every reproduction alias and saying so in the plan.
  If an alias changes on purpose, update its `REPRODUCTION_MEAN_MELT` value in the same commit and say why

## Agent Behavior

- Prioritize type stability and GPU compatibility
- Maintain AD-compatibility via Reactant.jl
- Follow established patterns in existing code
- Add tests for new functionality; update exports when adding public API
- Reference physics equations in comments when implementing dynamics

## Implementation plans

Major features, bug fixes with wide reach, and refactors that change substantial existing code need an
**implementation plan** that the human has **explicitly approved** before any code is written. A change of
direction (new bucket, different design) needs approval again.

Plans live in the internal `laddie-roadmap/`, one Markdown file per topic (see below). Skeleton:

```md
# Title

Status YYYY-MM-DD: **planned / approved / in progress / done**. One sentence on where it stands.
Branch, base revision, code, tests, docs: pointers only.

## Originating prompt
## Rationale            physics/problem, design decisions, not in scope
## Tasks                buckets, each small enough to review and test on its own; approval per bucket
## Results              dated findings, numbers copied in (runs, jobs, figures)
## Open questions
```

- Update the status line whenever the state changes
- Record a change of direction as a new dated entry rather than rewriting history
- When done, move the plan to `laddie-roadmap/archived/` and write the devnote

## Roadmaps and devnotes

- `laddie-roadmap/` is **internal and not committed**: implementation plans and working notes, written
  while a change is developed. Do not cite it from code, docs or commits
- `devnotes/` holds the settled findings, summarised from the roadmaps: `improvements/` (changes in main),
  `deadends/` (ideas dropped, with the reason), `open-questions/` (unresolved). Read the relevant devnote
  before reworking an area, and do not retry a dead end without addressing why it failed
- Write devnotes following `devnotes/rules.md`