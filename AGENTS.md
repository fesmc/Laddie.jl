# Laddie.jl — Agent Rules

## Project Overview

Laddie.jl is a Julia package for fast, friendly, flexible, sub-shelf cavity dynamics on CPUs, GPUs and TPUs.
It solves a vertically-averaged approximation of the Navier-Stokes equations in 2D for the ocean cavity along with common parameterizations. Laddie is designed to be modular in the sense that (almost) all components should be exchangeable with alternative implementations. Laddie is also intended to be a fully differentiable.

## Language & Environment

- **Julia 1.10+** | CPU and GPU (CUDA, AMD, Metal, OneAPI)
- **Key packages**: KernelAbstractions.jl, CUDA.jl, Enzyme.jl, Reactant.jl

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
- **Never hardcode Float64**: no literal `0.0` or `1.0` in kernels or constructors.
  Use `zero(grid)`, `one(grid)`, `convert(FT, 1//2)`, or rational literals

### Docstrings

- Use DocStringExtensions.jl with `$(TYPEDSIGNATURES)` when the signature does not include
  default values for args and/or kwargs.
- Use `jldoctest` blocks. Plain `julia` blocks are only exceptionally allowed since they are not tested by CI.
- Include `# output` with verifiable output; prefer `show` methods over boolean comparisons
- Use unicode for math (`Δt`, `η`, `ρ`), not LaTeX — LaTeX doesn't render in the REPL

## Naming Conventions

- **Files**: snake_case matching the type they define
- **Types/Constructors**: PascalCase **only for true constructors**
- **Functions**: snake_case — `time_step!`; functions that return values are never PascalCase
- **Kernels**: may prefix with underscore
- **Variables**: English long name or readable unicode math notation — never mix abbreviated and
  full forms (e.g., `cond` vs `condition`) to imply a difference; be specific

## Module Structure

```

```

## Common Pitfalls

1. **Type instability** in kernels ruins GPU performance
2. **Overconstraining types**: use annotations for dispatch, not documentation
3. **Missing imports**: tests will catch this — add to `using` statements
4. **Subtle bugs from missing method imports**, especially in extensions
5. **Expecting unexported names**: consider exporting them rather than changing user scripts
6. **Extending `getproperty` to fix undefined property bugs**: fix on the caller side instead
7. **"Type is not callable" errors**: variable name shadows a function — rename or qualify
8. **Quick fixes that break correctness**: if a test fails after a change, revisit the original edit
9.  **Commented-out code**: delete it. Git is the journal — don't leave commented code, debugging
    artifacts, or stale copy-paste remnants
10. **Hardcoded Float64**: never use `0.0`, `1.0` in kernels or constructors; use `zero(grid)` etc.

## Design Principles

- **Dispatch over conditionals**: use Julia's type system and multiple dispatch instead of
  `if`/`else` branching. Backend-specific code goes in `ext/` extensions, not `if` branches in `src/`
- **Use `on_architecture` for data transfers** — never manual `Array()` / `CuArray()` calls
- **Defaults serve the common case**: avoid `nothing` defaults when a concrete default (like `CPU()`)
  covers 80% of usage. Minimize boilerplate for the typical user.
- **Keyword argument names must be consistent** across related types and constructors
- **Always use explicit `return`** in functions longer than one expression
- **One operation per line** as default; break long expressions across lines

## Agent Behavior

- Prioritize type stability and GPU compatibility
- Follow established patterns in existing code
- Add tests for new functionality; update exports when adding public API
- Reference physics equations in comments when implementing dynamics

## Further Reading

Detailed reference docs are in `.agents/` — read on demand:

| Document | Content |
|----------|---------|
| `.agents/testing.md` | Full testing guidelines, running tests, debugging |
| `.agents/documentation.md` | Building docs, fast builds, doctest details, writing examples |
| `.agents/validation.md` | Reproducing paper results step-by-step |

### Auto-loading Rules

Rules in `.claude/rules/` load automatically when you touch matching files:
- `kernel-rules.md` — GPU kernel requirements (src/)
- `docstring-rules.md` — docstring and jldoctest conventions (src/)
- `testing-rules.md` — test writing and running (test/)
- `docs-rules.md` — documentation building and style (docs/)
- `examples-rules.md` — Literate.jl example conventions (examples/)
- `style-rules.md` — naming, notation, and comment style (src/, test/, validation/, examples/)
- `julia-repl-rules.md` — prefer an MCP Julia REPL over Bash when available (always)

### Skills (slash commands)

- `/run-tests` — run targeted tests, prioritized by what's likely to break
- `/build-docs` — build documentation locally
- `/add-feature` — checklist for adding new physics/features
- `/new-simulation` — set up, run, and visualize a new simulation (with or without a reference paper)
- `/babysit-ci` — monitor CI, auto-fix small issues, pause on bigger problems, retrigger flaky runs



















# CLAUDE.md — SpeedyWeather.jl Codebase Summary

## Overview

SpeedyWeather.jl is a global atmospheric general circulation model (GCM) written in Julia.
It is organised as a **Julia monorepo** with 5 interdependent packages. The design philosophy
prioritises flexibility and interactivity over production forecasting.

## Package Structure

```
SpeedyWeather.jl/
├── LowerTriangularArrays/    # Efficient storage for spherical harmonic coefficients
├── RingGrids/                 # Iso-latitude ring grids, fields, interpolation
├── SpeedyTransforms/          # Spherical harmonic transforms (FFT + Legendre)
├── SpeedyWeatherInternals/    # Device abstraction (CPU/GPU), parameters, utilities
└── SpeedyWeather/             # Main atmospheric model (depends on all above)
```

Each package has its own `Project.toml`, `src/`, and `test/` directories.
Run tests with `julia --project=<PackageName> <PackageName>/test/runtests.jl`.

## SpeedyWeather Package Layout

```
SpeedyWeather/src/
├── dynamics/          # Core dynamics (tendencies, time stepping, diffusion, …)
├── parameterizations/ # Parameterisations (convection, radiation, land, ocean, …)
│   ├── land/          # 2-layer soil bucket model
│   ├── radiation/     # Shortwave & longwave
│   └── surface_fluxes/
├── models/            # Model definitions and simulation runner
├── variables/         # Unified Variables system (prognostic, grid, tendencies, …)
├── output/            # NetCDF, JLD2, callbacks, restart files
└── input/             # Asset loading
```

## Key Types

| Type | Purpose |
|------|---------|
| `SpectralGrid` | Central config: resolution (`truncation`), grid type, `nlayers`, `NF`, architecture |
| `Variables` | Unified container for all simulation variables (prognostic, grid, tendencies, dynamics, parameterizations, particles, scratch) |
| `Simulation{V,M}` | Top-level container holding model + variables |
| `Leapfrog` | Time stepper with Robert/Williams filters |
| `SpectralTransform` | FFT/Legendre transform machinery |

### Model Type Hierarchy

```
AbstractModel
├── Barotropic           # Single-layer vorticity only
├── ShallowWater         # Single-layer with divergence + pressure
└── PrimitiveEquation
    ├── PrimitiveDry     # Multi-layer without humidity
    └── PrimitiveWet     # Multi-layer with full physics
```

All model components (orography, ocean, convection, etc.) inherit from
`AbstractModelComponent` and implement `initialize!`, optionally `timestep!`
and `finalize!`.

### Variables Structure

`Variables` is a unified container with 7 groups, each a NamedTuple:

| Group | Purpose |
|-------|---------|
| `prognostic` | State variables subject to time stepping (vorticity, divergence, temperature, pressure, tracers, etc.) |
| `grid` | Grid-point copies of spectral prognostic variables (u, v, temperature, etc.) |
| `tendencies` | Time derivatives; spectral at top level, grid-space under `.grid` namespace |
| `dynamics` | Working arrays for the dynamical core (pressure gradients, vertical velocity, etc.) |
| `parameterizations` | Working arrays for parameterizations (fluxes, radiation, cloud properties, etc.) |
| `particles` | Particle advection variables |
| `scratch` | Temporary storage (write-before-read) |

Variables can be organized by namespace (e.g. `:ocean`, `:land`, `:tracers`):

```julia
vars.prognostic.vorticity                       # Spectral vorticity
vars.prognostic.ocean.sea_surface_temperature   # Ocean namespace
vars.tendencies.vorticity                       # Spectral vorticity tendency
vars.tendencies.grid.u                          # Grid-space u-wind tendency
vars.dynamics.w                                 # Vertical velocity
```

Model components define their variables via variable type categories:
`PrognosticVariable`, `GridVariable`, `TendencyVariable`, `DynamicsVariable`,
`ParameterizationVariable`, `ParticleVariable`, `ScratchVariable`.
These are collected and allocated automatically by the `Variables(model)` constructor.

## Typical Usage

```julia
arch = SpeedyWeather.CPU()
spectral_grid = SpectralGrid(truncation=32, nlayers=8, architecture=arch)
model = PrimitiveWetModel(spectral_grid)
simulation = initialize!(model)
run!(simulation, period=Day(10), output=true)
```

## Running on GPU

To run a model on GPU, load a GPU backend package (e.g. `CUDA`, `AMDGPU`, or `Metal`)
before using SpeedyWeather, then pass `GPU()` as the architecture:

```julia
using CUDA              # or AMDGPU, Metal
using SpeedyWeather

arch = SpeedyWeather.GPU()
spectral_grid = SpectralGrid(truncation=32, nlayers=8, architecture=arch)
model = PrimitiveWetModel(spectral_grid)
simulation = initialize!(model)
run!(simulation, period=Day(10))
```

The GPU backend must be loaded first so that SpeedyWeather's extension modules
register the correct array types (`CuArray`, `ROCArray`, `MtlArray`).

GPU tests live in `SpeedyWeather/test/GPU/` and auto-detect the backend (`runtests.jl` tries
`AMDGPU`, then `CUDA`, then `Metal`). Each backend has its own self-contained test environment
under `SpeedyWeather/test/GPU/<Backend>/Project.toml` (with `SpeedyWeather` resolved via a
`[sources]` path entry, transitively pulling in its monorepo siblings), so no manual
`Pkg.develop`/`Pkg.add` is needed — just `instantiate` and run:

```bash
julia --project=SpeedyWeather/test/GPU/CUDA -e 'using Pkg; Pkg.instantiate()'
julia --project=SpeedyWeather/test/GPU/CUDA SpeedyWeather/test/GPU/runtests.jl
```

Substitute `AMDGPU` or `MetalGPU` for the backend directory as appropriate.

## Time-Step Information Flow

```
Spectral State (LowerTriangularArray)
    ↓  transform!()
Grid Variables (RingGrids Field)
    ↓  dynamics_tendencies!() + parameterization_tendencies!()
Grid Tendencies
    ↓  transform!() back to spectral
Spectral Tendencies
    ↓  timestep!() — Leapfrog + implicit correction
New Spectral State
```

Prognostic variables live in `vars.prognostic`, tendencies in `vars.tendencies` (spectral)
and `vars.tendencies.grid` (grid-space). Vorticity and divergence are scaled by `radius`
inside `run!` for spectral efficiency.

## Array Types 

Mainly two array types are used: `LowerTriangularArray` for spectral coefficients, and `Field` for gridded data. They can be initialized for testing as in the following: 

```julia 
arch = SpeedyWeather.CPU()
spectrum = Spectrum(truncation=10, architecture = arch)
nlayers = 5
coeffs_zero = zeros(ComplexF32, spectrum, nlayers)
coeffs_rand = rand(ComplexF32, spectrum, nlayers)

grid = HEALPixGrid(6, arch)
field_zero = zeros(Float32, grid, nlayers)
field_rand = rand(Float32, grid, nlayers)
```

In case there's already a `SpectralGrid` use it instead: 

```julia
arch = SpeedyWeather.CPU()
spectral_grid = SpectralGrid(truncation=11, architecture=arch)
coeffs = rand(ComplexF32, spectral_grid.spectrum)
field = rand(Float32, spectral_grid.grid)
```

## Key Source Files

| File | Notes |
|------|-------|
| `dynamics/tendencies.jl` (~1200 lines) | All dynamical tendency calculations |
| `dynamics/time_integration.jl` (~450 lines) | Main time loop (`time_stepping!`) |
| `dynamics/implicit.jl` | Semi-implicit gravity-wave treatment (allows CFL > 1) |
| `models/simulation.jl` | `run!`, `initialize!`, `finalize!` for `Simulation` |
| `parameterizations/convection.jl` | Betts-Miller convection |
| `parameterizations/large_scale_condensation.jl` | Implicit condensation |
| `output/netcdf_output.jl` (~540 lines) | NetCDF writing with interpolation |
| `output/callbacks.jl` | Hook system for runtime code injection |

## Numerical Methods

- **Spectral transform**: FFT in longitude + Associated Legendre polynomials in latitude
- **Time integration**: Leapfrog with Robert (1966) and Williams (Amezcua 2011) filters
- **Semi-implicit**: Gravity waves treated implicitly; allows larger timesteps
- **Grids**: Full and reduced (octahedral, HEALPix) Gaussian and Clenshaw-Curtis grids
- **Precision**: Float32 (default), Float64, experimental BFloat16 + stochastic rounding
- **GPU**: Device-agnostic via `KernelAbstractions` (CUDA, Metal, AMDGPU)

## Extending the Model

- **Custom component**: Inherit abstract type → implement `initialize!(c, model)` → pass as kwarg to model constructor
- **Custom parameterisation**: Use `CustomParameterization` slot in `PrimitiveWet/Dry`
- **Callbacks**: Implement `AbstractCallback`, attach via `model.callbacks`
- **Output variables**: Add/remove via `NetCDFOutput` variable list

## Testing

- Every new method that is implemented has to be tested in the unit tests of its respective submodule
- Keep the unit tests short and concise, they should finish quickly
- Call the test with the `--check-bounds=yes` flag activated
- Test files in `test/<subdir>/` are auto-discovered by `find_tests`; just drop a new `.jl` file
  there. For features implemented in **extensions** (`ext/SpeedyWeather*Ext.jl`), tests still
  live in the main package's `test/` tree — add the optional dependency to `test/Project.toml`
  and `using` it at the top of the test file. Do not consider a feature complete until its
  unit tests have been added and run successfully.

**Which `--project` to use.** `SpeedyWeather/Project.toml` has only the package's own
dependencies; the test-only ones (`JET`, `Enzyme`, `NCDatasets`, `CairoMakie`, `Terrarium`, …)
live in `SpeedyWeather/test/Project.toml`. So:

- `--project=SpeedyWeather` — only for a single test file that needs nothing beyond
  `SpeedyWeather` and `Test`.
- `--project=SpeedyWeather/test` — for anything else: files that `using` a test-only
  dependency (e.g. `dynamics/dispatch.jl` needs `JET`), or running several test files by hand.
  When in doubt use this one; it is a superset.

`Pkg.test(...)` builds its own environment and works from either.

```bash
# Single test file with no test-only dependencies (fast — load Test manually, skip Pkg.test overhead)
julia --project=SpeedyWeather --check-bounds=yes -e 'using Test, SpeedyWeather; include("SpeedyWeather/test/dynamics/vertical_coordinates.jl")'

# Single test file that needs test-only dependencies, or several files at once
julia --project=SpeedyWeather/test --check-bounds=yes -e 'using Test, SpeedyWeather; include("SpeedyWeather/test/dynamics/dispatch.jl")'

# Main model tests
julia --project=SpeedyWeather --check-bounds=yes -e 'using Pkg; Pkg.test("SpeedyWeather")'

# Extended tests
julia --project=SpeedyWeather SpeedyWeather/test/runtests.jl extended_tests

# Individual packages
julia --project=RingGrids --check-bounds=yes -e 'using Pkg; Pkg.test("RingGrids")'
```

Run these from the repository root — the `include` paths above are relative to it.

Test subdirectories: `dynamics/`, `parameterizations/`, `output/`, `GPU/`, `differentiability/`.

**Stale manifests.** `Manifest.toml` files are gitignored (except the deliberately pinned
`SpeedyWeather/test/differentiability/sensitivity_examples/Manifest.toml`). If loading fails
during precompilation of a *stdlib* — a symptom of a manifest resolved for an older Julia than
the one on `PATH` — check `grep -m1 julia_version <env>/Manifest.toml` against
`julia --version`, then regenerate: `rm <env>/Manifest.toml` and
`julia --project=<env> -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'`. CI covers both
Julia 1.10 and 1.12, so either is a valid target.

## Documentation

- Stable docs: https://speedyweather.github.io/SpeedyWeatherDocumentation/stable/
- Dev docs:    https://speedyweather.github.io/SpeedyWeatherDocumentation/dev/
- Build: `julia --project=docs docs/make.jl`

## Kernel Launching

Kernel infrastructure lives in `SpeedyWeatherInternals/src/KernelLaunching/KernelLaunching.jl`
(a standalone submodule of `SpeedyWeatherInternals`) and `SpeedyWeatherInternals/src/Architectures/`.
Other packages import it via `using SpeedyWeatherInternals.KernelLaunching`.

### Architecture abstraction

```julia
CPU()        # wraps KernelAbstractions.CPU()
CPUStatic()  # wraps KernelAbstractions.CPU(; static = true)
GPU()        # wraps backend-specific GPU (CUDA default; Metal/AMDGPU via extensions)
```

Backend-specific array types (`CuArray`, `ROCArray`, `MtlArray`) are registered
in the extension modules under `SpeedyWeatherInternals/ext/`.
Use `on_architecture(device, data)` to transfer data between devices.

### Work order types

Six types encode what iteration space a kernel covers:

| Type | Iteration space |
|------|----------------|
| `SpectralWorkOrder` | all `lm` harmonics × vertical layers |
| `SpectralInnerWorkOrder` | as above, but skip `lm=1` |
| `DiagonalWorkOrder` | diagonal elements of a `LowerTriangularArray` |
| `RingGridWorkOrder` | all grid points `ij` × vertical layers |
| `ArrayWorkOrder` | regular arrays preserving dimensionality |
| `LinearWorkOrder` | flattened 1D (eachindex) |

### Launching a kernel

```julia
launch!(arch, WorkOrderType, worksize, kernel!, args...)
```

Internally this calls `work_layout` to compute a sensible `workgroup` size
(capped at 256 threads, heuristic aspect ratios for 2D/3D), then dispatches
via the KernelAbstractions API:

```julia
loop! = kernel!(device(arch), workgroup, worksize)
loop!(args...)
```

### Defining a kernel

```julia
@kernel inbounds = true function my_kernel!(A, B, c)
    I = @index(Global, Linear)      # or NTuple / Cartesian for multi-dim
    A[I] = B[I] * c
end
```

- `inbounds = true` — standard on all performance kernels (28+ files)
- Use `@index(Global, Linear)` for 1D / flattened access,
  `@index(Global, NTuple)` for `ij, k` style,
  `@index(Global, Cartesian)` for `I[1], I[2]` style

### CPU vs GPU dispatch

Many functions branch on architecture type:

```julia
arch = architecture(field)
if arch isa GPU
    launch!(arch, LinearWorkOrder, (n,), my_kernel!, ...)
else
    my_cpu_loop!(...)   # plain Julia loops
end
```

Kernels are spread across ~36 files in `SpeedyWeather/src/dynamics/`,
`parameterizations/`, `SpeedyTransforms/src/`, `RingGrids/src/`, and
`LowerTriangularArrays/src/`.

### Branch-free code for Reactant

Write functions that are broadcast or passed to `set!` branch-free, so that Reactant can trace them:
use `ifelse(cond, a, b)`, `min`, `max` or `clamp` instead of `if`, `? :`, `&&`, `||` or an early
`return` on a value. Branches inside `@kernel`s are fine.

## Benchmarks

Benchmarks live in `SpeedyWeather/benchmark/`. Run the full suite with:

```bash
julia --project=SpeedyWeather/benchmark SpeedyWeather/benchmark/manual_benchmarking.jl
```

Results are written to `SpeedyWeather/benchmark/README.md`. Previous benchmark results
in that file serve as the baseline. When running benchmarks, compare new results against
the existing README.md values: deviations of +/- 20% are normal and acceptable, but
larger regressions should be reported to the user.

## Code Style

* The project uses [Runic.jl](https://github.com/fredrikekre/Runic.jl) for formatting.
* Prefer descriptive variable names that spell words out in full rather than abbreviating
  them, e.g. `temperature` not `temp`, `surface_pressure` not `surf_pres`,
  `layer_thickness` not `lyr_thk`, `spectral_grid` not `sg`. Established physics/maths
  shorthand that is itself the conventional name (e.g. `coslat`, `vor`, `div`) is fine.
* Spectral truncations are 1-based: always use the new 1-based truncations `T32`, `T64`, `T128`, …
  (e.g. `SpectralGrid(truncation = 32)`) in code, tests and docs, never the old 0-based
  `T31`, `T63`, `T127`, ….


## Pull Request Convention

Every PR **must** add a line to `CHANGELOG.md` under the `## Unreleased` section:

```
- Brief description of change [#NNN](https://github.com/SpeedyWeather/SpeedyWeather.jl/pull/NNN)
```

Add it as the first bullet under `## Unreleased`.

## Submodule Versioning

Each of the 5 packages (`LowerTriangularArrays`, `RingGrids`, `SpeedyTransforms`,
`SpeedyWeatherInternals`, `SpeedyWeather`) has its own `version = "..."` line in
its `Project.toml`. When a PR modifies the source of a submodule, the version of
that submodule must be bumped according to its size:

- **Minor change** (bug fix, internal refactor, additive non-breaking feature):
  append `+DEV` to the current version, e.g. `0.20.1` → `0.20.1+DEV`. The `+DEV`
  tag accumulates until the next release, signalling "there are unreleased
  changes on top of `0.20.1`".
- **Bigger change** (breaking API change, new public type, major feature):
  bump the version according to standard semantic versioning *and* append `-DEV`
  e.g. `0.20.1` → `0.21.0-DEV`. The `-DEV` tag marks the new version as in-progress until
  release.

Bump the version of every submodule whose source you actually touched in the PR.
A change confined to `SpeedyWeather/` does not require bumping
`LowerTriangularArrays`, etc.

## Implementation plans

All major feature additions, bug fixes, or refactoring that requires substantial changes to the existing code must be prefaced with an **implementation plan** that is reviewed and signed off by a human. These plan documents should be organized by date and stored in `docs/dev/YYYY-MM`. Each document should be prefaced by the following template:

```md
# Descriptive title

> Status: **planned**/**in progress**/**completed**. One sentence summary of current status.

Date of initial draft: YYYY-MM-dd

Base revision: <SHA1 of HEAD when plan was drafted>

## Originating prompt

> User prompts here

## Revision log

> User prompts here

## Problem description

## Background

```

The revision log should, to the greatest extent possible, briefly summarize changes to the plan that are made on-the-fly during development.

The remainder of the plan document may be adapted on a case-by-case basis but should generally follow this structure:

```md
## Summary of changes

## Testing and verification

## Documentation changes

## Known limitations

## Future work
```




























# Terrarium.jl — Agent Rules

Note: This `AGENTS.md` file is adapted from [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl/blob/main/AGENTS.md). See relevant copyright notices therein.

## Project Overview

Terrarium.jl is a Julia package for fast, friendly, flexible, and process-based land and
terrestrial ecosystem modeling on CPUs and GPUs. It solves the coupled heat and Richards
equations in 1D for the soil along with common 0D parameterizations of vegetation and surface
hydrology processes. Terrarium is designed to be modular in the sense that (almost) all components
should be exchangeable with alternative implementations. Terrarium is also intended to be a fully
differentiable land model with continuous-time dynamics. All process implementations must be defined
in terms of well-formed ordinary or partial differential equations. No instantaneous or discrete-time
dynamics are allowed except in very special cases where they must be clearly documented and justified.

## Language & Environment

- **Julia 1.10+** | CPU and GPU (CUDA)
- **Key packages**: KernelAbstractions.jl, CUDA.jl, Enzyme.jl
- **Style**: Explicit imports for source code; `using Terrarium` for examples/tests

## Testing

- Test-only dependencies (e.g. `SpecialFunctions`, `CUDA`) live in `test/Project.toml`, not the
  root project, so the test files cannot be `include`d from the bare `--project=.` environment.
- To run individual test files, activate the test environment from the project environment with
  [TestEnv.jl](https://github.com/JuliaTesting/TestEnv.jl):
  ```julia
  # started with: julia --project=.
  using TestEnv; TestEnv.activate()
  include("test/soil/soil_energy_tests.jl")
  ```
- To run the full suite, use `julia --project=. -e 'using Pkg; Pkg.test()'`.
- Only run the Enzyme (`Pkg.test(; test_args=["enzyme"])`) or Reactant (`test/reactant/`) test suites
  locally after directly editing the respective extension modules or unit tests, or when explicitly requested.
  These tests should otherwise be left to the CI/CD pipeline.
- Julia errors and associated stack traces are often very long due to long type signatures. To
  mitigate this, always write test output to temporary files and analyze this output using `grep`
  and similar tools.
- After running the full test suite, always run a draft doc build with ``julia --project=docs docs/make.jl --local --draft` and check for errors due to, e.g. stale docstring entires. Ignore errors from missing files due to the example scripts not being executed.
- The docs environment must load the local checkout of Terrarium, not a registered release; otherwise the build documents
  the release, and every docstring added since shows up as an "undefined binding" error. Before the first draft build in
  a new clone or worktree, run `julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'`
  (as the documentation CI workflow does) and check that `docs/Manifest.toml` lists a `path` for Terrarium. Errors about
  InterLinks inventories that cannot be loaded, and the unresolved external links that follow from them, mean that there
  is no network access and can be ignored.

## Critical Rules

### Kernels (GPU compatibility)

- Use `@kernel` / `@index` (KernelAbstractions.jl) to define device-agnostic kernels
- Functions marked with `@kernel` are called **kernels** which then invoke **kernel functions** with call pattern `compute_something(i, j, k, grid, fields, process::ProcessType, args...)` or `compute_something!(out, i, j, k, grid, fields, proces::ProcessType)`
    - Kernel functions defined for 2D kernels instead have `i, j` instead of `i, j, k`
- Kernels and their subsequent call graph must be fully **type-stable** and **allocation-free**
- Use `ifelse` — never short-circuiting `if`/`else` in kernels. `ifelse` evaluates **both**
  branches, so never index in a branch that may be out of range: select the *index*, not the
  loads (write `x[ifelse(cond, i, n)]`, not `ifelse(cond, x[i], x[n])`)
- No error messages, no `AbstractModel`s, and no `state` inside kernels. This bans **any reachable
  throw path** — `@assert`, bounds errors from un-elided `@inbounds`, `InexactError` from
  `convert`/`round(Int, …)`, integer `DivideError` — because under Reactant these lower to
  `llvm.intr.trap`, which cannot be raised to StableHLO and fails compilation *even if never
  triggered*. Put argument validation in host-side (keyword) constructors, not in the positional
  constructors that kernels call
- Always extract relevant input/output `Field`s with `get_fields` and related methods
- Favor explicit enumeration of process types when invoking kernels rather than passing `AbstractCoupledProcesses` types
- Mark functions called inside kernels with `@inline` or `@propagate_inbounds` when including indices
- **Never loop over grid points outside kernels** — use `launch!`

### Differentiability & Enzyme.jl

Terrarium targets full differentiability for inverse modeling and sensitivity analysis. Compatibility with Enzyme.jl
is a top priority and must be continuously tested.

- **Ensure type stability**: All code in kernels and within state-mutating methods (e.g. `initialize!`, `compute_tendencies!`, `compute_auxiliary!`) must be fully type stable.
- **Minimize allocations**: Allocations should be avoided wherever possible. Prefer mutation of output `Field`s. Never mutate input or prognostic `Field`s outside of `update_inputs!` or timestepper `timestep!` respectively.
- **No global state**: Initialize all parameters explicitly; never rely on global variables or implicit state
- **Test differentiability**: Use Enzyme to test that critical functions compute valid adjoints; include in test suite. See existing tests in `test/differentiability` for reference.
- **Document AD limitations**: If a function cannot be differentiated, mark it clearly with comments and docstrings

### Reactant compatibility

Terrarium runs through Reactant.jl (trace → MLIR/StableHLO → XLA). All Reactant-specific code lives
in `TerrariumReactantExt`; the user's only knob is the architecture, `ReactantState()`. A model on a
`ReactantState` grid allocates its state directly on the device grid and is initialized *eagerly*
on the device (eager KernelAbstractions launches run on the Reactant backend; Oceananigans'
`set_to_function!` for a device field detours through the CPU internally). Only
`run!`/`timestep!`/`run_timesteps!` are traced and compiled by XLA. Correctness is tested in
`test/reactant/` (own `Project.toml`, CI `Reactant_CI.yml`).

- **No reachable throw paths in kernels** — see the Kernels rule above (this is the single most
  common Reactant compile failure).
- **`using CUDA` is required alongside Reactant, even on CPU**: it provides the KernelAbstractions↔Reactant
  glue. Without it, kernel launches fail with `MethodError: ka_with_reactant(::Nothing, …)`.
- **Closures compiled into kernels must capture only `isbits` values.** A boundary-condition
  function that closes over a `Type` (e.g. `NF`/`Float32`) becomes a non-`isbits` kernel argument
  and fails to compile — hoist numeric constants out (`amplitude = NF(5); bc(x, t) = amplitude * …`).
- **Reverse-mode AD**: differentiate `run_timesteps!` with `Enzyme.autodiff(set_strong_zero(ReverseWithPrimal),
  …, Duplicated(integrator, dintegrator), …)` inside `@compile raise=true raise_first=true sync=true`.
  Pass a `checkpointing` scheme (`Reactant.Periodic(n)`) to `run!`/`run_timesteps!` to bound reverse-pass
  memory; it must not change the gradient. See `test/reactant/autodiff.jl`.
- **Do not run `test/reactant` under `--check-bounds=yes`**: forced bounds checks make every kernel
  un-raisable (see the trap rule); `runtests.jl` guards against this.

### Type Stability & Memory

- All structs must be concretely typed
- Minimize allocation; favor inline computation
- **Never hardcode Float64**: no literal `0.0` or `1.0` in kernels or constructors.
  Use `zero(grid)`, `one(grid)`, `NF(x)` where `x` is a number, `convert(FT, 1//2)`, or rational literals

### Type Annotations

- Type annotations are used to **dispatch to relevant types**, restrict method signatures, and enable compiler optimizations
- Type annotations express intent and document assumptions
- Annotate function arguments and struct fields to be as specific as possible but not more specific than necessary
- Avoid use of overly broad types like `Any` unless absolutely necessary; instead, use Union types or create a common abstract type. Use of `Any` is permitted for generic containers such as `state` and `grid`.
- Use `where` clauses to express type constraints that improve clarity and dispatch precision

### Imports

- Source code: explicit imports (checked by tests)
- Examples/docs: rely on `using Terrarium`; never explicitly import exported names

### Docstrings

- Use DocStringExtensions.jl with `$TYPEDSIGNATURES`
- **ALWAYS `jldoctest` blocks, NEVER plain `julia` blocks** — doctests are tested; plain blocks rot
- Include `# output` with verifiable output; prefer `show` methods over boolean comparisons
- Use unicode for math (`Δt`, `η`, `ρ`), not LaTeX — LaTeX doesn't render in the REPL
- Use parentheses instead of brackets for units (e.g. (m/s) instead of [m/s])
- Use DocumenterCitations.jl for references, e.g. `[GoerzQ2022](@cite)`. Make sure to include the Bibtex entry in `docs/src/references.bib`. Also add a `# References` section to the docstring, where you add e.g. `* [GoerzQ2022](@cite) Goerz et al. Quantum 6, 871 (2022)`

### Documentation pages

- Doc pages for processes and models should always consist of the following sections:
    - **Overview**: General overview of of the physical process, what the main inputs and output variables typically are, and general equations relevant for understanding the implementations. This section should not contain implementation-specific details.
    - **Implementations**: There should be sections for each implementation of the abstract process type, describing the general theory and any relevant implementation details. These sections should also each include a non-canonical docstring of the concrete process type.
        - For each concrete process implementation, provide docstrings for the implementation-specific dispatches of `initialize!`, `compute_auxiliary!`, and `compute_tendencies!` corresponding to each coupling interface.
    - **Methods**: Enumeration of process-specific methods.
    - **Kernel functions**: Enumeration of **kernel functions** of the form `compute_something(i, j, k, grid, fields, ...)` or `compute_something!(out, i, j, k, grid, fields, ...)`; do NOT include **kernels** (i.e. functions annotated with `@kernel`)
- All functions referenced in doc pages should be marked with `canonical = false` since the canonical versions of the docstrings are defined in a separate `@autodocs` block in the index
- All types and functions referenced in the doc pages must have docstrings otherwise the doc build will fail. Ensure docstrings are defined and add them if they are missing.
- Doc pages should always be prefaced with appropriate `@meta` and `@setup` blocks
- If a model or process is not fully implemented, an appropriate warning should be displayed on the doc page
- Do not use brackets for expressing units as this conflicts with Markdown link syntax; use parentheses instead
- All code examples should be given as `@example name` blocks, replacing `name` with an appropriate identifier for the page, which are executed by Documenter.jl.
- A fast version of the doc build can be executed with `julia --project=docs docs/make.jl --local --draft`

### Model Constructors

- `grid` is positional: `SoilModel(grid; initializer)`
- Omit semicolon when there are no keyword arguments: `SoilModel(grid)`

## Naming Conventions

- **Files**: snake_case matching the type they define — `soil_hydrology.jl`
- **Types/Constructors**: PascalCase **only for true constructors** — `SoilHydrology`
- **Functions**: snake_case — `compute_tendencies!` unless commonly combined in English, e.g. `timestep!`. This is not a hard rule, exceptions are permitted.
- **Kernels**: should always be suffixed with `_kernel!` — `compute_tendencies_kernel!`
- **Kernel functions**: should always be prefixed with `compute_`; mutating variants should use the standard bang `!` convention.
- **Variables**: Prefer English long name or readable unicode math notation — do not use abbreviations that may introduce ambiguity, e.g. `cond` could be either "condition" or "conductivity"; be as specific as possible.

## Physics & Process Equations

All dynamical processes must be grounded in physics:

- **Equation reference**: Every process implementation should cite the governing equations in code comments or docstrings
- **Continuous time**: Discrete-time updates (e.g., "update this once per day") are prohibited
- **Conservation where applicable**: If a process conserves a quantity (mass, energy), verify conservation in tests
- **Nondimensionalization**: Consider whether nondimensionalization would improve solver stability; document choices if used

## Module Structure

```
src/
├── Terrarium.jl                    # Main module, exports
├── abstract_model.jl               # CPU/GPU architecture abstractions
├── diagnostics/                    # Diagnostic and debugging utilities
├── grids/                          # Grid types and discretizations
├── input_output/                   # Types and functions for managing inputs and outputs
├── models/                         # Model implementations
├── processes/                      # Process implementations
├── timesteppers/                   # Time stepping schemes, integrators, and integration with Oceananigans `AbstractModel`
├── utils/                          # Miscellaneous utilities
```

## Common Pitfalls

1. **Type instability** in kernels ruins GPU performance
2. **Missing imports**: tests will catch this — add to `using` statements
3. **Plain `julia` blocks in docstrings**: always use `jldoctest`
4. **Subtle bugs from missing method imports**, especially in extensions
5. **Expecting unexported names**: consider exporting them rather than changing user scripts
6. **Extending `getproperty` to fix undefined property bugs**: fix on the caller side instead
7. **"Type is not callable" errors**: variable name shadows a function — rename or qualify
8. **Quick fixes that break correctness**: if a test fails after a change, revisit the original edit
9. **Commented-out code**: delete it. Git is the journal — don't leave commented code, debugging artifacts, or stale copy-paste remnants
10. **2D indexing on fields**: always use 3D indexing (`field[i, j, k]`). 2D indexing works by coincidence on some fields but is unsupported and will break
11. **Hardcoded Float64**: never use `0.0`, `1.0` in kernels or constructors; use `zero(grid)` etc.
12. **Scope creep in PRs**: keep changes focused on a single concern. Unrelated cleanup goes in a separate PR
13. **Modifying Project.toml dependencies**: never add, remove, or change `[deps]` or `[weakdeps]` in the root `Project.toml` unless the task absolutely requires it. Dependency changes have wide-reaching consequences — they affect CI, load time, and downstream compatibility. Only touch `[compat]` bounds when explicitly asked.
14. **Mutable function closures in kernels**: function closures that capture mutable state will not differentiate correctly — use explicit parameters instead
15. **Non-local dependencies in process equations**: process functions must not depend on global state; pass all dependencies as arguments for traceability and differentiability

## Benchmarks

Performance benchmarks live in `benchmark/` and are run manually, one architecture at a time:

```bash
cd benchmark
julia --project=. manual_benchmarking.jl                # CPU (auto-labelled cpu-arm or cpu-x86)
julia --project=. manual_benchmarking.jl gpu            # CUDA GPU
julia --project=. manual_benchmarking.jl reactant-cpu   # Reactant/XLA, CPU backend (auto-labelled reactant-cpu-arm or reactant-cpu-x86)
julia --project=. manual_benchmarking.jl reactant-gpu   # Reactant/XLA, CUDA backend
```

A second argument sets the duration: `quick` (0.25x time steps, resolution sweeps capped — use this to
check that the harness and configurations still run), `long` (10x, for numbers worth publishing), or a
numeric multiplier.

Each run merges its results into `benchmark/assets/benchmark_results.json` and regenerates
`benchmark/README.md` from the whole store, so running on one machine never overwrites another
architecture's numbers. The documentation page `docs/src/benchmarks.md` is generated from the same JSON
at doc-build time.

The committed README numbers are the regression baseline: deviations of ±20% are normal, larger ones
should be reported. Model configurations are defined in `benchmark/model_configurations.jl`; add one by
adding a `build_model(::Val{:name}, arch, NF; nlat_half, nz, model_kwargs)` method, following the same
registry pattern as `test/reactant/setup.jl`. 

Only ever run these benchmarks when explicilty instructed to or after a major revisions of the model compute functions. 

## Git Workflow

Follow [ColPrac](https://github.com/SciML/ColPrac). Feature branches, descriptive commits, update tests and docs with code changes, check CI before merging.

## Design Principles

- **Dispatch over conditionals**: use Julia's type system and multiple dispatch instead of `if`/`else` branching. Backend-specific code goes in `ext/` extensions, not `if` branches in `src/`
- **Use `on_architecture` for data transfers** — never manual `Array()` / `CuArray()` calls
- **Defaults serve the common case**: avoid `nothing` defaults when a concrete default (like `CPU()`) covers 80% of usage. Minimize boilerplate for the typical user.
- **Keyword argument names must be consistent** across related types and constructors
- **Always use explicit `return`** in functions longer than one expression
- **One operation per line** as default; break long expressions across lines

## Agent Behavior

- Prioritize type stability, GPU compatibility, and differentiability
- Follow established patterns in existing code
- Add tests for new functionality; update exports when adding public API
- Reference physics equations in comments when implementing dynamics
- Do not make unsolicited changes; focus on specific tasks
- When extending Enzyme.jl compatibility, verify adjoints with `Enzyme.autodiff`
- Ensure type annotations are restrictive enough to guide dispatch and minimize misuse

## Implementation plans

All major feature additions, bug fixes, or refactoring that requires substantial changes to the existing code must be prefaced with an **implementation plan** that is **reviewed and signed off by a human**. These plan documents should be organized by date and stored in `docs/dev/YYYY-MM`. You are **ABSOLUTELY REQUIRED** to get the explicit approval of a human developer before executing the implementation.

Each document should be prefaced by the following template:

```md
# Descriptive title

> Status: **planned**/**in progress**/**completed**. One sentence summary of current status.

Date of initial draft: YYYY-MM-dd

Author: <Name of model and version (if known)>, <harness name and version (if known)>

Reviewer: <Name of human reviewer> (<reviewer email address>)

Base revision: <SHA1 of HEAD when plan was drafted>

## Originating prompt

> User prompts here

## Revision log

> User prompts here

## Problem description

## Background

```

Notes on filling in the template (these notes are guidance about the template; do not copy them into
the plan document itself):

- Leave the `Reviewer` field blank until a human has approved the plan, then fill in both name and
  email after approval. Where the reviewer is the repository owner, `git config user.name` and
  `git config user.email` are an acceptable source for those values.
- The revision log should, to the greatest extent possible, briefly summarize changes to the plan
  that are made on-the-fly during development. Give each revision a number and a date, and record
  the model name and version (if known) for that revision, since a plan may be revised by a
  different model than the one that drafted it. **A human must approve each revision before
  implementation**.

Write plan prose with a line break at the end of every sentence (one sentence per line) rather than wrapping at an arbitrary column width.
This keeps diffs readable when individual sentences are revised.

The remainder of the plan document may be adapted on a case-by-case basis but should generally follow this structure:

```md
## Summary of changes

## Testing and verification

## Documentation changes

## Known limitations

## Future work
```

## Further Reading

For detailed guidance on specific workflows:
- `test/runtests.jl` — test organization and running conventions