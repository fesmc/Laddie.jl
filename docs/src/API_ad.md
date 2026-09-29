# Automatic differentiation (API)

```@index
Pages = ["API_ad.md"]
```

A run can be differentiated in two ways (see [Inverse problems](@ref) for both on one
example):

  * **ForwardDiff.jl**, on the CPU and CUDA backends: build the model at a `Dual` element
    type and run it with [`integrate!`](@ref). Cost grows with the number of parameters.
  * **Reactant.jl** with Enzyme, forward and reverse mode: compile a function of the
    model with [`reactant_compile`](@ref), with the parameters as inputs through
    [`trace_parameters`](@ref). Reverse mode gives the gradient with respect to every
    input for a fixed cost. See [Reactant backend](@ref) for setup and cost.

## The model as a function

```@docs
integrate!
```

## Reactant and Enzyme

```@docs
reactant_compile
trace_parameters
```

## Adaptive time step

```@docs
adaptive_schedule
DtSchedule
```
