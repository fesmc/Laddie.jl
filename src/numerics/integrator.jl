"""
Abstract supertype for the time integration of the prognostic fields, the scheme that
`time_step!` and `integrate!` advance a [`Model`](@ref) with. Pass a concrete instance as
`Params(; integrator = ...)`: [`LeapfrogIntegrator`](@ref) (the default). See
[Time stepping](@ref).
"""
abstract type AbstractIntegrator end

"""
$(TYPEDEF)

Leapfrog with a Robert–Asselin filter, as LADDIE v1:

```
qⁿ⁺¹ = qⁿ⁻¹ + 2Δt F(qⁿ, qⁿ⁻¹),    qⁿ ← qⁿ + ν/2 (qⁿ⁻¹ + qⁿ⁺¹ − 2qⁿ)
```

The filter damps the computational (2Δt) mode. The first step, and the first after a change
of Δt, is a forward step over Δt.

Select via `Params(; integrator = LeapfrogIntegrator(; nu = 0.8))` (the default).

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct LeapfrogIntegrator{FT} <: AbstractIntegrator
    "Robert–Asselin filter coefficient ν (–, default `0.8`)"
    nu::FT = 0.8
end
