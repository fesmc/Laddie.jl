# Ice conduction

The interface heat budget has an ice-side term, the heat conducted into the ice
(Wiskandt & Jourdain 2025, Eqs. 1–2):

```math
\dot m\,L = c_p\,\gamma_T\,(T - T_b) + q_\text{cond},
```

with ``q_\text{cond}`` negative for an ice surface colder than the base. The ice column
is not resolved, so ``q_\text{cond}`` is a closure, selected with
`Params(; ice_conduction = ...)`.

The default term ``-c_i (T_i - T_b)\,\dot m`` in ``L_\text{eff}`` (see
[Melting](melting.md)) is the linearisation of Holland & Jenkins (1999), which assumes that
the ice velocity is proportional to the melt rate. The other closures are those compared by
Wiskandt & Jourdain (2025):

| closure | heat conducted into the ice | when it is reasonable |
|---|---|---|
| [`HollandJenkinsConduction`](@ref) (default) | ``\dot m\,c_i\,(T_b - T_i)``, melting and refreezing | melting; the v1/v2 behaviour |
| [`MeltOnlyConduction`](@ref) | the same for ``\dot m > 0``, zero otherwise | as above, and a refreezing base of marine ice near the freezing point |
| [`NoConduction`](@ref) | none, ``L_\text{eff} = L`` | marine ice only; overestimates melt by ``c_i (T_s - T_f)/L`` (≈ 11 % for ``T_s = -20`` °C) |
| [`LinearProfileConduction`](@ref) | ``k_i (T_f - T_s)/H_i``, independent of ``\dot m`` | thick, slow ice shelves in thermal steady state (e.g. Ross) |

The switch of `MeltOnlyConduction` is the sign of ``T - T_f(S, z_b)``, which is the sign of
``\dot m``, so the melt equations are solved once. `LinearProfileConduction` takes ``T_f``
at the layer salinity so that the flux does not depend on ``\dot m`` and the equation
stays quadratic, and the ice thickness from the draft by hydrostatic equilibrium,
``H_i = -z_b\,\rho_0/\rho_\text{ice}``. A thermal steady state between base and surface
takes of order ``(H_i/2)^2/\kappa_i``, about 1000 years for 400 m of ice, which only the
large, slow ice shelves reach.

In the kernels, every closure reduces to two factors on ``c_i`` (where the layer melts and
where it does not) and a fixed conductive flux, which amounts to evaluating the melt
equations at ``T - q/(c_p\gamma_T)`` in place of ``T``; see the
[equation reference](equations.md).

!!! warning "Experimental"
    Every closure but the default is experimental, and constructing one logs a warning.
    The `T_ice_base` of the ice forcing is used as the ice temperature ``T_i`` by all of
    them. In Holland & Jenkins (1999) and Wiskandt & Jourdain (2025) this is the ice-shelf
    *surface* temperature ``T_s`` for every closure (the ice at the base itself is at the
    freezing point); Wiskandt & Jourdain use the 1980–2019 mean snow-surface temperature.

## API

```@docs
AbstractIceConduction
HollandJenkinsConduction
MeltOnlyConduction
NoConduction
LinearProfileConduction
```
