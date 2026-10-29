# Equation of state

The equation of state sets the buoyancy of the layer against the ambient cavity water,
as the reduced density contrast

```math
\delta\rho = \frac{\rho_a - \rho}{\rho_0},
```

which pushes the light meltwater up the draft. It is the one place where the EOS enters
Laddie.jl: through ``\delta\rho`` in the pressure gradient, the entrainment and the
convective stability.

Every equation of state has the form

```math
\delta\rho = \beta\,(S_a - S) - \bigl(a_0 + a_T\,\bar T + a_Z\,Z\bigr)(T_a - T),
\qquad \bar T = \tfrac12 (T_a + T),
```

with ``Z`` the depth of the layer mid-point, ``-z_b + H/2``. The thermal expansion
``\alpha = a_0 + a_T \bar T + a_Z Z`` is linear in temperature and in depth, and the
equations of state differ only in the four coefficients. The seawater equation of state is
nonlinear mostly through ``\alpha``, which falls towards the freezing point
(**cabbeling**) and rises with depth (**thermobaricity**).

| equation of state | ``\delta\rho`` | ``\alpha`` |
|---|---|---|
| [`LinearEOS`](@ref) (default) | ``\beta\,\Delta S - \alpha\,\Delta T`` | constant, `Params.alpha` |
| [`RoquetEOS`](@ref) | ``\beta\,\Delta S - (a_0 + a_T \bar T + a_Z Z)\,\Delta T`` | ``\bigl[C_b(\bar\Theta - \Theta_0) + T_h Z\bigr]/\rho_0`` |

where ``\Delta X = X_a - X``.

## Linear

[`LinearEOS`](@ref) uses the constants `Params.alpha` and `Params.beta`
(``a_T = a_Z = 0``). It is the equation of state of [LADDIE v1 and v2](../versions.md), and
reproduces their contrast to the last bit.

## [Roquet et al. (2015)](@cite roquet_defining_2015)

[`RoquetEOS`](@ref) is the simplified polynomial of [Roquet et al. (2015)](@cite roquet_defining_2015)
(their Eq. 17). In the paper's own variables, Conservative Temperature ``\Theta`` (°C), Absolute
Salinity ``S_A`` (g kg⁻¹) and depth ``Z`` (m), the density anomaly is

```math
\rho'(\Theta, S_A, Z) = -\tfrac{C_b}{2}\,(\Theta - \Theta_0)^2 - T_h\,Z\,\Theta + b_0\,S_A,
```

with ``C_b = 0.011`` kg m⁻³ K⁻², ``\Theta_0 = -4.5`` °C, ``T_h = 2.5\times10^{-5}`` kg m⁻⁴ K⁻¹ and
``b_0 = 0.77`` kg m⁻³ (g kg⁻¹)⁻¹.

To use it in the form of the equation of state above, take the difference between the ambient
and the layer, ``\rho_a - \rho = \rho'(\Theta_a, S_{A,a}, Z) - \rho'(\Theta, S_A, Z)``. The quadratic
difference factorises, ``(\Theta_a-\Theta_0)^2 - (\Theta-\Theta_0)^2 = 2\,(\Theta_a-\Theta)(\bar\Theta-\Theta_0)``
with ``\bar\Theta = \tfrac12(\Theta_a+\Theta)``, so

```math
\rho_a - \rho = b_0\,(S_{A,a} - S_A) - \bigl[C_b\,(\bar\Theta - \Theta_0) + T_h Z\bigr](\Theta_a - \Theta).
```

Dividing by ``\rho_0`` (`Params.rho0_seawater`) gives the dimensionless form, with
``\beta = b_0/\rho_0``, ``a_0 = -C_b\Theta_0/\rho_0``, ``a_T = C_b/\rho_0`` and ``a_Z = T_h/\rho_0``,
which is the coefficient set `_eos_coefficients` uses for `RoquetEOS`.

The thermal expansion is then ``\alpha = \bigl[C_b(\bar\Theta - \Theta_0) + T_h Z\bigr]/\rho_0``,
or ``0.029`` kg m⁻³ K⁻¹ at the surface freezing point (about ``-1.9`` °C), ``0.041`` at 500 m, and
``0.068`` at 0.5 °C and 500 m, against the constant ``0.038`` of the linear default. So the buoyancy
contrast of a plume against its ambient is weaker where it is warm or deep (by about 5 % for a cold
plume and 25 % for a warm one at 500 m).

!!! warning "Experimental"
    `RoquetEOS` is experimental and logs a warning when constructed. [Roquet et al. (2015)](@cite roquet_defining_2015) fit it to the
    open ocean in Conservative Temperature and Absolute Salinity, which Laddie takes as its temperature
    and salinity (only salinity differences enter, so the offset to practical salinity cancels), and say
    it serves understanding rather than realistic simulation (TEOS-10 does).
    Supercooled cavity water is an extrapolation. `Params.alpha` and `Params.beta` are used by
    `LinearEOS` only, and an ambient profile built as neutral for them (as in the examples) is not neutral
    for another equation of state.

## API

```@docs
AbstractEquationOfState
LinearEOS
RoquetEOS
```
