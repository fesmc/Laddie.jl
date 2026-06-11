# Numerics

## Spatial discretisation

LADDIE uses an Arakawa **C-grid**: scalar fields (``D``, ``T``, ``S``) and all
diagnostic quantities live at cell centres, while the velocity components ``U``
and ``V`` are located at the east and north cell faces respectively. Staggered
interpolation operators move quantities between the two locations as needed by
the pressure-gradient and Coriolis terms. Boundary conditions are encoded in
floating-point mask arrays (`tmask`, `umask`, `vmask`) rather than in branching
logic, so the same kernels execute unchanged on CPU and GPU.

## Spatial operators

Advective fluxes of volume, heat, and salt are computed with an
**upstream-biased** scheme (`convT`, `convU`, `convV`); viscous and diffusive
fluxes use a five-point **Laplacian** (`lapT`, `lapU`, `lapV`). Both families
are nearest-neighbour stencils built from `circshift`-style shifts, keeping the
mask-based boundary treatment uniform across all operators.

## Time integration

The prognostic variables advance with a **leapfrog** scheme. Each variable
carries three time levels — past, present, and future — updated each step as

```math
q^{n+1} = q^{n-1} + 2\,\Delta t\; \mathcal{F}(q^n),
```

where ``\mathcal{F}`` collects all right-hand-side terms evaluated at the present
level. After the leapfrog advance, a **Robert–Asselin filter** (strength ``\nu``)
damps the computational (leapfrog) mode:

```math
q^n \;\leftarrow\; q^n + \frac{\nu}{2}\bigl(q^{n-1} + q^{n+1} - 2\,q^n\bigr).
```

The three levels are then cycled (past ← present, present ← filtered future)
before the next step.

## Scheme compatibility

The spatial and temporal discretisation choices are not independent — they must
be consistent or the scheme is unconditionally unstable.

Leapfrog is a *neutral* scheme: it neither adds nor removes energy from the
resolved scales. A purely upwind advection scheme, by contrast, is *dissipative*
— it carries an implicit diffusion proportional to the grid spacing. Combining
leapfrog (neutral) with an upwind (dissipative) operator creates a situation
where the computational mode introduced by leapfrog is *amplified* rather than
damped: the scheme is **unconditionally unstable** regardless of the time step.
(This was confirmed during the `CGridProto` advection tests, where leapfrog
paired with upwind advection grew to ``10^{53}`` within a few hundred steps.)

LADDIE avoids this by using a **centred** (non-dissipative) advection stencil, so
leapfrog's computational mode has no diffusive source to feed on. The residual
mode is then damped by two mechanisms that are therefore **load-bearing for
stability**, not optional niceties:

- The **Robert–Asselin filter** directly suppresses the ``2\Delta t`` oscillation.
- The explicit **Laplacian viscosity** (``A_h``) and **diffusivity** (``K_h``)
  provide the small amount of physical dissipation that removes energy from the
  grid scale.

Removing either the filter or the Laplacian terms causes the scheme to eventually
diverge.

## Domain, boundaries, and masks

The domain is padded with a **one-cell grounded border** on all sides. This lets
all stencil operators use **periodic wrap** throughout — there are no special edge
cases — while the border cells are masked out and contribute nothing to the
dynamics. The physical boundaries are:

- **Grounding line**: zero-flux walls with a partial-slip coefficient `slip`.
- **Ice front**: configurable open boundary, either zero-gradient inflow
  (`ZeroGradientInflow`) or outflow-only (`NoInflow`).

## Stability safeguards

Two additional limiters guard against numerical blow-up in extreme conditions.
A **minimum layer thickness** ``D_\min`` is enforced through extra entrainment: if
a cell thins below the threshold, the entrainment source is increased to restore
it. A **velocity cutoff** ``v_\text{cut}`` clips ``|U|`` and ``|V|``, preventing
the leapfrog step from generating unbounded momentum near the grounding line where
``D \to 0``.

## Vertical forcing lookup

The ambient profiles ``T_a(z)`` and ``S_a(z)`` are stored on a uniform 1 m depth
grid. At each time step the local plume-base depth ``z_b - D`` indexes this grid
and linear interpolation supplies the ``T_a``, ``S_a`` values that enter the
entrainment and melt parameterisations.
