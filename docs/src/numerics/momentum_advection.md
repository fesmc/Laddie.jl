# Momentum advection

Tracer and thickness advection is always upwind. **Momentum** advection, the momentum
carried through the faces of a velocity control volume, is a choice,
`Params(; momentum_advection = ...)`:

- [`CentredMomentumAdvection`](@ref), the default, evaluates the advected
  velocity as the average of the two either side of the face, as LADDIE v1 does. It adds
  no numerical viscosity of its own.
- [`UpstreamMomentumAdvection`](@ref) is flux-form donor cell, as [LADDIE v2](../versions.md):
  the momentum rides on the thickness equation's own mass fluxes, averaged onto
  the staggered faces, and is taken from the upstream side. Its implicit
  viscosity is of order ``|u|\,\Delta/2`` — ≈ 50 m² s⁻¹ at 1 km and
  0.1 m s⁻¹, far above a typical ``A_h`` — so the layer is less energetic.

v2's two schemes, `'upstream'` and `'fesom'`, differ only in how the face-normal velocity
is built from the staggered fields of its mesh, a question that does not arise on a
C-grid; `UpstreamMomentumAdvection` stands for both.

Building the fluxes from the mass fluxes has two consequences. Walls carry no momentum,
because they carry no mass, so [wall momentum advection](wall_advection.md) has no effect
under the upstream scheme; and an inflow face at the ice front carries the cell's own
thickness and velocity, matching [`ZeroGradientInflow`](@ref). The first one means that
the scheme and ``A_h`` have to be chosen together; see
[Wall momentum advection](wall_advection.md).

## API

```@docs
AbstractMomentumAdvection
CentredMomentumAdvection
UpstreamMomentumAdvection
```
