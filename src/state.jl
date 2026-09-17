"""
$(TYPEDEF)

The prognostic state of a [`Model`](@ref): layer thickness `D`, velocities `U`, `V`,
temperature `T` and salinity `S`, each a three-level leapfrog [`Laddie.Var`](@ref)
on its C-grid location.  Read through the model, e.g. `model.D.present`.
"""
mutable struct State{FT,A<:AbstractMatrix{FT}}
    D::Var{Center,Center,FT,A}
    U::Var{Face,Center,FT,A}
    V::Var{Center,Face,FT,A}
    T::Var{Center,Center,FT,A}
    S::Var{Center,Center,FT,A}
end

State(FT::Type, nx::Int, ny::Int) = State{FT,Matrix{FT}}(
    Var(Center, Center, FT, nx, ny),
    Var(Face, Center, FT, nx, ny),
    Var(Center, Face, FT, nx, ny),
    Var(Center, Center, FT, nx, ny),
    Var(Center, Center, FT, nx, ny),
)
