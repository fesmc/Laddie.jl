"""
$(TYPEDEF)

The prognostic state of a [`Model`](@ref): layer thickness `D`, velocities `U`, `V`,
temperature `T` and salinity `S`, each a three-level leapfrog [`Laddie.Var`](@ref)
on its C-grid location.  Read through the model, e.g. `model.D.present`.

# Fields
$(TYPEDFIELDS)
"""
mutable struct State{
    D<:Var{Center,Center},
    U<:Var{Face,Center},
    V<:Var{Center,Face},
    T<:Var{Center,Center},
    S<:Var{Center,Center},
}
    "layer thickness (m), on T-points"
    D::D
    "depth-averaged velocity along x (m s⁻¹), on u-points"
    U::U
    "depth-averaged velocity along y (m s⁻¹), on v-points"
    V::V
    "layer temperature (°C), on T-points"
    T::T
    "layer salinity (psu), on T-points"
    S::S
end

State(FT::Type, nx::Int, ny::Int) = State(
    Var(Center, Center, FT, nx, ny),
    Var(Face, Center, FT, nx, ny),
    Var(Center, Face, FT, nx, ny),
    Var(Center, Center, FT, nx, ny),
    Var(Center, Center, FT, nx, ny),
)
