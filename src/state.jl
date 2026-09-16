
# ============================================================================
# State{FT, A} — five prognostic leapfrog variables.
# ============================================================================

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
