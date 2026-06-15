
# ============================================================================
# Compact display — without these methods, printing a Model (or any of its
# sub-structs) at the REPL would dump every matrix it contains.
# The 2-arg `show` is the one-line form (also used inside other printing);
# `text/plain` adds detail for direct REPL display where useful.
# ============================================================================

Base.show(io::IO, v::Var{LX,LY,FT}) where {LX,LY,FT} = print(
    io,
    "Var{",
    nameof(LX),
    ", ",
    nameof(LY),
    "}(",
    FT,
    ", ",
    size(v.present, 1),
    "×",
    size(v.present, 2),
    ", levels: past/present/future)",
)

function Base.show(io::IO, s::State{FT}) where {FT}
    ny, nx = size(s.D.present)
    print(io, "State{", FT, "}: D, U, V, T, S — 3-level Vars of ", ny, "×", nx)
end

function Base.show(io::IO, c::Cache{FT}) where {FT}
    nmat = count(fn -> getfield(c, fn) isa AbstractMatrix, fieldnames(typeof(c)))
    ny, nx = size(c.melt)
    print(io, "Cache{", FT, "}: ", nmat, " scratch/diagnostic arrays of ", ny, "×", nx)
end

function Base.show(io::IO, s::IOState{FT}) where {FT}
    rd = isempty(s.rundir) ? "I/O disabled" : string("rundir = \"", s.rundir, "\"")
    print(io, "IOState{", FT, "}: t = ", s.t, ", ", rd)
end

# Generic one-liner for any forcing; relies only on the Tz/Sz/z profile fields
# that the model requires of every AbstractForcing.  extrema/length are
# reductions, so this is GPU-safe (no scalar indexing).
function _show_forcing(io::IO, f::AbstractForcing)
    zlo, zhi = extrema(f.z)
    Tlo, Thi = extrema(f.Tz)
    Slo, Shi = extrema(f.Sz)
    print(
        io,
        nameof(typeof(f)),
        "{",
        eltype(f.Tz),
        "}: ",
        length(f.z),
        "-point profile, z ∈ [",
        zlo,
        ", ",
        zhi,
        "] m, T ∈ [",
        round(Tlo; digits = 3),
        ", ",
        round(Thi; digits = 3),
        "] °C, S ∈ [",
        round(Slo; digits = 3),
        ", ",
        round(Shi; digits = 3),
        "] psu",
    )
end

Base.show(io::IO, f::AbstractForcing) = _show_forcing(io, f)
function Base.show(io::IO, f::ISOMIPForcing)
    _show_forcing(io, f)
    print(io, " — ISOMIP+ :", f.isomipcond)
end

function Base.show(io::IO, g::Grid{FT}) where {FT}
    print(
        io,
        "Grid{",
        FT,
        "}: ",
        g.Ny,
        "×",
        g.Nx,
        " cells (",
        g.Ny - 2,
        "×",
        g.Nx - 2,
        " interior), dx = ",
        g.dx,
        " m, dy = ",
        g.dy,
        " m",
    )
end

function Base.show(io::IO, ::MIME"text/plain", g::Grid)
    show(io, g)
    msk = g.mask
    print(
        io,
        "\n  cells: ",
        count(==(3), msk),
        " shelf, ",
        count(==(2), msk),
        " grounded, ",
        count(==(0), msk),
        " ocean, ",
        count(==(1), msk),
        " land/border",
    )
end

function Base.show(io::IO, p::Params{FT}) where {FT}
    print(
        io,
        "Params{",
        FT,
        "}(",
        nameof(typeof(p.entpar)),
        " + ",
        nameof(typeof(p.meltpar)),
        " + ",
        nameof(typeof(p.convpar)),
        " + ",
        nameof(typeof(p.openbc)),
        " + ",
        nameof(typeof(p.glbc)),
        " + ",
        nameof(typeof(p.tstep)),
        ")",
    )
end

Base.show(io::IO, ::FixedDt) = print(io, "FixedDt()")
Base.show(io::IO, ts::AdaptiveDt) = print(
    io,
    "AdaptiveDt(cfl_target = ",
    ts.cfl_target,
    ", q = ",
    ts.q,
    ", ncheck = ",
    ts.ncheck,
    ", dt ∈ [",
    ts.dtmin,
    ", ",
    ts.dtmax,
    "])",
)

function Base.show(io::IO, ::MIME"text/plain", p::Params{FT}) where {FT}
    println(io, "Params{", FT, "}:")
    scal = [
        (fn, getfield(p, fn)) for
        fn in fieldnames(typeof(p)) if getfield(p, fn) isa Number
    ]
    for chunk in Iterators.partition(scal, 4)
        println(
            io,
            "  ",
            join((rpad(string(k, " = ", v), 22) for (k, v) in chunk), " "),
        )
    end
    println(io, "  entrainment    = ", p.entpar)
    println(io, "  melt           = ", p.meltpar)
    println(io, "  convection     = ", p.convpar)
    println(io, "  open boundary  = ", p.openbc)
    println(io, "  grounding line = ", p.glbc)
    print(io, "  time stepper   = ", p.tstep)
end

_backend_name(m::Model) = nameof(typeof(KA.get_backend(getfield(m, :grid).zb)))

function Base.show(io::IO, m::Model{FT}) where {FT}
    g = getfield(m, :grid)
    print(
        io,
        "Model{",
        FT,
        "} on ",
        _backend_name(m),
        ": ",
        g.Ny - 2,
        "×",
        g.Nx - 2,
        " interior, ",
        nameof(typeof(getfield(m, :forcing))),
        " forcing",
    )
end

function Base.show(io::IO, ::MIME"text/plain", m::Model{FT}) where {FT}
    g = getfield(m, :grid)
    rc = getfield(m, :rc)
    p = getfield(m, :params)
    println(io, "Model{", FT, "} on ", _backend_name(m))
    println(
        io,
        "  grid:    ",
        g.Ny - 2,
        "×",
        g.Nx - 2,
        " interior cells, dx = ",
        g.dx,
        " m, dy = ",
        g.dy,
        " m, ",
        count(==(3), g.mask),
        " shelf cells",
    )
    println(io, "  forcing: ", getfield(m, :forcing))
    println(
        io,
        "  params:  dt0 = ",
        p.dt0,
        " s, ",
        nameof(typeof(p.entpar)),
        " + ",
        nameof(typeof(p.meltpar)),
        " + ",
        nameof(typeof(p.convpar)),
        " + ",
        nameof(typeof(p.openbc)),
        " + ",
        nameof(typeof(p.glbc)),
        " + ",
        nameof(typeof(p.tstep)),
    )
    if rc.saveday > 0
        print(
            io,
            "  output:  every ",
            rc.saveday,
            " d → ",
            joinpath(rc.resultdir, rc.name),
        )
    else
        print(io, "  output:  disabled (saveday = 0)")
    end
end
