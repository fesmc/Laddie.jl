using JLD2
using NCDatasets
using Printf
using TOML


# ============================================================================
# RunConfig — static run + I/O configuration (all fields have defaults).
# ============================================================================

"""
$(TYPEDSIGNATURES)

Static configuration for a model run: duration, I/O cadence, output flags,
and restart options.  All fields have sensible defaults; a plain `RunConfig()`
disables file I/O (`saveday = 0`).

Set `saveday > 0` to enable NetCDF output at that interval (days).
"""
Base.@kwdef struct RunConfig
    name::String = "run"
    days::Float64 = 30.0
    saveday::Float64 = 0.0     # 0 = I/O disabled
    diagday::Float64 = 1.0
    restday::Float64 = 30.0
    resultdir::String = "./output/"
    logfilename::String = "log.txt"
    forcenewdir::Bool = true
    fromrestart::Bool = false
    restartfile::String = ""
    save_Ut::Bool = true
    save_Uu::Bool = false
    save_Vt::Bool = true
    save_Vv::Bool = false
    save_D::Bool = true
    save_T::Bool = true
    save_S::Bool = true
    save_melt::Bool = true
    save_entr::Bool = false
    save_ent2::Bool = false
    save_detr::Bool = false
    save_Tbase::Bool = false
    save_Tamb::Bool = false
    save_gammaT::Bool = false
    save_mask::Bool = true
    save_zb::Bool = true
end

# ============================================================================
# IOState{FT, A} — mutable runtime I/O state: counters, run directory, log,
# coordinate vectors, and time-average accumulators.
# A is the concrete matrix type (matches Grid/State/Cache).  Accumulators are
# allocated 0×0 at construction; `prepare_output!` replaces the enabled ones
# with full-size device arrays (the `save_*` flags in RunConfig guard access).
# x/y coordinate vectors stay on the CPU — they are only written to NetCDF.
# ============================================================================

mutable struct IOState{FT,A<:AbstractMatrix{FT}}
    # Time accounting.  `dt` is the runtime time step; because IOState is last
    # in the Model forwarding chain and Params holds `dt0` (not `dt`), `m.dt`
    # resolves here — so the same field tracks a step that may vary under
    # adaptive time stepping.  `t_sim` accumulates simulated seconds this run.
    t::Int        # completed steps this run (diagnostic / blow-up msg)
    dt::FT         # current time step (s) — resolves as `m.dt`
    t_sim::Float64    # accumulated simulated time this run (s)
    t_start::Float64    # restart offset (days), added by `_t_days`
    # Time-average accumulation window
    count::Int        # steps accumulated since the last output write
    t_accum::Float64    # simulated time accumulated since the last write (s)
    # Next-event times for periodic I/O (s since the start of this run)
    nextsave::Float64
    nextdiag::Float64
    nextrest::Float64
    # Run directory and log
    rundir::String
    logfile::String
    walltime_start::Float64
    # Interior cell-centre coordinates (m), CPU-resident
    x::Vector{FT}
    y::Vector{FT}
    # Time-average accumulators
    Utav::A;
    Uuav::A;
    Vtav::A;
    Vvav::A
    Dav::A;
    Tav::A;
    Sav::A;
    meltav::A
    entrav::A;
    ent2av::A;
    detrav::A
    Tbav::A;
    Taav::A;
    gamTav::A
end

function IOState(FT::Type, x::AbstractVector, y::AbstractVector)
    IOState{FT,Matrix{FT}}(
        0,
        FT(0),
        0.0,
        0.0,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        "",
        "",
        0.0,
        Vector{FT}(x),
        Vector{FT}(y),
        ntuple(_ -> Matrix{FT}(undef, 0, 0), 14)...,
    )
end

# Absolute simulation time in days: accumulated simulated time of this run
# offset by the restart time, so output/restart files of a continuation run
# never collide with the files of the run they restarted from.
_t_days(m) = m.t_start + m.t_sim / 86400.0

# ============================================================================
# Run directory + log
# ============================================================================

function _print2log(m, text)
    isempty(m.logfile) && return
    elapsed = time() - m.walltime_start
    h = floor(Int, elapsed / 3600)
    rem = elapsed - 3600h
    mn = floor(Int, rem / 60)
    sc = rem - 60mn
    open(m.logfile, "a") do f
        write(f, @sprintf("[%02d:%02d:%04.1f] %s\n", h, mn, sc, text))
    end
end

# One-line record of an adaptive-dt change (no-op when I/O is disabled).  Kept
# here because Printf is imported in this file; called by the dt controller.
_log_dt_change!(m, dt_old, dt_new, cfl) = _print2log(
    m,
    @sprintf(
        "%.3f days: dt %.1f → %.1f s (CFL %.2f)",
        _t_days(m),
        Float64(dt_old),
        Float64(dt_new),
        cfl
    )
)

"""
$(TYPEDSIGNATURES)

Create the output directory at `joinpath(m.resultdir, m.name)` and open the
log file.  Skips creating a new directory when `m.forcenewdir = false` and
the directory already exists (continuation run).
"""
function create_rundir!(m)
    rundir = joinpath(m.resultdir, m.name)
    if m.forcenewdir || !isdir(rundir)
        mkpath(rundir)
    end
    m.rundir = rundir
    m.logfile = joinpath(rundir, m.logfilename)
    m.walltime_start = time()
    _print2log(m, "Run directory: $(rundir)")
    return m
end

# ============================================================================
# Run provenance metadata
# ============================================================================

_toml_value(v::Bool) = v
_toml_value(v::Integer) = Int(v)
_toml_value(v::AbstractFloat) = Float64(v)
_toml_value(v::String) = v
_toml_value(v::Symbol) = String(v)
_toml_value(::Any) = nothing   # arrays etc. are skipped

# Type name + all TOML-representable fields of a struct.
function _scalar_fields(x)
    d = Dict{String,Any}("type" => string(nameof(typeof(x))))
    for fn in fieldnames(typeof(x))
        v = _toml_value(getfield(x, fn))
        v === nothing || (d[string(fn)] = v)
    end
    return d
end

# Write the effective configuration of this run — parameters, forcing, grid,
# precision, backend, package/Julia versions — so any output directory can be
# traced back to what produced it.  Never overwrites: a continuation run into
# the same directory gets run_metadata_1.toml, _2.toml, ...
function _write_run_metadata(m)
    p = getfield(m, :params)
    params_d = _scalar_fields(p)
    params_d["entrainment"] = _scalar_fields(p.entpar)
    params_d["melt"] = _scalar_fields(p.meltpar)
    params_d["convection"] = _scalar_fields(p.convpar)
    params_d["open_boundary"] = _scalar_fields(p.openbc)
    params_d["grounding_line"] = _scalar_fields(p.glbc)
    params_d["time_stepper"] = _scalar_fields(p.tstep)
    meta = Dict{String,Any}(
        "run" => Dict{String,Any}(
            "created" => Libc.strftime("%Y-%m-%dT%H:%M:%S", time()),
            "julia_version" => string(VERSION),
            "laddie_version" => string(pkgversion(@__MODULE__)),
            "backend" => string(nameof(typeof(KA.get_backend(m.tmask)))),
            "float_type" => string(m.FT),
            "t_start_days" => m.t_start,
        ),
        "grid" => Dict{String,Any}(
            "nx" => m.nx,
            "ny" => m.ny,
            "dx" => Float64(m.dx),
            "dy" => Float64(m.dy),
        ),
        "forcing" => _scalar_fields(getfield(m, :forcing)),
        "params" => params_d,
        "run_config" => _scalar_fields(getfield(m, :rc)),
    )
    path = joinpath(m.rundir, "run_metadata.toml")
    k = 1
    while isfile(path)
        path = joinpath(m.rundir, "run_metadata_$(k).toml")
        k += 1
    end
    open(io -> TOML.print(io, meta), path, "w")
    _print2log(m, "Wrote run metadata → $(basename(path))")
    return
end

# ============================================================================
# Output preparation
# ============================================================================

"""
$(TYPEDSIGNATURES)

Initialise time-average accumulators and save/diagnostic interval counters.
Must be called after prognostics are initialised (needs `m.ny`, `m.nx`) and
after `create_rundir!`.
"""
function prepare_output!(m)
    m.t = 0
    m.count = 0
    m.t_accum = 0.0

    # Allocate on same device and with same FT as the model arrays.
    # Full grid size (including halos) so _accum! can do bare .+= without
    # border-stripping; halos are masked out when writing to NetCDF.
    z = zero(m.tmask)
    m.save_Ut && (m.Utav = copy(z))
    m.save_Uu && (m.Uuav = copy(z))
    m.save_Vt && (m.Vtav = copy(z))
    m.save_Vv && (m.Vvav = copy(z))
    m.save_D && (m.Dav = copy(z))
    m.save_T && (m.Tav = copy(z))
    m.save_S && (m.Sav = copy(z))
    m.save_melt && (m.meltav = copy(z))
    m.save_entr && (m.entrav = copy(z))
    m.save_ent2 && (m.ent2av = copy(z))
    m.save_detr && (m.detrav = copy(z))
    m.save_Tbase && (m.Tbav = copy(z))
    m.save_Tamb && (m.Taav = copy(z))
    m.save_gammaT && (m.gamTav = copy(z))
    _write_run_metadata(m)
    return m
end

# ============================================================================
# Per-step accumulation helpers
# ============================================================================

_int(a) = Array(a)[2:(end-1), 2:(end-1)]

# t-grid velocity accumulation fused with the staggered average — avoids the
# two circshift allocations per step that im_half()/jm_half() would cost.  Accumulation
# is dt-weighted (× dt) so the time average is correct when dt varies; with a
# fixed dt this is the constant dt × the old step-weighted sum.
@kernel function _accum_ut_kernel!(av, @Const(U), Nx, dt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(av)
        half = FT(1) / (FT(1) + FT(1))
        w = _west(j, Nx)
        av[i, j] += (U[i, j] + U[i, w]) * half * dt
    end
end

@kernel function _accum_vt_kernel!(av, @Const(V), Ny, dt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(av)
        half = FT(1) / (FT(1) + FT(1))
        s = _south(i, Ny)
        av[i, j] += (V[i, j] + V[s, j]) * half * dt
    end
end

function _accum!(m)
    dt = m.dt
    m.count += 1
    m.t_accum += dt
    m.save_Ut &&
        launch!(_accum_ut_kernel!, m.Utav, m.Utav, m.U.present, size(m.Utav, 2), dt)
    m.save_Uu && (m.Uuav .+= m.U.present .* dt)
    m.save_Vt &&
        launch!(_accum_vt_kernel!, m.Vtav, m.Vtav, m.V.present, size(m.Vtav, 1), dt)
    m.save_Vv && (m.Vvav .+= m.V.present .* dt)
    m.save_D && (m.Dav .+= m.D.present .* dt)
    m.save_T && (m.Tav .+= m.T.present .* dt)
    m.save_S && (m.Sav .+= m.S.present .* dt)
    m.save_melt && (m.meltav .+= m.melt .* dt)
    m.save_entr && (m.entrav .+= m.entr .* dt)
    m.save_ent2 && (m.ent2av .+= m.ent2 .* dt)
    m.save_detr && (m.detrav .+= m.detr .* dt)
    m.save_Tbase && (m.Tbav .+= m.Tb .* dt)
    m.save_Tamb && (m.Taav .+= m.Ta .* dt)
    m.save_gammaT && (m.gamTav .+= m.gamT .* dt)
end

function _reset_accum!(m)
    m.count = 0
    m.t_accum = 0.0
    io = getfield(m, :io)
    for k in (
        :Utav,
        :Uuav,
        :Vtav,
        :Vvav,
        :Dav,
        :Tav,
        :Sav,
        :meltav,
        :entrav,
        :ent2av,
        :detrav,
        :Tbav,
        :Taav,
        :gamTav,
    )
        fill!(getfield(io, k), 0)   # disabled accumulators are 0×0 — no-op
    end
end

# ============================================================================
# Time-average output
# ============================================================================

function _write_output!(m, t_days)
    # dt-weighted accumulators divided by accumulated time → time average
    # (equals the old ÷ count when dt is constant).
    n = m.t_accum
    tmask_int = _int(m.tmask)
    filename = joinpath(m.rundir, @sprintf("output_%06.0f.nc", t_days))

    NCDataset(filename, "c") do ds
        defDim(ds, "y", m.ny);
        defDim(ds, "x", m.nx)
        defVar(ds, "x", Float64, ("x",))[:] = m.x
        defVar(ds, "y", Float64, ("y",))[:] = m.y
        ds.attrib["time_start_days"] = max(m.t_start, t_days - m.saveday)
        ds.attrib["time_end_days"] = t_days
        ds.attrib["model"] = "Laddie.jl"

        function wv(name, av, scale, units, longname)
            v = defVar(
                ds,
                name,
                Float64,
                ("y", "x");
                fillvalue = NaN,
                attrib = ["units" => units, "long_name" => longname],
            )
            av_int = _int(av)   # one Array() transfer + halo strip per field, at write time
            v[:, :] = ifelse.(tmask_int .> 0, av_int ./ n .* scale, NaN)
        end

        m.save_Ut && wv("Ut", m.Utav, 1.0, "m s-1", "x-velocity on t-grid")
        m.save_Uu && wv("Uu", m.Uuav, 1.0, "m s-1", "x-velocity on u-grid")
        m.save_Vt && wv("Vt", m.Vtav, 1.0, "m s-1", "y-velocity on t-grid")
        m.save_Vv && wv("Vv", m.Vvav, 1.0, "m s-1", "y-velocity on v-grid")
        m.save_D && wv("D", m.Dav, 1.0, "m", "mixed-layer thickness")
        m.save_T && wv("T", m.Tav, 1.0, "degC", "layer-averaged temperature")
        m.save_S && wv("S", m.Sav, 1.0, "psu", "layer-averaged salinity")
        m.save_melt && wv("melt", m.meltav, spy, "m yr-1", "basal melt rate")
        m.save_entr && wv("entr", m.entrav, spy, "m yr-1", "entrainment rate")
        m.save_ent2 && wv("ent2", m.ent2av, spy, "m yr-1", "additional entrainment")
        m.save_detr && wv("detr", m.detrav, spy, "m yr-1", "detrainment rate")
        m.save_Tbase && wv("Tbase", m.Tbav, 1.0, "degC", "temperature at ice base")
        m.save_Tamb &&
            wv("Tamb", m.Taav, 1.0, "degC", "ambient temperature at layer base")
        m.save_gammaT &&
            wv("gammaT", m.gamTav, 1.0, "m s-1", "turbulent heat exchange velocity")

        if m.save_mask
            defVar(ds, "mask", Int32, ("y", "x"))[:, :] = Int32.(_int(m.mask))
        end
        if m.save_zb
            v = defVar(ds, "zb", Float64, ("y", "x"); attrib = ["units" => "m"])
            v[:, :] = _int(m.zb)
        end
    end
    _print2log(m, @sprintf("%.3f days: saved output → %s", t_days, basename(filename)))
end

# A periodic event is due once accumulated time reaches the next event time.
# The half-step tolerance mirrors the run! stopping rule (round-half-up), so a
# fixed dt fires at the same steps the old `t % interval == 0` test did.
_event_due(m, next) = m.t_sim + m.dt / 2 >= next

"""
$(TYPEDSIGNATURES)

Accumulate model fields into time averages and write a NetCDF output file
at every `m.saveday`-day interval.  Called once per time step inside `run!`;
the final partial window is flushed by `run!` after the loop.
"""
function savefields!(m)
    _accum!(m)
    if _event_due(m, m.nextsave)
        _write_output!(m, _t_days(m))
        _reset_accum!(m)
        m.nextsave += m.saveday * 86400.0
    end
end

# Flush any unwritten accumulation as a final output file (end of run).
function flush_output!(m)
    m.count > 0 || return
    _write_output!(m, _t_days(m))
    _reset_accum!(m)
end

# ============================================================================
# Restart I/O
# ============================================================================

function _write_restart!(m, t_days)
    filename = joinpath(m.rundir, @sprintf("restart_%06.0f.jld2", t_days))

    _v(var) = (
        past = Array(var.past),
        present = Array(var.present),
        future = Array(var.future),
    )
    # Save the current dt so an adaptive run resumes at the step it left off
    # (the saved leapfrog levels are separated by this dt); FixedDt saves dt0.
    jldsave(
        filename;
        t_days,
        dt = Float64(m.dt),
        D = _v(m.D),
        U = _v(m.U),
        V = _v(m.V),
        T = _v(m.T),
        S = _v(m.S),
    )

    cp(filename, joinpath(m.rundir, "restart_latest.jld2"); force = true)
    _print2log(m, @sprintf("%.3f days: saved restart → %s", t_days, basename(filename)))
end

"""
$(TYPEDSIGNATURES)

Write a JLD2 restart file containing all three leapfrog levels of D, U, V,
T, S at every `m.restday`-day interval.  Called once per time step inside
`run!`; the final state is written by `run!` after the loop.  Also writes
`restart_latest.jld2`.  Arrays are moved to CPU before saving so the file is
backend-agnostic; the native `FT` precision is preserved.
"""
function saverestart!(m)
    _event_due(m, m.nextrest) || return
    _write_restart!(m, _t_days(m))
    m.nextrest += m.restday * 86400.0
end

"""
$(TYPEDSIGNATURES)

Load D, U, V, T, S from the JLD2 restart file at `m.restartfile` into all
three leapfrog levels of the existing `Var` structs, then call
`update_secondary_fields!` and one bootstrap integration step.

Geometry (masks, zb) must already be initialised before calling this.
"""
function init_from_restart!(m)
    jldopen(m.restartfile, "r") do f
        m.t_start = f["t_days"]
        # Resume at the saved dt when present (adaptive runs); older files
        # without it keep the dt set at build (dt0).
        haskey(f, "dt") && (m.dt = m.FT(f["dt"]))
        for (name, var) in (("D", m.D), ("U", m.U), ("V", m.V), ("T", m.T), ("S", m.S))
            data = f[name]
            var.past .= data.past
            var.present .= data.present
            var.future .= data.future
        end
    end
    update_secondary_fields!(m)
    leapfrog_step!(m, 1)
    _print2log(m, "Restarted from $(m.restartfile) at $(m.t_start) days")
    return m
end

# ============================================================================
# Diagnostics
# ============================================================================

"""
$(TYPEDSIGNATURES)

Write a one-line diagnostic to the log file at every `m.diagday`-day interval.
"""
function printdiags(m)
    _event_due(m, m.nextdiag) || return
    m.nextdiag += m.diagday * 86400.0
    t_days = _t_days(m)

    tmask = Array(m.tmask)
    D = Array(m.D.present)
    melt = Array(m.melt)
    entr = Array(m.entr)
    ent2 = Array(m.ent2)
    detr = Array(m.detr)
    convD = Array(m.convD)
    drho = Array(m.drho)
    conv = Array(m.convection)
    U = Array(m.U.present)
    V = Array(m.V.present)

    dxdy = m.dx * m.dy
    area = sum(tmask) * dxdy

    d_Dav = sum(D .* tmask) * dxdy / area
    icecells = findall(tmask .> 0)
    d_Dmin = minimum(D[icecells])
    d_Dmax = maximum(D[icecells])

    d_Mmax = spy * maximum(melt)
    d_Mav = spy * sum(melt .* tmask) * dxdy / area

    total = sum((melt .+ entr .+ ent2 .- detr) .* tmask) * dxdy
    d_MWF = total > 0 ? 100.0 * sum(melt .* tmask) * dxdy / total : 0.0

    d_Etot = 1e-6 * sum(entr .* tmask) * dxdy
    d_E2tot = 1e-6 * sum(ent2 .* tmask) * dxdy
    d_DEtot = 1e-6 * sum(detr .* tmask) * dxdy
    d_PSI = -1e-6 * sum(convD .* tmask) * dxdy

    # max t-grid speed without the im/jm circshift allocations
    ny, nx = size(U)
    d_Vmax = 0.0
    for j = 1:nx, i = 1:ny
        tmask[i, j] > 0 || continue
        w = j == 1 ? nx : j - 1
        s = i == 1 ? ny : i - 1
        u_t = (U[i, j] + U[i, w]) / 2
        v_t = (V[i, j] + V[s, j]) / 2
        spd = sqrt(u_t^2 + v_t^2)
        spd > d_Vmax && (d_Vmax = spd)
    end

    d_drho = 1000.0 * minimum(ifelse.(tmask .> 0, drho, 100.0))
    d_conv = sum(conv .* (tmask .> 0))

    line = @sprintf(
        "%8.3f days || %5.1f [%4.2f %4.0f] m || %5.2f | %3.0f m/yr || %5.2f %% || %5.3f + %5.3f - %5.3f | %5.3f Sv || %3.2f m/s || %5.5f %3.0f []",
        t_days,
        d_Dav,
        d_Dmin,
        d_Dmax,
        d_Mav,
        d_Mmax,
        d_MWF,
        d_Etot,
        d_E2tot,
        d_DEtot,
        d_PSI,
        d_Vmax,
        d_drho,
        d_conv
    )
    _print2log(m, line)
end
