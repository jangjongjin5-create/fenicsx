

# ==================================================================
# FEML / PSE Lab -- PEM Fuel Cell Cathode, M^2 Two-Phase Mixture Model
# Reproducing Abdollahzadeh et al., Energy 68 (2014) 478-494, Table 1,
# cross-checked and corrected against Chang, Chen & Teng, J. Power
# Sources 160 (2006) 268-276 (same M^2 family, fully self-consistent
# closed forms for gamma, jl, Leverett J(s), Bruggeman D_eff).
# DOLFINx 0.11  --  shared physics module (imported from %%fenicsx cells)
# ==================================================================
#
# Governing equations solved (2D cathode GDL cross-section, membrane
# at x=0 -> channel land at x=Lx, interdigitated inlet/rib/outlet on
# the channel side, symmetry top/bottom):
#   Mass + Darcy momentum   -> mixture pressure p         (mass cons., Darcy limit)
#   Water mixture mass fraction C_w  -> transport           (Chang Eq. 2/4, alpha=H2O)
#   Gas O2 mass fraction Cg_O2       -> transport            (Chang Eq. 2/4, alpha=O2)
#   Liquid saturation s = s(C_w)      -> algebraic equilibrium
#   Local current density I(y)        -> Tafel kinetics at the catalyst layer
#   Average current density            -> integration along the catalyst layer
#
# --------------------------------------------------------------------
# NUMERICS -- Newton/SNES on a monolithic (p, C_w, Cg_O2) system
# --------------------------------------------------------------------
# Earlier versions of this notebook used segregated Picard iteration:
# solve pressure, then water, then oxygen, each with frozen coefficients
# from the previous iterate; saturation s was tracked as its own field,
# updated from C_w and fed back (lagged) into the other three equations'
# coefficients. Six different remedies -- fixed relaxation, adaptive
# (bold-driver) relaxation on s alone, the same extended to all three
# fields with a proper atol+rtol convergence check, smoothing the s(C_w)
# kink, adaptive eta step-halving continuation, and finally a loosened
# ~1% convergence tolerance -- were tried in sequence and ALL failed
# once saturation became significant (roughly eta >~ 0.33-0.35 V):
# Picard would run to its iteration cap without converging, at every
# eta step in that range, regardless of step size (bisecting the eta
# increment down to ~1/32 of its original size didn't help either,
# ruling out "too large a continuation step" as the cause) or how loose
# the tolerance was made. That pattern -- consistent failure exactly
# where s becomes significant, independent of damping, smoothness, step
# size, or tolerance -- pointed at the SEGREGATED lag between s and C_w
# itself as the structural problem, not any particular numerical
# remedy's tuning.
#
# Fix: eliminate that lag entirely. s is smooth in C_w already (see
# s_smooth_eps below), so it can be substituted as a differentiable UFL
# EXPRESSION of C_w directly everywhere it's used (mixture density,
# viscosity, Darcy coefficient, capillary diffusion, Bruggeman
# diffusivities, the Tafel BC's (1-s) factor) rather than tracked as a
# separate lagged field. grad(s) then also comes for free via UFL's
# chain rule -- no separate L2 projection step needed. Pressure, water,
# and oxygen are combined into ONE mixed function space and solved
# together as a single nonlinear residual via dolfinx's Newton solver
# (PETSc SNES under the hood), which uses the full analytic Jacobian
# (via UFL automatic differentiation) and genuine quadratic convergence
# near a solution, rather than fixed-point iteration with hand-tuned
# damping. eta is a dolfinx Constant updated between solves (the
# symbolic form is built once, not rebuilt per eta), with the same
# continuation-with-bisection-fallback strategy as before for
# robustness across the flooding-onset transition.
#
# CORRECTIONS applied earlier, after cross-checking Chang et al. (2006),
# which gives fully closed, self-consistent forms the first
# (Abdollahzadeh-only) version of this notebook had to guess or got
# wrong (still in effect; only the SOLUTION METHOD changed above, not
# the physics):
#   * Leverett/capillary-pressure scaling is sqrt(eps/K), not sqrt(K*eps)
#     -- confirmed identically by both papers (Eq. 10 / Eq. 17).
#   * The capillary flux jl (Eq. 14/16/20 in Chang) carries NO extra
#     rho_l prefactor.
#   * The advection correction factor gamma^alpha (Eq. 3/9) is *defined*
#     so that gamma^alpha * rho*C^alpha = rho*(lambda_l*Cl^alpha +
#     lambda_g*Cg^alpha) identically (Eq. 24, 30), with Cl^H2O=1,
#     Cg^H2O=min(C_w, Cg_sat) (smoothed), Cl^O2=0.
#   * Species diffusion is weighted by the GAS density rho_g (Chang
#     Eq. 29), not the mixture density.
#   * D_eff uses the Bruggeman tortuosity exponent tau (Chang Eq. 33:
#     D_eff=[eps(1-s)]^tau * D_g); tau=1.5 (literature default, not
#     given numerically in either excerpt).
#   * Porosity eps multiplies only the *diffusive* terms (Chang Eq. 1-2),
#     never the convective flux divergence; standard Darcy superficial
#     velocity u=-(K/mu)grad(p), no extra eps.
#   * Membrane pressure Neumann BC: kappa*dp/dn = N_w - N_O2 (a
#     difference, O2 and H2O flux point in opposite directions through
#     the membrane plane; cross-checked two independent ways).
#   * Species membrane flux signs: water +N_w, oxygen -N_O2 (re-derived
#     via D*dC/dn = -(physical_flux . n_hat), the sign Fick's law
#     introduces that a naive "flux along the outward normal" shortcut
#     misses).
#   * Psat_atm's correlation (Table 2) needs T in CELSIUS, not Kelvin.
#
# SUPG STABILIZATION -- CURRENTLY DISABLED IN THIS (NEWTON) MODULE.
# Was added after comparing Fig. 8's single-phase vs. two-phase local
# current density profiles: the two-phase curve showed severe
# node-to-node oscillation along the channel that the single-phase one
# never did. Root cause diagnosis: the Bruggeman-shrunk diffusivity
# D_eff=[eps(1-s)]^tau*D at high s (e.g. ~1% of its unshrunk value at
# s=0.9) combined with the membrane's advective flux pushes the local
# Peclet number well past where unstabilized P1 Galerkin is known to
# wiggle. Standard SUPG was added to both the water and oxygen
# equations (tau=1/sqrt((2|a|/h)^2+(4D/h^2)^2), h=cell diameter,
# residual=non-conservative a.grad(u)) and initially seemed to cause
# FFCx JIT compilation to hang for the full 3-field coupled Newton
# system -- but this was never isolated against a no-SUPG baseline, so
# it's not confirmed SUPG was actually the cause rather than the base
# coupling already being this expensive to compile. Removed here to
# get a working baseline; if that baseline itself compiles/runs fine,
# SUPG (or a cheaper alternative) should be re-added and re-tested in
# isolation. The Picard companion notebook still has SUPG (its
# per-equation LinearProblem forms are far simpler to compile than one
# monolithic 3-field nonlinear residual+Jacobian) and isn't showing
# this compile-time problem, so it remains the more reliable source
# for two-phase-regime figures in the meantime.
#
# TEMPERATURE-DEPENDENT EXCHANGE CURRENT DENSITY I0(T) (added after
# checking the Fig. 10-12 temperature sweep against the actual paper
# text, which the notebook now has full access to): I0 was held
# constant at 100 A/m^2 across the whole T sweep, giving I *decreasing*
# monotonically with T at fixed eta -- opposite to the paper. Table 2
# lists I0=100 A/m^2 explicitly "(at 333.0 K)", and Sec. 4.3 states
# plainly: "Increase in operating temperature leads to higher fuel cell
# performance due to increase in rates of oxygen reduction reaction
# through enhancement of kinetics of electrochemical reaction (increase
# in Reference current density)" -- i.e. I0 itself is meant to rise with
# T, not stay fixed; held constant, T only enters through the Tafel
# exponential's own 1/T in the denominator, which decreases with T.
# Fixed with a standard Arrhenius correction, I0(T) = I0_ref *
# exp[-(Ea/R)(1/T - 1/T_ref)], anchored at T_ref=333K per Table 2. The
# paper doesn't give its own Ea numerically (an under-specification, like
# GDL thickness or Bruggeman tau elsewhere in this file), so Ea=72.4
# kJ/mol is used -- a standard, widely-cited literature value for Pt ORR
# activation energy in PEMFC modeling. At eta=0.30 V this comfortably
# flips the net trend the right way (Ea=72.4 kJ/mol dominates the Tafel
# exponential's own alpha_c*F*eta~14.5 kJ/mol "reverse" contribution at
# that eta), consistent with the paper's stated direction, though the
# exact quantitative match isn't guaranteed since the true Ea is unknown.
#
# See the "Modeling assumptions" markdown cell in the notebook for what
# is still assumed rather than confirmed (GDL thickness, U_oc/R_contact
# for a full U-I curve, Bruggeman tau's numeric value).
# ==================================================================

import numpy as np
import basix.ufl
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
import ufl
import os
import time

PROGRESS_LOG = "/content/pemfc_m2_progress.log"


def _progress_log(line):
    """Append a timestamped line to PROGRESS_LOG, fsync'd so it's visible
    to a *separate* cell/process reading the file while this one is still
    running. %%fenicsx appears to buffer a cell's stdout until the whole
    cell finishes (or errors) rather than streaming it -- print(...,
    flush=True) flushes Python's own buffer but not whatever the magic
    itself is doing upstream of that -- so this is the only way to get
    real-time visibility into a long-running run_case() call from another
    cell without waiting for the first one to finish."""
    try:
        with open(PROGRESS_LOG, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {line}\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass  # never let logging itself break the actual computation


def _classify_branch_state(eta_val, s_max_val, deta_sign):
    """Heuristic, human-readable label for roughly where a point sits on
    the traced branch -- NOT a rigorous branch identifier (there's no way
    to do that without genuine bifurcation-theoretic bookkeeping across
    the whole history), just enough to make checkpoints and snapshots
    recognizable at a glance (e.g. when deciding whether a resumed run
    picked up where you expect, or browsing saved snapshots for an
    interesting one) without manually cross-checking eta/s_max/direction
    every time."""
    if s_max_val < 0.05:
        phase = "near-single-phase"
    elif s_max_val < 0.3:
        phase = "light flooding"
    elif s_max_val < 0.6:
        phase = "moderate flooding"
    elif s_max_val < 0.9:
        phase = "severe flooding"
    else:
        phase = "near-saturated (s~cap)"
    direction = "eta increasing" if deta_sign > 0 else "eta decreasing"
    return f"{phase}, {direction} (eta={eta_val:.4f}, s_max={s_max_val:.3f})"


def _jacobian_singular_values(problem, n_smallest=6, n_largest=3):
    """Extract the SNES's current Jacobian (the state after its most
    recent solve() call, converged or not) and return its singular value
    spectrum, both raw and after Jacobi (diagonal) equilibration.

    Direct diagnostic for whether a hard region is a genuine fold
    bifurcation: the defining mathematical signature of a fold (turning
    point) is that the Jacobian of the governing system becomes singular
    there -- if the smallest singular value collapses toward zero (while
    the largest stays bounded, i.e. the condition number blows up
    specifically as eta approaches the problem region) that's a fold, not
    merely a hard but regular nonlinearity.

    The RAW spectrum from this monolithic (p, C_w, Cg_O2) system is not
    directly usable for that, though: pressure (~1e5 Pa) and the mass
    fractions (~0.01-1) differ by 5-7 orders of magnitude in natural
    scale, and Dirichlet BC rows are set to exact identity -- both
    contribute a large, roughly CONSTANT (eta-independent) baseline
    condition number that has nothing to do with proximity to a fold and
    would swamp any genuine fold signature if not removed. Jacobi
    (diagonal) equilibration -- rescaling each row/column by
    1/sqrt(|diagonal entry|), a standard, cheap technique -- removes most
    of that scale artifact; the SCALED smallest singular value's trend
    across eta is what actually reflects the underlying system's
    conditioning.

    The system here (~4500x4500 for the default mesh) is small enough
    that a one-off dense SVD (a few seconds) is a reasonable diagnostic
    cost, even though it would be far too expensive to run on every
    iteration of a production sweep."""
    import scipy.sparse as sp
    J_mat = problem.solver.getJacobian()[0]  # (J, P[, callback]) depending on petsc4py version
    n = J_mat.getSize()[0]
    indptr, indices, data = J_mat.getValuesCSR()
    J_dense = sp.csr_matrix((data, indices, indptr), shape=(n, n)).toarray()

    diag = np.abs(np.diag(J_dense))
    scale = 1.0 / np.sqrt(np.maximum(diag, 1e-30))
    J_scaled = (J_dense * scale[None, :]) * scale[:, None]

    svals_raw = np.sort(np.linalg.svd(J_dense, compute_uv=False))
    svals_scaled = np.sort(np.linalg.svd(J_scaled, compute_uv=False))
    return (svals_raw[:n_smallest], svals_raw[-n_largest:],
            svals_scaled[:n_smallest], svals_scaled[-n_largest:])
    return svals[:n_smallest], svals[-n_largest:]

DEFAULT_PARAMS = dict(
    # --- Table 2 (Abdollahzadeh base validation case, T = 330 K) ---
    F=96485.0, R=8.314,
    K_perm=1.2e-12, eps_p=0.30, p_in=1.007e5, p_out=1.0e5,
    mu_g=2.03e-5, D_O2_g=1.775e-5, D_H2O_g=2.56e-5,
    I0=100.0, alpha_c=0.5, alpha_w=0.5, rho_l=1000.0,
    T_ref_I0=330.0,      # Anchored at the actual base-case/validation
                          # operating temperature (Table 2: "Operating
                          # temperature T K 330"), NOT the "333.0 K" label
                          # on the I0 row -- that's most likely just citing
                          # the source measurement's condition, not a
                          # correction meant to apply within the 330K base
                          # case itself. Anchoring here keeps I0=100 exactly
                          # at the validated default (leaving Fig. 4's
                          # calibration and every other already-checked
                          # figure undisturbed) while still giving the
                          # correct increasing trend once T_cell is swept
                          # away from 330K in the Fig. 10-12 study.
    Ea_orr=72400.0,      # J/mol -- ORR activation energy on Pt (72.4 kJ/mol,
                          # a standard literature value; not given explicitly
                          # in the excerpted paper text, but the paper's own
                          # Sec. 4.3 explicitly says higher T raises I0
                          # ("increase in Reference current density") --
                          # holding I0 constant across a T sweep, as earlier
                          # versions of this notebook did, produces the
                          # opposite (decreasing) trend, since T then only
                          # enters through the Tafel exponential's own
                          # denominator
    M_H2O=18.0e-3, M_O2=0.032, M_N2=0.028, mu_l=3.15e-4,
    RH_in=0.0, T_cell=330.0, C_O2_ref=40.0,
    theta_c_deg=91.0, sigma_st=0.0625,
    X_O2_in=0.21, X_N2_in=0.79,
    # --- Table 3 (interdigitated geometry) --------------------------
    w_ch=0.8e-3, w_rb=1.2e-3, Lx=0.30e-3,
    # --- closures confirmed / added after Chang et al. (2006) -------
    n_corey=3.0,        # Eq. 12-13 -- confirmed identical in both papers
    tau_brug=1.5,       # Eq. 33 Bruggeman exponent -- literature default,
                         # not given numerically in either excerpt
    s_max=0.99,
    s_smooth_eps=0.02,  # smoothing width (s-units) for the two-phase onset;
                         # also what lets s be substituted as a differentiable
                         # UFL expression of C_w for the Newton solve
    single_phase=False,   # if True, s is forced to 0 everywhere (reproduces
                            # the paper's "single-phase flow" comparison curves)
    # --- numerics -----------------------------------------------------
    nx=24, ny=60,
    newton_rtol=1e-7, newton_atol=1e-9, newton_max_it=50,
)


def Psat_atm(T):
    """Saturation pressure of water, atm (Table 2 correlation). T in CELSIUS."""
    return 10.0 ** (-2.1794 + 0.02953 * T - 9.1837e-5 * T**2 + 1.4454e-7 * T**3)


def _smooth_outflow(x, eps=1e-5):
    """Smooth (regularized) approximation to max(x, 0), used for clipping
    the outlet boundary's advective flux to its outflow-only part (see
    the outlet-BC stabilization discussion at each F_water/F_oxy
    definition). A HARD ufl.max_value(x, 0.0) was tried first and works
    fine for warm-started continuation (small Newton steps near an
    already-good solution rarely cross the kink), but was confirmed to
    cause SNES divergence for a COLD START from a naive/inlet-like
    initial guess (large early Newton steps repeatedly hit the
    non-smooth kink) -- exactly the situation at the very start of every
    continuation run (eta_start, no checkpoint yet). This smooth version
    removes that kink while remaining numerically equivalent to the hard
    clip away from x=0 (to within eps)."""
    return 0.5 * (x + ufl.sqrt(x**2 + eps**2))


def J_leverett(s, hydrophobic=True):
    """Leverett J-function (reference only -- the Newton solve builds dJds
    directly as a flat UFL expression inline inside run_case). hydrophobic branch
    (theta_c>90 deg, Chang Eq.19 / Abdollahzadeh Eq.10):
    J(s)=1.417s-2.120s^2+1.263s^3. hydrophilic branch (theta_c<90 deg,
    Chang Eq. 18): J(s)=1.417(1-s)-2.120(1-s)^2+1.263(1-s)^3."""
    if hydrophobic:
        return 1.417 * s - 2.120 * s**2 + 1.263 * s**3
    u = 1.0 - s
    return 1.417 * u - 2.120 * u**2 + 1.263 * u**3


def dJ_leverett(s, hydrophobic=True):
    if hydrophobic:
        return 1.417 - 4.240 * s + 3.789 * s**2
    u = 1.0 - s
    return -(1.417 - 4.240 * u + 3.789 * u**2)


def extract_line(coords, arrays, x_target=0.0, tol=1e-7):
    """Pull out a profile along the vertical line x=x_target (default: the
    membrane), sorted by y. `arrays` is a dict of {name: array-over-all-dofs};
    returns a dict with 'y' plus the same keys, restricted/sorted to that line.
    Pure NumPy -- works equally in a %%fenicsx cell or a later plotting cell."""
    x = coords[:, 0]
    mask = np.abs(x - x_target) < tol
    y_sel = coords[mask, 1]
    order = np.argsort(y_sel)
    out = {"y": y_sel[order]}
    for name, arr in arrays.items():
        out[name] = np.asarray(arr)[mask][order]
    return out


def extract_interface(coords, s_arr, threshold=0.02):
    """For each distinct y (along-channel position), find the x (GDL-
    thickness position, membrane=0) where saturation first drops below
    `threshold` moving away from the membrane -- i.e. the boundary between
    the two-phase and single-phase zones (paper Figs. 12/17/20/25). Returns
    (y_values, x_interface). x_interface=0 means no two-phase zone at that y;
    x_interface=Lx means two-phase all the way to the channel."""
    x, y = coords[:, 0], coords[:, 1]
    y_unique = np.unique(np.round(y, 12))
    xi = np.zeros_like(y_unique)
    for i, yv in enumerate(y_unique):
        row = np.abs(y - yv) < 1e-9
        xr, sr = x[row], s_arr[row]
        order = np.argsort(xr)
        xr, sr = xr[order], sr[order]
        below = sr < threshold
        if below[0]:
            xi[i] = 0.0
        elif not np.any(below):
            xi[i] = xr[-1]
        else:
            idx = int(np.argmax(below))
            x0, x1, s0, s1 = xr[idx - 1], xr[idx], sr[idx - 1], sr[idx]
            xi[i] = x1 if s1 == s0 else x0 + (threshold - s0) * (x1 - x0) / (s1 - s0)
    return y_unique, xi


def run_parametric_study(param_key, values, eta_list=None, profile_eta=0.55,
                          verbose=True, **base_overrides):
    """Run run_case once per value of `param_key` on top of `base_overrides`,
    collecting both the full eta sweep (I_avg, s_max) and a field snapshot
    near `profile_eta` (membrane s(y)/I(y)/Cg_O2(y) profiles and the
    two-phase interface position x(y)). Used by every parametric-study
    figure in the paper (temperature, humidity, porosity, pressure
    difference, GDL thickness, permeability, contact angle) so the sweep
    logic lives in exactly one place.

    Returns {value: {eta, I_avg, s_max, iters, membrane_y, membrane_s,
    membrane_I, membrane_CgO2, interface_y, interface_x}}.
    """
    results = {}
    for v in values:
        if verbose:
            print(f"\n--- {param_key} = {v} ---", flush=True)
        overrides = dict(base_overrides)
        overrides[param_key] = v
        r = run_case(overrides=overrides, eta_list=eta_list, verbose=verbose,
                      save_fields_at=(profile_eta,))
        tag = f"eta{profile_eta:.3f}"
        fld = r["fields"].get(tag)
        if fld is None and r["fields"]:
            fld = next(iter(r["fields"].values()))   # fall back to nearest match found
        fld = fld or {}
        results[v] = dict(
            eta=r["eta"], I_avg=r["I_avg"], s_max=r["s_max"], iters=r["iters"],
            membrane_y=fld.get("membrane_y"), membrane_s=fld.get("membrane_s"),
            membrane_I=fld.get("membrane_I"), membrane_CgO2=fld.get("membrane_CgO2"),
            interface_y=fld.get("interface_y"), interface_x=fld.get("interface_x"))
    return results


def save_study_npz(path, param_key, values, results):
    """Flatten a run_parametric_study() result dict into a single .npz
    (npz only supports a flat name->array mapping). Keys are indexed by
    position (v0__eta, v0__I_avg, ...) rather than by the raw float value,
    to sidestep float-to-string formatting ambiguity; `values` gives the
    index -> parameter-value mapping. See load_study_npz (duplicated,
    NumPy-only, in the plotting cells -- they run outside the FEniCSx conda
    env and can't import this dolfinx-dependent module)."""
    payload = {"param_key": np.array([param_key]), "values": np.asarray(values, dtype=float)}
    for i, v in enumerate(values):
        for k, arr in results[v].items():
            if arr is not None:
                payload[f"v{i}__{k}"] = np.asarray(arr)
    np.savez(path, **payload)


def run_case(overrides=None, eta_list=None, verbose=True, save_fields_at=(0.55,),
             jacobian_diagnostics=False):
    """Solve the M^2 two-phase PEMFC cathode model over an eta sweep, using
    a monolithic Newton solve on a mixed (p, C_w, Cg_O2) space at each eta
    (saturation s embedded as a smooth UFL expression of C_w -- see the
    module header for why this replaced segregated Picard iteration).

    Parameters
    ----------
    overrides : dict, optional overrides on top of DEFAULT_PARAMS.
    eta_list  : array of overpotentials (V) to sweep, default 18 pts.
    save_fields_at : eta values (nearest match) to snapshot full 2D
        fields for, in addition to the polarization data.
    jacobian_diagnostics : if True, after every Newton solve attempt
        (converged or not), extract the SNES's current Jacobian and log
        its smallest few and largest singular values -- a direct,
        principled way to check whether a stuck/slow region is a genuine
        fold (turning point) bifurcation (smallest singular value -> 0,
        i.e. Jacobian going singular) versus merely a hard-but-regular
        nonlinear region (Jacobian stays well-conditioned, Newton is just
        slow/needs a better initial guess). Adds real cost (a dense SVD
        of the ~4500x4500 system per solve, several seconds each) so it's
        off by default -- meant for targeted diagnostic sweeps, not the
        production parametric studies.

    Returns
    -------
    dict with 'eta', 'I_avg' (A/m^2), 's_max', 'iters' (Newton iteration
    count actually used, for diagnostics), and 'fields' (a dict
    eta_tag -> {coords, p, Cw, CgO2, s, ...}).
    """
    P = dict(DEFAULT_PARAMS)
    try:
        open(PROGRESS_LOG, "w").close()  # fresh log for this call
    except OSError:
        pass
    _progress_log(f"run_case() started (overrides={overrides}, "
                  f"eta_list size={None if eta_list is None else len(eta_list)})")
    if overrides:
        P.update(overrides)

    F, R = P["F"], P["R"]
    K_perm, eps_p = P["K_perm"], P["eps_p"]
    p_in, p_out = P["p_in"], P["p_out"]
    mu_g, D_O2_g, D_H2O_g = P["mu_g"], P["D_O2_g"], P["D_H2O_g"]
    I0_ref, alpha_c, alpha_w = P["I0"], P["alpha_c"], P["alpha_w"]
    T_ref_I0, Ea_orr = P["T_ref_I0"], P["Ea_orr"]
    rho_l = P["rho_l"]
    M_H2O, M_O2, M_N2 = P["M_H2O"], P["M_O2"], P["M_N2"]
    mu_l = P["mu_l"]
    T_cell, C_O2_ref = P["T_cell"], P["C_O2_ref"]
    # Arrhenius-corrected exchange current density (paper Sec. 4.3: higher T
    # raises I0 via faster ORR kinetics -- "increase in Reference current
    # density"). I0_ref is anchored at T_ref_I0=330K (the base-case
    # operating temperature), not the "333.0K" label on Table 2's I0 row.
    I0 = I0_ref * np.exp(-(Ea_orr / R) * (1.0 / T_cell - 1.0 / T_ref_I0))
    RH_in = P["RH_in"]
    theta_c_deg = P["theta_c_deg"]
    theta_c = theta_c_deg * np.pi / 180.0
    sigma_st = P["sigma_st"]
    X_O2_in, X_N2_in = P["X_O2_in"], P["X_N2_in"]
    w_ch, w_rb, Lx = P["w_ch"], P["w_rb"], P["Lx"]
    n_corey, tau_brug, s_max = P["n_corey"], P["tau_brug"], P["s_max"]
    s_smooth_eps = P["s_smooth_eps"]
    single_phase = P["single_phase"]
    nx, ny = P["nx"], P["ny"]
    newton_rtol, newton_atol, newton_max_it = P["newton_rtol"], P["newton_atol"], P["newton_max_it"]

    P_sat = Psat_atm(T_cell - 273.15) * 101325.0
    Ly = w_ch + w_rb
    y_inlet_end = 0.5 * w_ch
    y_rib_end = 0.5 * w_ch + w_rb

    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [Lx, Ly]], [nx, ny], mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    def on_membrane(x): return np.isclose(x[0], 0.0)
    def on_inlet(x):    return np.isclose(x[0], Lx) & (x[1] <= y_inlet_end + 1e-12)
    def on_rib(x):      return np.isclose(x[0], Lx) & (x[1] > y_inlet_end + 1e-12) & (x[1] <= y_rib_end + 1e-12)
    def on_outlet(x):   return np.isclose(x[0], Lx) & (x[1] > y_rib_end + 1e-12)

    TAG_MEMBRANE, TAG_INLET, TAG_RIB, TAG_OUTLET = 1, 2, 3, 4
    facet_indices, facet_markers = [], []
    for tag, locator in [(TAG_MEMBRANE, on_membrane), (TAG_INLET, on_inlet),
                          (TAG_RIB, on_rib), (TAG_OUTLET, on_outlet)]:
        idx = mesh.locate_entities_boundary(domain, fdim, locator)
        facet_indices.append(idx)
        facet_markers.append(np.full_like(idx, tag))
    facet_indices = np.concatenate(facet_indices)
    facet_markers = np.concatenate(facet_markers)
    order = np.argsort(facet_indices)
    facet_tags = mesh.meshtags(domain, fdim, facet_indices[order], facet_markers[order])

    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    dx = ufl.Measure("dx", domain=domain)
    nvec = ufl.FacetNormal(domain)

    # Scalar space (for output/projection only) + mixed space (for the
    # coupled Newton solve).
    V = fem.functionspace(domain, ("Lagrange", 1))
    P1e = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    ME = basix.ufl.mixed_element([P1e, P1e, P1e])
    W = fem.functionspace(domain, ME)

    w = fem.Function(W, name="w")
    v_te = ufl.TestFunction(W)
    p_hat_sol, Cw_sol, CgO2_sol = ufl.split(w)
    v_p, v_cw, v_cgo2 = ufl.split(v_te)

    # Non-dimensionalized pressure unknown: p_hat = (p - p_out)/P_scale,
    # P_scale a fixed O(driving-pressure-difference) constant (NOT
    # recomputed per case, so parametric sweeps over p_in stay on a
    # consistent scale). Raw p (~1e5 Pa) mixed with C_w/Cg_O2 (~O(1) mass
    # fractions) in one monolithic Jacobian meant the pressure block's
    # entries were ~5 orders of magnitude larger than the others purely
    # from unit choice -- confirmed by direct Jacobian SVD diagnostics
    # (see module header) to dominate the *raw* condition number
    # (~1e9-1e10 even in the easy, far-from-any-fold regime); this
    # doesn't touch the genuine fold near the flooding-onset transition
    # (that's a real feature of the underlying system, not a scaling
    # artifact -- confirmed by the SAME diagnostics showing the
    # *equilibrated* smallest singular value itself, not just the raw
    # condition number, collapsing there), but it does remove the
    # constant baseline ill-conditioning everywhere else.
    P_scale = 1000.0  # Pa, O(the typical driving pressure difference)
    p_sol = p_out + P_scale * p_hat_sol  # reconstruct physical pressure; everything
                                          # downstream (u_darcy, F_pres, field saving)
                                          # references p_sol exactly as before

    # --- inlet composition (RH_in-dependent, dry-gas fractions renormalised) ---
    X_H2O_in = np.clip(RH_in * P_sat / p_in, 0.0, 0.98)
    dry_scale = 1.0 - X_H2O_in
    X_O2_eff = X_O2_in * dry_scale
    X_N2_eff = X_N2_in * dry_scale
    M_mix_in = X_O2_eff * M_O2 + X_N2_eff * M_N2 + X_H2O_in * M_H2O
    Y_O2_in = X_O2_eff * M_O2 / M_mix_in
    Y_H2O_in = X_H2O_in * M_H2O / M_mix_in
    M_dry_in = X_O2_in * M_O2 + X_N2_in * M_N2   # dry-basis MW for the saturation formula

    rho_g_const = p_out * M_mix_in / (R * T_cell)
    Cg_sat = (P_sat * M_H2O) / (P_sat * M_H2O + (p_in - P_sat) * M_dry_in)
    Cl_sat = 1.0

    # --- BCs on the mixed space ---
    inlet_facets = facet_tags.find(TAG_INLET)
    outlet_facets = facet_tags.find(TAG_OUTLET)
    # Scalar-valued BCs on a mixed-space subspace use the single-space dof
    # form (W.sub(i)) -- the (subspace, collapsed_subspace) tuple form is
    # only needed when the BC value is itself a Function on the collapsed
    # subspace, which isn't the case here (plain constants).
    dofs_p_in = fem.locate_dofs_topological(W.sub(0), fdim, inlet_facets)
    dofs_p_out = fem.locate_dofs_topological(W.sub(0), fdim, outlet_facets)
    dofs_cw_in = fem.locate_dofs_topological(W.sub(1), fdim, inlet_facets)
    dofs_cgo2_in = fem.locate_dofs_topological(W.sub(2), fdim, inlet_facets)
    bcs = [
        fem.dirichletbc(default_scalar_type((p_in - p_out) / P_scale), dofs_p_in, W.sub(0)),
        fem.dirichletbc(default_scalar_type(0.0), dofs_p_out, W.sub(0)),
        fem.dirichletbc(default_scalar_type(Y_H2O_in), dofs_cw_in, W.sub(1)),
        fem.dirichletbc(default_scalar_type(Y_O2_in), dofs_cgo2_in, W.sub(2)),
    ]
    membrane_dofs_V = fem.locate_dofs_topological(V, fdim, facet_tags.find(TAG_MEMBRANE))

    # --- saturation as a smooth (differentiable) UFL expression of C_w ---
    def s_expr(Cw):
        if single_phase:
            return 0.0 * Cw   # a genuine UFL zero expression (grad(0.0) would fail; grad(0*Cw) is fine)
        raw = rho_g_const * (Cw - Cg_sat) / (rho_l * (Cl_sat - Cw) + rho_g_const * (Cw - Cg_sat) + 1e-12)
        s_smooth = 0.5 * (raw + ufl.sqrt(raw**2 + s_smooth_eps**2))
        s_smooth = ufl.max_value(s_smooth, 0.0)
        return ufl.min_value(s_smooth, s_max)

    def min_smooth(a, b, eps=s_smooth_eps):
        return 0.5 * (a + b - ufl.sqrt((a - b) ** 2 + eps ** 2))

    hydrophobic = theta_c_deg >= 90.0

    # --- assemble the coupled residual (see module header for the
    # derivation/sign history of every term; only the LAG between s and
    # C_w was eliminated here, the physics itself is unchanged). Every
    # s-dependent quantity is computed EXACTLY ONCE below and reused by
    # plain variable reference from then on -- earlier drafts routed
    # these through nested closures (e.g. Gamma_e() called Dc_e() again
    # internally even though Dc was already computed at top level, and
    # Dc_e() itself re-derived lambda_l/lambda_g/mu_mix/rho_mix from
    # scratch instead of reusing the top-level values), which silently
    # blew up the symbolic expression tree by ~10-20x before FFCx
    # compiled it and UFL symbolically differentiated it for the
    # Jacobian -- almost certainly why the solve hung for tens of
    # minutes instead of the few seconds this mesh size should take.
    s = s_expr(Cw_sol)

    krl = s ** n_corey
    krg = (1.0 - s) ** n_corey
    lam_l = (krl / mu_l) / (krl / mu_l + krg / mu_g + 1e-30)
    lam_g = 1.0 - lam_l
    rho_mix = rho_g_const * (1.0 - s) + rho_l * s
    nu_g_c, nu_l_c = mu_g / rho_g_const, mu_l / rho_l
    nu_mix = 1.0 / (krg / nu_g_c + krl / nu_l_c + 1e-30)
    mu_mix = nu_mix * rho_mix
    kappa = rho_mix * K_perm / mu_mix

    if hydrophobic:
        dJds = 1.417 - 4.240 * s + 3.789 * s**2
    else:
        u_hp = 1.0 - s
        dJds = -(1.417 - 4.240 * u_hp + 3.789 * u_hp**2)

    Dc = ((lam_l * lam_g * K_perm / (nu_mix + 1e-30))
          * sigma_st * np.cos(theta_c) * (eps_p / K_perm) ** 0.5 * dJds)
    Gamma = Dc * (1.0 - Cg_sat) / (rho_l - rho_g_const * Cg_sat)
    grads = ufl.grad(s)   # comes for free via UFL's chain rule -- no projection needed

    u_darcy = -(kappa / rho_mix) * ufl.grad(p_sol)

    CgH2O_local = min_smooth(Cw_sol, Cg_sat)
    Weff_w = lam_l + lam_g * CgH2O_local

    Deff_O2 = rho_g_const * (eps_p * (1.0 - s)) ** tau_brug * D_O2_g
    Diff_w = rho_g_const * (eps_p * (1.0 - s)) ** tau_brug * D_H2O_g + Gamma * rho_mix

    Jl_vec = Dc * grads
    F_conv_w = Weff_w * rho_mix * u_darcy
    W_O2 = rho_mix * lam_g * u_darcy

    eta_const = fem.Constant(domain, default_scalar_type(0.0))
    C_O2_molar = rho_g_const * CgO2_sol / M_O2
    I_local = (1.0 - s) * I0 * (C_O2_molar / C_O2_ref) * ufl.exp(alpha_c * F * eta_const / (R * T_cell))
    N_O2_expr = (M_O2 / (4.0 * F)) * I_local
    N_w_expr = (M_H2O * (1.0 + 2.0 * alpha_w) / (2.0 * F)) * I_local

    F_pres = (kappa * ufl.inner(ufl.grad(p_sol), ufl.grad(v_p)) * dx
              - (N_w_expr - N_O2_expr) * v_p * ds(TAG_MEMBRANE))

    # --- SUPG stabilization: DISABLED for now -----------------------------
    # Was added to fix node-to-node oscillation seen in the two-phase
    # regime (Bruggeman-shrunk diffusivity + membrane flux pushing the
    # local Peclet number high), but JIT compilation hung even after
    # simplifying it to the cheap non-conservative a.grad(u) form -- and
    # that was never isolated against a no-SUPG baseline to confirm SUPG
    # itself was the cause (vs. the base 3-field coupling already being
    # this expensive to compile). Removed for now to get a working
    # baseline; re-add (or find a cheaper alternative) once that's
    # confirmed. The Picard companion notebook still has SUPG and isn't
    # showing this compile-time problem, so it remains the more reliable
    # source for two-phase-regime figures in the meantime.
    # ----------------------------------------------------------------------

    F_water = (Diff_w * ufl.inner(ufl.grad(Cw_sol), ufl.grad(v_cw)) * dx
               - ufl.inner(F_conv_w, ufl.grad(v_cw)) * dx
               + _smooth_outflow(ufl.inner(F_conv_w, nvec)) * v_cw * ds(TAG_OUTLET)
               - N_w_expr * v_cw * ds(TAG_MEMBRANE))

    F_oxy = (-CgO2_sol * ufl.inner(W_O2, ufl.grad(v_cgo2)) * dx
             + Deff_O2 * ufl.inner(ufl.grad(CgO2_sol), ufl.grad(v_cgo2)) * dx
             + CgO2_sol * ufl.inner(Jl_vec, ufl.grad(v_cgo2)) * dx
             + CgO2_sol * _smooth_outflow(ufl.inner(W_O2, nvec)) * v_cgo2 * ds(TAG_OUTLET)
             + N_O2_expr * v_cgo2 * ds(TAG_MEMBRANE))

    F_total = F_pres + F_water + F_oxy

    if verbose:
        print("  [building nonlinear form + JIT-compiling residual/Jacobian -- can take a "
              "while the first time; no further output until this finishes]", flush=True)
    _progress_log("building nonlinear form + starting JIT compilation")
    snes_petsc_options = {
        "snes_type": "newtonls",
        "snes_rtol": newton_rtol,
        "snes_atol": newton_atol,
        "snes_max_it": newton_max_it,
        "snes_error_if_not_converged": False,   # handle non-convergence ourselves (bisection fallback)
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    # Explicit quadrature degree cap: sqrt/min_value/max_value in s_expr()
    # (and, previously, in SUPG's tau) are non-polynomial, so FFCx's
    # automatic degree-estimation heuristic has no exact answer to fall
    # back on and can pick an excessively high degree -- ballooning both
    # the generated C code size and its compile time, independent of how
    # symbolically "redundant" the UFL expression tree itself is. Capping
    # it explicitly avoids that; degree 4 is more than enough accuracy
    # for smooth (if non-polynomial) coefficients on P1 elements.
    form_compiler_options = {"quadrature_degree": 4}
    problem = NonlinearProblem(F_total, w, bcs=bcs, petsc_options=snes_petsc_options,
                                petsc_options_prefix="pemfc_newton_",
                                form_compiler_options=form_compiler_options)
    if verbose:
        print("  [JIT compilation done -- starting eta sweep]", flush=True)
    _progress_log("JIT compilation done -- starting eta sweep")

    # --- initial guess: uniform inlet-like state ---
    w.sub(0).interpolate(lambda x: np.full(x.shape[1], (p_in - p_out) / P_scale))
    w.sub(1).interpolate(lambda x: np.full(x.shape[1], Y_H2O_in))
    w.sub(2).interpolate(lambda x: np.full(x.shape[1], Y_O2_in))
    w.x.scatter_forward()

    def project_scalar(expr, prefix):
        u_tr, vt = ufl.TrialFunction(V), ufl.TestFunction(V)
        a_p = u_tr * vt * dx
        L_p = expr * vt * dx
        prob = LinearProblem(a_p, L_p, bcs=[],
                              petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                              petsc_options_prefix=prefix)
        return prob.solve().x.array.copy()

    def checkpoint(): return w.x.array.copy()
    def restore(ckpt): w.x.array[:] = ckpt

    def solve_at_eta(eta_target, eta_from, max_bisect=4):
        """Newton-solve at eta_target, warm-started from the current state
        (converged at eta_from). Falls back to a bisected intermediate eta
        (and retries from there) if Newton fails to converge -- Newton can
        still fail from a poor-enough initial guess even though it no
        longer suffers the segregated-lag oscillation; this keeps the same
        continuation robustness the Picard version had."""
        pending = [eta_target]
        eta_prev = eta_from
        n_bisect_total = 0
        result = None
        while pending:
            eta_now = pending[-1]
            ckpt = checkpoint()
            eta_const.value = eta_now
            _progress_log(f"attempting Newton solve at eta={eta_now:.4f} ...")
            try:
                problem.solve()
                converged_reason = problem.solver.getConvergedReason()
                n_its = problem.solver.getIterationNumber()
                converged = converged_reason > 0
                _progress_log(f"  -> solve returned: converged={converged}, n_its={n_its}")
                if jacobian_diagnostics:
                    try:
                        sv_small_raw, sv_large_raw, sv_small_sc, sv_large_sc = \
                            _jacobian_singular_values(problem)
                        cond_raw = sv_large_raw[-1] / max(sv_small_raw[0], 1e-300)
                        cond_sc = sv_large_sc[-1] / max(sv_small_sc[0], 1e-300)
                        _progress_log(
                            f"  Jacobian @ eta={eta_now:.4f}  "
                            f"RAW: smallest={np.array2string(sv_small_raw, precision=3)} "
                            f"largest={np.array2string(sv_large_raw, precision=3)} "
                            f"cond~{cond_raw:.3e}  ||  "
                            f"SCALED (equilibrated, the trustworthy one): "
                            f"smallest={np.array2string(sv_small_sc, precision=3)} "
                            f"largest={np.array2string(sv_large_sc, precision=3)} "
                            f"cond~{cond_sc:.3e}")
                    except Exception as e:
                        _progress_log(f"  Jacobian diagnostic failed: {e}")
            except RuntimeError as e:
                n_its, converged = newton_max_it, False
                _progress_log(f"  -> RuntimeError: {e}")
            if converged or n_bisect_total >= max_bisect:
                pending.pop()
                eta_prev = eta_now
                result = (n_its, converged)
            else:
                restore(ckpt)
                n_bisect_total += 1
                pending.append(0.5 * (eta_prev + eta_now))
        return result[0], result[1], n_bisect_total

    if eta_list is None:
        eta_list = np.concatenate([np.linspace(0.02, 0.30, 8), np.linspace(0.33, 0.60, 10)])
    eta_list = np.asarray(eta_list)

    # Precompute exactly which eta(s) in eta_list are the single closest
    # match to each requested snapshot target -- NOT "within half the
    # first interval's spacing" (eta_list[1]-eta_list[0]), which breaks
    # down for non-uniformly spaced eta_list (silently matching multiple
    # neighboring points) and crashes outright for a single-point sweep
    # (eta_list[1] out of bounds) -- same fix already applied on the
    # Picard side.
    snapshot_etas = {float(eta_list[int(np.argmin(np.abs(eta_list - target)))])
                      for target in save_fields_at}

    I_avg_list, s_max_list, iters_list, fields = [], [], [], {}
    eta_prev = float(eta_list[0]) - 1e-6
    for eta in eta_list:
        n_its, converged, n_bisect = solve_at_eta(float(eta), eta_prev)
        eta_prev = float(eta)

        s_arr = project_scalar(s, "pemfc_sproj_")
        Jloc_arr = project_scalar(I_local, "pemfc_jlocproj_")
        CgO2_arr = project_scalar(CgO2_sol, "pemfc_cgo2proj_")
        I_avg = float(np.mean(Jloc_arr[membrane_dofs_V]))
        s_max_now = float(s_arr.max())
        CgO2_membrane_min = float(CgO2_arr[membrane_dofs_V].min())

        I_avg_list.append(I_avg)
        s_max_list.append(s_max_now)
        iters_list.append(n_its)
        if verbose:
            flags = []
            if CgO2_membrane_min > Y_O2_in + 1e-6:
                flags.append(" \u26a0 O2 ABOVE INLET AT MEMBRANE")
            if not converged:
                flags.append(" \u26a0 NEWTON DID NOT CONVERGE")
            if n_bisect:
                flags.append(f" [{n_bisect} continuation bisection(s)]")
            line = (f"  eta={eta:6.3f} V   I_avg={I_avg:9.3f} A/m^2 = {I_avg/1e4:6.4f} A/cm^2   "
                    f"s_max={s_max_now:.3f}   Cg_O2,membrane={CgO2_membrane_min:.4f}   "
                    f"Newton iters={n_its}{''.join(flags)}")
            print(line, flush=True)
            _progress_log(line)

        if float(eta) in snapshot_etas:
            coords = V.tabulate_dof_coordinates()[:, :2]
            p_arr = project_scalar(p_sol, "pemfc_pproj_")
            Cw_arr = project_scalar(Cw_sol, "pemfc_cwproj_")
            vx = project_scalar(u_darcy[0], "pemfc_vx_")
            vy = project_scalar(u_darcy[1], "pemfc_vy_")
            # Phase velocities (Table 1 Eq. 14): rho_l*ul = jl + lambda_l*rho*u,
            # rho_g*ug = -jl + lambda_g*rho*u.
            ul_expr = (Jl_vec + lam_l * rho_mix * u_darcy) / rho_l
            ug_expr = (-Jl_vec + lam_g * rho_mix * u_darcy) / rho_g_const
            ulx = project_scalar(ul_expr[0], "pemfc_ulx_")
            uly = project_scalar(ul_expr[1], "pemfc_uly_")
            ugx = project_scalar(ug_expr[0], "pemfc_ugx_")
            ugy = project_scalar(ug_expr[1], "pemfc_ugy_")
            y_line, xi_line = extract_interface(coords, s_arr, threshold=0.02)
            membrane = extract_line(coords, {"s": s_arr, "I_local": Jloc_arr, "CgO2": CgO2_arr})
            fields[f"eta{eta:.3f}"] = dict(
                coords=coords.copy(), p=p_arr, Cw=Cw_arr, CgO2=CgO2_arr, s=s_arr,
                vx=vx, vy=vy, ulx=ulx, uly=uly, ugx=ugx, ugy=ugy, eta=float(eta),
                interface_y=y_line, interface_x=xi_line,
                membrane_y=membrane["y"], membrane_s=membrane["s"],
                membrane_I=membrane["I_local"], membrane_CgO2=membrane["CgO2"])

    return dict(eta=eta_list, I_avg=np.array(I_avg_list), s_max=np.array(s_max_list),
                iters=np.array(iters_list), fields=fields, params=P)


def run_case_arclength(overrides=None, eta_start=0.30, ds=0.01, ds_min=1e-4, ds_max=0.03,
                        n_steps=60, max_newton_it=30, newton_tol=1e-6, verbose=True,
                        resume=False, checkpoint_path="/content/pemfc_m2_arclength_checkpoint.npz",
                        refresh_tangent=False, refresh_step=1e-4, snapshot_dir=None,
                        svd_diagnostic_every=None, svd_out_path=None, init_w_array=None):
    """Trace the (eta, w) solution branch using pseudo-arclength continuation
    (Keller's bordering algorithm) instead of run_case()'s eta-parametrized
    Newton continuation.

    WHY: direct Jacobian SVD diagnostics (run_case(..., jacobian_diagnostics=
    True) through the eta~0.30-0.36 range, base case) showed the *scaled*
    (Jacobi-equilibrated, removing the artificial ill-conditioning from
    mixing ~1e5 Pa pressure with O(1) mass fractions -- see P_scale above)
    smallest singular value collapsing from ~1.6e-5 at eta=0.355 to ~2.9e-10
    at eta=0.360 -- a genuine fold (turning point), not a scaling artifact
    or merely a hard-but-regular nonlinearity. At a fold, eta is *not* a
    valid local coordinate for the solution branch (dI/deta -> infinity, or
    the branch turns back on itself), so run_case()'s approach -- solve
    F(w, eta_target)=0 for w at a FIXED eta_target -- cannot converge near
    one no matter how good the warm start or how finely eta is bisected
    (confirmed empirically: bisecting all the way to eta=0.3563 still gave
    a near-singular Jacobian). Pseudo-arclength continuation instead treats
    eta as an UNKNOWN alongside w, adding an arclength constraint in its
    place -- allowing the path (w(s), eta(s)) to be traced smoothly through
    a fold, with eta itself moving non-monotonically (or even backwards)
    across it if that's what the branch does.

    ALGORITHM (Keller's bordering, avoids needing an augmented (n+1)x(n+1)
    matrix -- reuses the same n x n Jacobian factorization twice per Newton
    iteration instead):
      Predictor:  w_pred = w0 + ds*dw_tan,  eta_pred = eta0 + ds*deta_tan
      Corrector (Newton on the bordered system, iterating on (w_pred, eta_pred)):
        Solve J v1 = -F(w_pred, eta_pred)
        Solve J v2 = -dF/deta(w_pred, eta_pred)
        N = dw_tan.(w_pred-w0) + deta_tan*(eta_pred-eta0) - ds   [arclength residual]
        delta_eta = (-N - dw_tan.v1) / (deta_tan + dw_tan.v2)
        delta_w   = v1 + delta_eta*v2
        w_pred += delta_w; eta_pred += delta_eta; repeat until small
      Tangent update (at the new converged point, reusing J, dF/deta there):
        Solve J z = -dF/deta
        deta_new = 1/(deta_tan_old + dw_tan_old.z);  dw_new = deta_new*z
        normalize (dw_new, deta_new) to unit length; flip sign if it
        reverses direction relative to the previous tangent.

    STATUS: this is a substantially different, substantially less battle-
    tested code path than run_case() -- it duplicates run_case()'s physics
    setup verbatim (deliberately kept separate rather than refactored to
    share code, so there is zero risk of this destabilizing the validated
    run_case() implementation) but replaces the entire solve strategy with
    hand-rolled PETSc calls that have NOT been run successfully end-to-end
    even once. Treat every part of the bordering/tangent logic below as a
    first draft that will likely need real debugging with actual execution
    feedback -- the physics/residual/BC setup is the validated part, the
    continuation algorithm around it is not.

    CHECKPOINTING: after every accepted step, the current point, tangent,
    and full history are saved to `checkpoint_path` (a .npz file) --
    meant for long unattended runs where a Colab disconnect partway
    through would otherwise lose everything. Call with resume=True (same
    `overrides` as the original call -- mesh/physical parameters must
    match for the saved state to mean anything) to continue from the
    last checkpoint instead of re-bootstrapping from eta_start; `n_steps`
    in a resumed call is the number of ADDITIONAL steps to take from
    where the checkpoint left off, not a total. If no checkpoint file
    exists yet, resume=True is silently ignored and a fresh run starts.

    refresh_tangent=True (only meaningful together with resume=True):
    discards the checkpoint's saved tangent and re-bootstraps a fresh one
    via two ordinary Newton solves right at the checkpoint's current
    point, before continuing the main loop. Useful if a run stalls for
    many steps at a fixed eta with no improving trend -- check with
    diagnose_fold_point() first: if the stuck point is NOT substantially
    (many orders of magnitude) worse-conditioned than an ordinary
    reference point, the stall is more likely from tangent drift
    (accumulated floating-point error over hundreds of prior steps) than
    a genuine fold/bifurcation, and a fresh tangent may resolve it
    without needing any change to the continuation algorithm itself.

    Refreshing recomputes the tangent DIRECTLY from the Jacobian at the
    CURRENT point (w_last, eta_last) -- solving J*z = -F_eta (the same
    linear system used for the post-step tangent update) and setting the
    new tangent from z, rather than doing new nonlinear solves at nearby
    eta values. An earlier version bootstrapped via two ordinary Newton
    solves (at eta_last and eta_last +/- a small refresh_step) and took
    their secant -- this turned out to be fundamentally unsound: an
    independent test (solving from two different initial guesses at a
    fixed eta near a stuck point) confirmed the TRUE eta-fixed solution
    there had a dramatically different s_max than the branch being
    traced, meaning s(eta) was locally near-vertical right where a
    refresh was needed -- exactly the situation ordinary (eta-fixed)
    Newton cannot be trusted in, which is the whole reason arclength
    continuation is used in the first place. The direct-Jacobian
    approach never solves a new nonlinear problem, so it cannot jump to
    a different branch; `refresh_step` is kept as a parameter for
    backward compatibility but no longer affects anything.

    Returns dict with 'eta', 'I_avg', 's_max', 'ds_used', 'converged'
    (arrays, one entry per accepted continuation step, INCLUDING any
    prior steps restored from a checkpoint) -- eta may not be monotonic
    if the branch actually turns back across a fold.
    """
    from petsc4py import PETSc

    P = dict(DEFAULT_PARAMS)
    try:
        open(PROGRESS_LOG, "w").close()
    except OSError:
        pass
    _progress_log(f"run_case_arclength() started (overrides={overrides}, eta_start={eta_start}, ds={ds})")
    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)
        _progress_log(f"  snapshot_dir={snapshot_dir} (saving full fields every accepted step)")
    if overrides:
        P.update(overrides)

    F, R = P["F"], P["R"]
    K_perm, eps_p = P["K_perm"], P["eps_p"]
    p_in, p_out = P["p_in"], P["p_out"]
    mu_g, D_O2_g, D_H2O_g = P["mu_g"], P["D_O2_g"], P["D_H2O_g"]
    I0_ref, alpha_c, alpha_w = P["I0"], P["alpha_c"], P["alpha_w"]
    T_ref_I0, Ea_orr = P["T_ref_I0"], P["Ea_orr"]
    rho_l = P["rho_l"]
    M_H2O, M_O2, M_N2 = P["M_H2O"], P["M_O2"], P["M_N2"]
    mu_l = P["mu_l"]
    T_cell, C_O2_ref = P["T_cell"], P["C_O2_ref"]
    I0 = I0_ref * np.exp(-(Ea_orr / R) * (1.0 / T_cell - 1.0 / T_ref_I0))
    RH_in = P["RH_in"]
    theta_c_deg = P["theta_c_deg"]
    theta_c = theta_c_deg * np.pi / 180.0
    sigma_st = P["sigma_st"]
    X_O2_in, X_N2_in = P["X_O2_in"], P["X_N2_in"]
    w_ch, w_rb, Lx = P["w_ch"], P["w_rb"], P["Lx"]
    n_corey, tau_brug, s_max_cap = P["n_corey"], P["tau_brug"], P["s_max"]
    s_smooth_eps = P["s_smooth_eps"]
    single_phase = P["single_phase"]
    nx, ny = P["nx"], P["ny"]
    newton_rtol, newton_atol, newton_max_it = P["newton_rtol"], P["newton_atol"], P["newton_max_it"]

    P_sat = Psat_atm(T_cell - 273.15) * 101325.0
    Ly = w_ch + w_rb
    y_inlet_end = 0.5 * w_ch
    y_rib_end = 0.5 * w_ch + w_rb

    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [Lx, Ly]], [nx, ny], mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    def on_membrane(x): return np.isclose(x[0], 0.0)
    def on_inlet(x):    return np.isclose(x[0], Lx) & (x[1] <= y_inlet_end + 1e-12)
    def on_rib(x):      return np.isclose(x[0], Lx) & (x[1] > y_inlet_end + 1e-12) & (x[1] <= y_rib_end + 1e-12)
    def on_outlet(x):   return np.isclose(x[0], Lx) & (x[1] > y_rib_end + 1e-12)

    TAG_MEMBRANE, TAG_INLET, TAG_RIB, TAG_OUTLET = 1, 2, 3, 4
    facet_indices, facet_markers = [], []
    for tag, locator in [(TAG_MEMBRANE, on_membrane), (TAG_INLET, on_inlet),
                          (TAG_RIB, on_rib), (TAG_OUTLET, on_outlet)]:
        idx = mesh.locate_entities_boundary(domain, fdim, locator)
        facet_indices.append(idx)
        facet_markers.append(np.full_like(idx, tag))
    facet_indices = np.concatenate(facet_indices)
    facet_markers = np.concatenate(facet_markers)
    order = np.argsort(facet_indices)
    facet_tags = mesh.meshtags(domain, fdim, facet_indices[order], facet_markers[order])

    ds_meas = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    dx = ufl.Measure("dx", domain=domain)
    nvec = ufl.FacetNormal(domain)

    V = fem.functionspace(domain, ("Lagrange", 1))
    P1e = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    ME = basix.ufl.mixed_element([P1e, P1e, P1e])
    W = fem.functionspace(domain, ME)

    w = fem.Function(W, name="w")
    v_te = ufl.TestFunction(W)
    p_hat_sol, Cw_sol, CgO2_sol = ufl.split(w)
    v_p, v_cw, v_cgo2 = ufl.split(v_te)


    # Automatic mesh-resolution scaling for the arclength step sizes: the
    # plain Euclidean norm on the raw dof vector (used throughout this
    # function for ||delta_w||) grows roughly as sqrt(n_dofs) under mesh
    # refinement for a FIXED underlying physical perturbation -- a properly
    # area-weighted L2 norm would be refinement-invariant, but this simpler
    # vector norm isn't. Confirmed empirically: doubling mesh density made
    # eta advance ~sqrt(2)x slower per unit ds at a fixed ds_max. Rescaling
    # ds/ds_max/ds_min/refresh_step by sqrt(n_dofs / n_dofs_reference)
    # keeps the arclength step corresponding to roughly the same physical
    # step size regardless of mesh resolution; this is a no-op (factor
    # exactly 1.0) at the default nx/ny.
    n_dofs_reference = 3 * (DEFAULT_PARAMS["nx"] + 1) * (DEFAULT_PARAMS["ny"] + 1)
    n_dofs_actual = len(w.x.array)
    mesh_scale = float(np.sqrt(n_dofs_actual / n_dofs_reference))
    if abs(mesh_scale - 1.0) > 1e-6:
        _progress_log(f"  mesh resolution differs from reference ({n_dofs_actual} vs "
                      f"{n_dofs_reference} dofs) -- auto-scaling ds/ds_max/ds_min/refresh_step "
                      f"by {mesh_scale:.4f}x")
    ds = ds * mesh_scale
    ds_max = ds_max * mesh_scale
    ds_min = ds_min * mesh_scale
    refresh_step = refresh_step * mesh_scale
    P_scale = 1000.0
    p_sol = p_out + P_scale * p_hat_sol

    X_H2O_in = np.clip(RH_in * P_sat / p_in, 0.0, 0.98)
    dry_scale = 1.0 - X_H2O_in
    X_O2_eff = X_O2_in * dry_scale
    X_N2_eff = X_N2_in * dry_scale
    M_mix_in = X_O2_eff * M_O2 + X_N2_eff * M_N2 + X_H2O_in * M_H2O
    Y_O2_in = X_O2_eff * M_O2 / M_mix_in
    Y_H2O_in = X_H2O_in * M_H2O / M_mix_in
    M_dry_in = X_O2_in * M_O2 + X_N2_in * M_N2

    rho_g_const = p_out * M_mix_in / (R * T_cell)
    Cg_sat = (P_sat * M_H2O) / (P_sat * M_H2O + (p_in - P_sat) * M_dry_in)
    Cl_sat = 1.0

    inlet_facets = facet_tags.find(TAG_INLET)
    outlet_facets = facet_tags.find(TAG_OUTLET)
    dofs_p_in = fem.locate_dofs_topological(W.sub(0), fdim, inlet_facets)
    dofs_p_out = fem.locate_dofs_topological(W.sub(0), fdim, outlet_facets)
    dofs_cw_in = fem.locate_dofs_topological(W.sub(1), fdim, inlet_facets)
    dofs_cgo2_in = fem.locate_dofs_topological(W.sub(2), fdim, inlet_facets)
    bcs = [
        fem.dirichletbc(default_scalar_type((p_in - p_out) / P_scale), dofs_p_in, W.sub(0)),
        fem.dirichletbc(default_scalar_type(0.0), dofs_p_out, W.sub(0)),
        fem.dirichletbc(default_scalar_type(Y_H2O_in), dofs_cw_in, W.sub(1)),
        fem.dirichletbc(default_scalar_type(Y_O2_in), dofs_cgo2_in, W.sub(2)),
    ]
    bc_dofs_all = np.concatenate([dofs_p_in, dofs_p_out, dofs_cw_in, dofs_cgo2_in])
    membrane_dofs_V = fem.locate_dofs_topological(V, fdim, facet_tags.find(TAG_MEMBRANE))

    def s_expr(Cw):
        if single_phase:
            return 0.0 * Cw
        raw = rho_g_const * (Cw - Cg_sat) / (rho_l * (Cl_sat - Cw) + rho_g_const * (Cw - Cg_sat) + 1e-12)
        s_smooth = 0.5 * (raw + ufl.sqrt(raw**2 + s_smooth_eps**2))
        s_smooth = ufl.max_value(s_smooth, 0.0)
        return ufl.min_value(s_smooth, s_max_cap)

    def min_smooth(a, b, eps=s_smooth_eps):
        return 0.5 * (a + b - ufl.sqrt((a - b) ** 2 + eps ** 2))

    hydrophobic = theta_c_deg >= 90.0

    s = s_expr(Cw_sol)
    krl = s ** n_corey
    krg = (1.0 - s) ** n_corey
    lam_l = (krl / mu_l) / (krl / mu_l + krg / mu_g + 1e-30)
    lam_g = 1.0 - lam_l
    rho_mix = rho_g_const * (1.0 - s) + rho_l * s
    nu_g_c, nu_l_c = mu_g / rho_g_const, mu_l / rho_l
    nu_mix = 1.0 / (krg / nu_g_c + krl / nu_l_c + 1e-30)
    mu_mix = nu_mix * rho_mix
    kappa = rho_mix * K_perm / mu_mix

    if hydrophobic:
        dJds = 1.417 - 4.240 * s + 3.789 * s**2
    else:
        u_hp = 1.0 - s
        dJds = -(1.417 - 4.240 * u_hp + 3.789 * u_hp**2)

    Dc = ((lam_l * lam_g * K_perm / (nu_mix + 1e-30))
          * sigma_st * np.cos(theta_c) * (eps_p / K_perm) ** 0.5 * dJds)
    Gamma = Dc * (1.0 - Cg_sat) / (rho_l - rho_g_const * Cg_sat)
    grads = ufl.grad(s)

    u_darcy = -(kappa / rho_mix) * ufl.grad(p_sol)

    CgH2O_local = min_smooth(Cw_sol, Cg_sat)
    C_O2_mix = rho_g_const * CgO2_sol / rho_mix  # Fig. 7c: oxygen mixture (rho*C^O2 = rho_g*Cg^O2)
    Weff_w = lam_l + lam_g * CgH2O_local

    Deff_O2 = rho_g_const * (eps_p * (1.0 - s)) ** tau_brug * D_O2_g
    Diff_w = rho_g_const * (eps_p * (1.0 - s)) ** tau_brug * D_H2O_g + Gamma * rho_mix

    Jl_vec = Dc * grads
    F_conv_w = Weff_w * rho_mix * u_darcy
    W_O2 = rho_mix * lam_g * u_darcy

    eta_const = fem.Constant(domain, default_scalar_type(0.0))
    C_O2_molar = rho_g_const * CgO2_sol / M_O2
    I_local = (1.0 - s) * I0 * (C_O2_molar / C_O2_ref) * ufl.exp(alpha_c * F * eta_const / (R * T_cell))
    N_O2_expr = (M_O2 / (4.0 * F)) * I_local
    N_w_expr = (M_H2O * (1.0 + 2.0 * alpha_w) / (2.0 * F)) * I_local

    # OUTLET STABILIZATION (max_value(...,0.0) clip on the boundary normal
    # flux): the outlet is a natural/Neumann advective-outflow condition,
    # which implicitly assumes genuine outflow there. diagnose_fold_point()
    # + check_outlet_flow() traced the second (non-fold, ~40x-conditioned)
    # stuck point directly to this -- W_O2's normal (y) component CHANGES
    # SIGN inside the outlet segment as flooding intensifies (confirmed at
    # both the original and a refined mesh), meaning part of the "outlet"
    # locally has reversed (inflow) flow. The unclipped weak form has no
    # prescribed composition for that reversed inflow (nothing quantifies
    # what's flowing back IN), which makes its own Jacobian contribution
    # locally indefinite right where the sign flips -- a well known failure
    # mode for natural outflow BCs under flow reversal. Clipping the
    # boundary-integral velocity to its outflow-only part (>=0) removes
    # that sign-indefinite contribution; the interior advection term
    # (-inner(F_conv_w, grad(v_cw))*dx / the CgO2 analogue) is untouched.
    F_pres = (kappa * ufl.inner(ufl.grad(p_sol), ufl.grad(v_p)) * dx
              - (N_w_expr - N_O2_expr) * v_p * ds_meas(TAG_MEMBRANE))
    F_water = (Diff_w * ufl.inner(ufl.grad(Cw_sol), ufl.grad(v_cw)) * dx
               - ufl.inner(F_conv_w, ufl.grad(v_cw)) * dx
               + _smooth_outflow(ufl.inner(F_conv_w, nvec)) * v_cw * ds_meas(TAG_OUTLET)
               - N_w_expr * v_cw * ds_meas(TAG_MEMBRANE))
    F_oxy = (-CgO2_sol * ufl.inner(W_O2, ufl.grad(v_cgo2)) * dx
             + Deff_O2 * ufl.inner(ufl.grad(CgO2_sol), ufl.grad(v_cgo2)) * dx
             + CgO2_sol * ufl.inner(Jl_vec, ufl.grad(v_cgo2)) * dx
             + CgO2_sol * _smooth_outflow(ufl.inner(W_O2, nvec)) * v_cgo2 * ds_meas(TAG_OUTLET)
             + N_O2_expr * v_cgo2 * ds_meas(TAG_MEMBRANE))
    F_total = F_pres + F_water + F_oxy

    if verbose:
        print("  [building forms + JIT-compiling F, J, dF/deta -- can take a while]", flush=True)
    _progress_log("building forms for arclength continuation")

    fc_opts = {"quadrature_degree": 4}
    F_form = fem.form(F_total, form_compiler_options=fc_opts)
    J_ufl = ufl.derivative(F_total, w)
    J_form = fem.form(J_ufl, form_compiler_options=fc_opts)
    # Note: ufl.derivative(F_total, eta_const, ...) is NOT supported by UFL
    # for a dolfinx Constant ("Invalid coefficient type" -- confirmed by
    # direct testing, not just a missing-direction-argument issue). dF/deta
    # is instead computed via finite differencing below (assemble_Feta_vec),
    # since eta enters F_total only through eta_const in the smooth Tafel
    # exponential -- a well-behaved quantity to finite-difference.

    snes_petsc_options = {
        "snes_type": "newtonls", "snes_rtol": newton_rtol, "snes_atol": newton_atol,
        "snes_max_it": newton_max_it, "snes_error_if_not_converged": False,
        "ksp_type": "preonly", "pc_type": "lu",
    }
    bootstrap_problem = NonlinearProblem(F_total, w, bcs=bcs, petsc_options=snes_petsc_options,
                                          petsc_options_prefix="pemfc_arclen_boot_",
                                          form_compiler_options=fc_opts)
    # Separate, much more conservative SNES config used ONLY for the
    # refresh_tangent bootstrap (small snes_max_it): that solve starts
    # from an already-good warm start and only needs to move by a tiny
    # eta perturbation, so it should converge in just a few iterations.
    # Confirmed happening in practice: with the SAME generous iteration
    # budget as the from-scratch initial bootstrap, a refresh solve can
    # wander away from the warm start's branch entirely and converge to a
    # different, nearby solution instead (s_max collapsing from 0.306 to
    # 0.012 over a eta step of only 1.5e-4) -- capping iterations here
    # keeps the solve local to the warm start.
    refresh_snes_petsc_options = dict(snes_petsc_options)
    refresh_snes_petsc_options["snes_max_it"] = 8
    refresh_bootstrap_problem = NonlinearProblem(
        F_total, w, bcs=bcs, petsc_options=refresh_snes_petsc_options,
        petsc_options_prefix="pemfc_arclen_refresh_boot_", form_compiler_options=fc_opts)
    if verbose:
        print("  [JIT compilation done]", flush=True)
    _progress_log("JIT compilation done")

    def assemble_F_vec(x_arr, eta_val):
        w.x.array[:] = x_arr
        w.x.scatter_forward()
        eta_const.value = eta_val
        b = fem.petsc.assemble_vector(F_form)
        fem.petsc.apply_lifting(b, [J_form], bcs=[bcs], x0=[w.x.petsc_vec], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, bcs, w.x.petsc_vec, -1.0)
        return b

    def assemble_J_mat():
        A = fem.petsc.assemble_matrix(J_form, bcs=bcs)
        A.assemble()
        return A

    def assemble_Feta_vec(x_arr, eta_val, fd_eps=1e-6):
        """Central-difference approximation of dF/deta at (x_arr, eta_val).
        eta enters F_total only through eta_const in the smooth Tafel
        exponential, so this is a well-behaved quantity to finite-
        difference (no UFL symbolic differentiation w.r.t. a Constant
        needed -- confirmed unsupported by this UFL version)."""
        b_plus = assemble_F_vec(x_arr, eta_val + fd_eps)
        b_minus = assemble_F_vec(x_arr, eta_val - fd_eps)
        b_plus.axpy(-1.0, b_minus)          # b_plus <- b_plus - b_minus
        b_plus.scale(1.0 / (2.0 * fd_eps))  # central difference
        b_minus.destroy()
        arr = b_plus.array_w
        arr[bc_dofs_all] = 0.0  # dF/deta is trivially 0 at Dirichlet dofs (BC values don't depend on eta)
        return b_plus

    def lu_solve(A_mat, rhs_vec):
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A_mat)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        x_vec = A_mat.createVecRight()
        ksp.solve(rhs_vec, x_vec)
        ksp.destroy()
        return x_vec

    def I_avg_at(x_arr, eta_val):
        w.x.array[:] = x_arr
        w.x.scatter_forward()
        eta_const.value = eta_val
        Jloc_arr = _project_scalar_standalone(I_local, V, dx)
        return float(np.mean(Jloc_arr[membrane_dofs_V]))

    def s_max_at(x_arr):
        w.x.array[:] = x_arr
        w.x.scatter_forward()
        s_arr = _project_scalar_standalone(s, V, dx)
        return float(s_arr.max())

    def svd_diag_at(x_arr, eta_val):
        """Lightweight Jacobian SVD diagnostic at (x_arr, eta_val): just
        the row-norm-equilibrated smallest/largest singular values and
        condition number -- no reference-point comparison, no null
        vectors (see diagnose_fold_point() for the full version). Cheap
        enough to call periodically (every svd_diagnostic_every accepted
        steps) to build a Fig. 3-style dataset covering the WHOLE traced
        branch, not just a one-off pre-fold sweep."""
        import scipy.sparse as sp
        w.x.array[:] = x_arr
        w.x.scatter_forward()
        eta_const.value = eta_val
        J_mat_local = assemble_J_mat()
        n = J_mat_local.getSize()[0]
        indptr, indices, data = J_mat_local.getValuesCSR()
        J_dense = sp.csr_matrix((data, indices, indptr), shape=(n, n)).toarray()
        row_norms = np.sqrt(np.sum(J_dense**2, axis=1))
        scale = 1.0 / np.sqrt(np.maximum(row_norms, 1e-30))
        J_scaled = (J_dense * scale[None, :]) * scale[:, None]
        svals = np.linalg.svd(J_scaled, compute_uv=False)
        sv_min, sv_max = float(svals.min()), float(svals.max())
        J_mat_local.destroy()
        return sv_min, sv_max, sv_max / max(sv_min, 1e-300)

    _snapshot_coords = V.tabulate_dof_coordinates()[:, :2] if snapshot_dir else None

    # Periodic SVD diagnostic tracking (Fig. 3-style dataset covering the
    # whole branch, not just a one-off pre-fold sweep). Independent of
    # the main checkpoint file so it can be inspected/plotted without
    # touching continuation state; persists across resumed sessions by
    # loading whatever's already at svd_out_path.
    svd_eta_hist, svd_smallest_hist, svd_largest_hist, svd_cond_hist = [], [], [], []
    if svd_out_path and os.path.exists(svd_out_path):
        try:
            _svd_prior = np.load(svd_out_path)
            svd_eta_hist = list(_svd_prior["eta"])
            svd_smallest_hist = list(_svd_prior["sv_smallest"])
            svd_largest_hist = list(_svd_prior["sv_largest"])
            svd_cond_hist = list(_svd_prior["cond"])
            _progress_log(f"  loaded {len(svd_eta_hist)} prior SVD diagnostic point(s) "
                          f"from {svd_out_path}")
        except Exception as e:
            _progress_log(f"  could not load prior SVD diagnostics ({e}), starting fresh")

    def maybe_record_svd(x_arr, eta_val, accepted_step_count):
        if not svd_diagnostic_every or accepted_step_count % svd_diagnostic_every != 0:
            return
        sv_min, sv_max, cond = svd_diag_at(x_arr, eta_val)
        svd_eta_hist.append(eta_val)
        svd_smallest_hist.append(sv_min)
        svd_largest_hist.append(sv_max)
        svd_cond_hist.append(cond)
        _progress_log(f"  svd_diag: eta={eta_val:.5f} sv_min={sv_min:.4e} cond={cond:.4e}")
        if svd_out_path:
            try:
                np.savez(svd_out_path, eta=np.array(svd_eta_hist),
                         sv_smallest=np.array(svd_smallest_hist),
                         sv_largest=np.array(svd_largest_hist),
                         cond=np.array(svd_cond_hist))
            except Exception as e:
                _progress_log(f"  svd_diag save failed (continuing anyway): {e}")

    def save_snapshot_now(x_arr, eta_val, step_label, deta_sign=1.0):
        """Project and save the full Fig. 7 field set (s, p, Cw, CgO2,
        CgH2O water vapor, C_O2_mix oxygen mixture) for THIS accepted
        step, if snapshot_dir was provided. Assumes w is already set to
        (x_arr, eta_val) by the caller (I_avg_at/s_max_at just before
        this is always called that way in practice) -- re-sets it
        explicitly anyway to not depend on call order."""
        if snapshot_dir is None:
            return
        try:
            w.x.array[:] = x_arr
            w.x.scatter_forward()
            eta_const.value = eta_val
            s_arr = _project_scalar_standalone(s, V, dx)
            p_arr = _project_scalar_standalone(p_sol, V, dx)
            Cw_arr = _project_scalar_standalone(Cw_sol, V, dx)
            CgO2_arr = _project_scalar_standalone(CgO2_sol, V, dx)
            CgH2O_arr = _project_scalar_standalone(CgH2O_local, V, dx)
            CO2mix_arr = _project_scalar_standalone(C_O2_mix, V, dx)
            branch_state = _classify_branch_state(eta_val, float(s_arr.max()), deta_sign)
            fname = os.path.join(snapshot_dir, f"step_{step_label}_eta_{eta_val:.5f}.npz")
            np.savez(fname, coords=_snapshot_coords, eta=eta_val, s=s_arr, p=p_arr,
                     Cw=Cw_arr, CgO2=CgO2_arr, CgH2O=CgH2O_arr, C_O2_mix=CO2mix_arr,
                     branch_state=branch_state)
        except Exception as e:
            _progress_log(f"  snapshot save FAILED for step {step_label} (continuing anyway): {e}")

    def save_checkpoint(w_last, eta_last, dw_tan, deta_tan, eta_hist, I_hist,
                         smax_hist, ds_hist, cur_ds):
        if not (np.all(np.isfinite(w_last)) and np.isfinite(eta_last)
                and np.all(np.isfinite(dw_tan)) and np.isfinite(deta_tan)):
            _progress_log("  REFUSING to save checkpoint: state contains NaN/Inf -- "
                          "keeping the last good checkpoint on disk instead")
            return
        branch_state = _classify_branch_state(eta_last, float(smax_hist[-1]) if smax_hist else 0.0,
                                               deta_tan)
        try:
            np.savez(checkpoint_path, w_last=w_last, eta_last=eta_last, dw_tan=dw_tan,
                     deta_tan=deta_tan, eta_hist=np.array(eta_hist), I_hist=np.array(I_hist),
                     smax_hist=np.array(smax_hist), ds_hist=np.array(ds_hist), cur_ds=cur_ds,
                     branch_state=branch_state)
        except OSError as e:
            _progress_log(f"  checkpoint save failed (continuing anyway): {e}")

    resumed = False
    if resume and os.path.exists(checkpoint_path):
        try:
            ckpt = np.load(checkpoint_path)
            w_last = ckpt["w_last"]
            eta_last = float(ckpt["eta_last"])
            dw_tan = ckpt["dw_tan"]
            deta_tan = float(ckpt["deta_tan"])
            eta_hist = list(ckpt["eta_hist"])
            I_hist = list(ckpt["I_hist"])
            smax_hist = list(ckpt["smax_hist"])
            ds_hist = list(ckpt["ds_hist"])
            cur_ds = float(ckpt["cur_ds"])
            conv_hist = [True] * len(eta_hist)
            resumed = True
            if "branch_state" in ckpt.files:
                branch_state = str(ckpt["branch_state"])
            else:
                branch_state = _classify_branch_state(eta_last, smax_hist[-1] if smax_hist else 0.0,
                                                       deta_tan)
            _progress_log(f"RESUMED from checkpoint '{checkpoint_path}': "
                          f"{len(eta_hist)} prior step(s), last eta={eta_last:.5f}, "
                          f"cur_ds={cur_ds:.5f}, branch_state=[{branch_state}]")
            if verbose:
                print(f"  [resumed from checkpoint: {len(eta_hist)} prior steps, "
                      f"last eta={eta_last:.5f}, branch_state=[{branch_state}]]", flush=True)

            if refresh_tangent:
                # Direction from recent history (same logic as before):
                # this branch can be heading in EITHER eta direction (e.g.
                # past a fold, tracing the back side where eta decreases).
                if len(eta_hist) >= 2 and eta_hist[-1] < eta_hist[-2]:
                    direction = -1.0
                else:
                    direction = 1.0

                # Recompute the tangent DIRECTLY from the Jacobian at the
                # CURRENT point (w_last, eta_last) -- no new nonlinear
                # solve at a perturbed eta. This replaces an earlier
                # design (two ordinary Newton solves at eta_last and
                # eta_last +/- a small step, then a secant) that turned
                # out to be fundamentally unsound: an independent test
                # (two different initial guesses solved at the same fixed
                # eta near a stuck point) confirmed the "true" eta-fixed
                # solution there has s_max~0.01, NOT the s_max~0.31 the
                # branch was actually tracing -- i.e. exactly the kind of
                # near-vertical s(eta) dependence that motivated using
                # arclength continuation in the first place. Fixing eta
                # and running ordinary Newton is the wrong tool in
                # precisely the regions where a tangent refresh is
                # needed, since those are the delicate regions to begin
                # with. A pure linearization at the current point cannot
                # jump to a different branch (it never solves a new
                # nonlinear problem), and is also cheaper (no extra
                # nonlinear solves at all).
                _progress_log(f"  refresh_tangent=True: recomputing tangent directly via "
                              f"Jacobian at eta={eta_last:.5f} (no new nonlinear solve), "
                              f"direction={direction:+.0f} from recent history")
                if verbose:
                    print(f"  [refreshing tangent directly via Jacobian at eta={eta_last:.5f}, "
                          f"direction={direction:+.0f}]", flush=True)

                F_vec = assemble_F_vec(w_last, eta_last)
                J_mat = assemble_J_mat()
                Feta_vec = assemble_Feta_vec(w_last, eta_last)
                Feta_vec.scale(-1.0)
                z = lu_solve(J_mat, Feta_vec)

                if not np.all(np.isfinite(z.array)):
                    J_mat.destroy(); z.destroy(); F_vec.destroy(); Feta_vec.destroy()
                    raise RuntimeError(
                        f"refresh_tangent: NaN/Inf in direct Jacobian tangent computation "
                        f"at eta={eta_last:.5f}. Checkpoint left UNCHANGED.")

                dw_tan_new = direction * z.array
                deta_tan_new = float(direction)
                norm_t = float(np.sqrt(np.sum(dw_tan_new**2) + deta_tan_new**2))
                if not (np.isfinite(norm_t) and norm_t > 1e-300):
                    J_mat.destroy(); z.destroy(); F_vec.destroy(); Feta_vec.destroy()
                    raise RuntimeError(
                        f"refresh_tangent: degenerate tangent norm ({norm_t}) at "
                        f"eta={eta_last:.5f}. Checkpoint left UNCHANGED.")
                dw_tan = dw_tan_new / norm_t
                deta_tan = deta_tan_new / norm_t
                J_mat.destroy(); z.destroy(); F_vec.destroy(); Feta_vec.destroy()

                # The point itself (w_last, eta_last) is UNCHANGED -- only
                # the tangent was recomputed -- so there's no new history
                # entry or snapshot to add here, just a checkpoint update
                # with the fresh tangent.
                cur_ds = min(cur_ds, ds)
                _progress_log(f"  refresh done: tangent recomputed at eta={eta_last:.5f} "
                              f"(point unchanged)")
                save_checkpoint(w_last, eta_last, dw_tan, deta_tan, eta_hist, I_hist,
                                smax_hist, ds_hist, cur_ds)
        except Exception as e:
            _progress_log(f"  checkpoint load failed ({e}), starting fresh instead")
            resumed = False

    if not resumed:
        # --- bootstrap: two ordinary (SNES) solves to get a starting secant tangent ---
        eta_const.value = eta_start
        if init_w_array is not None:
            # Warm-start from an existing solution (e.g. the main run's
            # current state, reused for a separate segment targeting the
            # same or a nearby eta) instead of a naive/inlet-like guess --
            # skips the warmup ramp entirely below, since we're already
            # starting from a state that's presumably close to converged
            # somewhere in the easy, near-single-phase regime.
            w.x.array[:] = init_w_array
            w.x.scatter_forward()
            _progress_log(f"bootstrap: warm-starting from a supplied init_w_array "
                          f"at eta={eta_start} (skipping the naive-guess warmup ramp)")
        else:
            w.sub(0).interpolate(lambda x: np.full(x.shape[1], (p_in - p_out) / P_scale))
            w.sub(1).interpolate(lambda x: np.full(x.shape[1], Y_H2O_in))
            w.sub(2).interpolate(lambda x: np.full(x.shape[1], Y_O2_in))
            w.x.scatter_forward()

        # Warm-up ramp: solving directly at eta_start (default 0.30) from
        # this naive/inlet-like guess is NOT reliable on its own (confirmed:
        # SNES fails to converge, leaving w in a divergent state with Cw
        # far outside its physical [0,1] range) -- the same issue already
        # found and fixed in fold_svd_sweep(). Ramp up gradually from a
        # much smaller eta first, exactly like that fix, before the actual
        # two-point secant bootstrap below. Skipped entirely if warm-
        # starting from an existing solution (init_w_array) instead.
        if init_w_array is None and eta_start > 0.05:
            warmup = list(np.linspace(0.02, eta_start, 8))[:-1]
            _progress_log(f"bootstrap: warming up from eta=0.02 to eta={eta_start} "
                          f"before the main bootstrap...")
            for eta_now in warmup:
                eta_const.value = eta_now
                bootstrap_problem.solve()
                if bootstrap_problem.solver.getConvergedReason() <= 0:
                    raise RuntimeError(
                        f"run_case_arclength: bootstrap warm-up solve failed to converge "
                        f"at eta={eta_now:.4f} -- cannot safely bootstrap the main "
                        f"continuation from a divergent state.")

        _progress_log(f"bootstrap solve 1 at eta={eta_start}")
        bootstrap_problem.solve()
        if bootstrap_problem.solver.getConvergedReason() <= 0:
            raise RuntimeError(
                f"run_case_arclength: bootstrap solve 1 failed to converge at "
                f"eta={eta_start} even after warm-up. Checkpoint left UNCHANGED.")
        w0_arr = w.x.array.copy()
        eta0 = eta_start

        _progress_log(f"bootstrap solve 2 at eta={eta_start + ds}")
        eta_const.value = eta_start + ds
        bootstrap_problem.solve()
        if bootstrap_problem.solver.getConvergedReason() <= 0:
            raise RuntimeError(
                f"run_case_arclength: bootstrap solve 2 failed to converge at "
                f"eta={eta_start + ds}. Checkpoint left UNCHANGED.")
        w1_arr = w.x.array.copy()
        eta1 = eta_start + ds

        dw_tan = w1_arr - w0_arr
        deta_tan = eta1 - eta0
        secant_len = float(np.sqrt(np.sum(dw_tan**2) + deta_tan**2))
        dw_tan = dw_tan / secant_len
        deta_tan = deta_tan / secant_len
        _progress_log(f"bootstrap done: eta0={eta0}, eta1={eta1}, |secant|={secant_len:.4e}")

        eta_hist = [eta0, eta1]
        I_hist = [I_avg_at(w0_arr, eta0), I_avg_at(w1_arr, eta1)]
        smax_hist = [s_max_at(w0_arr), s_max_at(w1_arr)]
        ds_hist = [0.0, ds]
        conv_hist = [True, True]
        cur_ds = ds
        save_snapshot_now(w0_arr, eta0, "boot0", deta_sign=1.0)
        save_snapshot_now(w1_arr, eta1, "boot1", deta_sign=1.0)
        maybe_record_svd(w0_arr, eta0, 1)
        maybe_record_svd(w1_arr, eta1, 2)

        w_last, eta_last = w1_arr, eta1
        save_checkpoint(w_last, eta_last, dw_tan, deta_tan, eta_hist, I_hist,
                         smax_hist, ds_hist, cur_ds)

    for step in range(n_steps):
        w_pred = w_last + cur_ds * dw_tan
        eta_pred = eta_last + cur_ds * deta_tan
        _progress_log(f"step {step}: predictor eta={eta_pred:.5f}  ds={cur_ds:.5f}")

        x, eta_try = w_pred.copy(), eta_pred
        converged = False
        for it in range(max_newton_it):
            F_vec = assemble_F_vec(x, eta_try)
            J_mat = assemble_J_mat()
            Feta_vec = assemble_Feta_vec(x, eta_try)

            F_vec.scale(-1.0)
            v1 = lu_solve(J_mat, F_vec)
            Feta_vec.scale(-1.0)
            v2 = lu_solve(J_mat, Feta_vec)

            if not (np.all(np.isfinite(v1.array)) and np.all(np.isfinite(v2.array))):
                _progress_log(f"  step {step} it {it}: NaN/Inf in linear solve "
                              f"(Jacobian likely singular here too) -- treating as failed corrector")
                J_mat.destroy(); v1.destroy(); v2.destroy(); F_vec.destroy(); Feta_vec.destroy()
                break

            N_val = float(np.dot(dw_tan, x - w_last) + deta_tan * (eta_try - eta_last) - cur_ds)
            dw_dot_v1 = float(np.dot(dw_tan, v1.array))
            dw_dot_v2 = float(np.dot(dw_tan, v2.array))
            denom = deta_tan + dw_dot_v2
            if abs(denom) < 1e-300:
                _progress_log(f"  step {step} it {it}: bordering denom ~0, aborting corrector")
                J_mat.destroy(); v1.destroy(); v2.destroy(); F_vec.destroy(); Feta_vec.destroy()
                break
            delta_eta = (-N_val - dw_dot_v1) / denom
            delta_x = v1.array + delta_eta * v2.array

            if not (np.all(np.isfinite(delta_x)) and np.isfinite(delta_eta)):
                _progress_log(f"  step {step} it {it}: NaN/Inf in computed update -- "
                              f"treating as failed corrector")
                J_mat.destroy(); v1.destroy(); v2.destroy(); F_vec.destroy(); Feta_vec.destroy()
                break

            x = x + delta_x
            eta_try = eta_try + delta_eta
            res_norm = float(np.sqrt(np.sum(delta_x**2) + delta_eta**2))
            J_mat.destroy(); v1.destroy(); v2.destroy(); F_vec.destroy(); Feta_vec.destroy()
            if res_norm < newton_tol:
                converged = True
                _progress_log(f"  step {step}: corrector converged in {it+1} it, "
                              f"eta={eta_try:.5f}")
                break
        if not converged:
            cur_ds *= 0.5
            _progress_log(f"  step {step}: corrector FAILED, halving ds to {cur_ds:.5f}")
            if cur_ds < ds_min:
                _progress_log("  ds below ds_min, stopping continuation")
                break
            continue

        I_now = I_avg_at(x, eta_try)
        smax_now = s_max_at(x)

        # Branch-jump sanity check (same threshold as the refresh
        # bootstrap's own check): a converged Newton step can still land
        # on a genuinely different, disconnected solution if multiple
        # solutions coexist near this eta -- residual convergence alone
        # doesn't guarantee it's still on the branch being traced.
        # Confirmed to happen in practice: s_max jumped from 0.306 to
        # 0.011 over a single accepted step with eta moving by only
        # ~0.005, right after an outlet-BC stabilization fix widened
        # Newton's basin of attraction enough to make a nearby, simpler
        # (near-single-phase) solution reachable in one step.
        smax_prev = smax_hist[-1] if smax_hist else smax_now
        threshold = 0.05 + 0.1 * smax_prev
        _progress_log(f"  step {step}: branch-jump check: smax_prev={smax_prev:.4f} "
                      f"smax_now={smax_now:.4f} diff={abs(smax_now-smax_prev):.4f} "
                      f"threshold={threshold:.4f} len(smax_hist)={len(smax_hist)}")
        if abs(smax_now - smax_prev) > threshold:
            _progress_log(f"  step {step}: REJECTED as likely branch jump "
                          f"(s_max {smax_prev:.3f} -> {smax_now:.3f} over a single step, "
                          f"eta {eta_last:.5f} -> {eta_try:.5f}) -- halving ds and retrying")
            cur_ds *= 0.5
            if cur_ds < ds_min:
                _progress_log("  ds below ds_min, stopping continuation")
                break
            continue

        eta_hist.append(eta_try); I_hist.append(I_now); smax_hist.append(smax_now)
        ds_hist.append(cur_ds); conv_hist.append(True)
        save_snapshot_now(x, eta_try, f"{len(eta_hist):05d}", deta_sign=deta_tan)
        maybe_record_svd(x, eta_try, len(eta_hist))
        branch_state_now = _classify_branch_state(eta_try, smax_now, deta_tan)
        step_summary = (f"  [arclength] step={step}  eta={eta_try:.4f}  I={I_now/1e4:.4f} A/cm^2  "
                         f"s_max={smax_now:.3f}  ds={cur_ds:.4f}  branch_state=[{branch_state_now}]")
        _progress_log(step_summary)
        if verbose:
            print(step_summary, flush=True)

        # tangent update at the new point
        F_vec = assemble_F_vec(x, eta_try)  # also sets w to (x, eta_try) as a side effect
        J_mat = assemble_J_mat()
        Feta_vec = assemble_Feta_vec(x, eta_try)
        Feta_vec.scale(-1.0)
        z = lu_solve(J_mat, Feta_vec)
        if not np.all(np.isfinite(z.array)):
            _progress_log(f"  step {step}: NaN/Inf in tangent-update solve -- "
                          f"keeping previous tangent unchanged")
        else:
            dw_dot_z = float(np.dot(dw_tan, z.array))
            denom_t = deta_tan + dw_dot_z
            deta_new = 1.0 / denom_t if abs(denom_t) > 1e-300 else deta_tan
            dw_new = deta_new * z.array
            norm_t = float(np.sqrt(np.sum(dw_new**2) + deta_new**2))
            if np.isfinite(norm_t) and norm_t > 1e-300:
                dw_new, deta_new = dw_new / norm_t, deta_new / norm_t
                if np.all(np.isfinite(dw_new)) and np.isfinite(deta_new):
                    if dw_new @ dw_tan + deta_new * deta_tan < 0:
                        dw_new, deta_new = -dw_new, -deta_new
                    dw_tan, deta_tan = dw_new, deta_new
                else:
                    _progress_log(f"  step {step}: normalized tangent has NaN/Inf -- "
                                  f"keeping previous tangent unchanged")
            else:
                _progress_log(f"  step {step}: degenerate tangent norm ({norm_t}) -- "
                              f"keeping previous tangent unchanged")
        J_mat.destroy(); z.destroy(); F_vec.destroy(); Feta_vec.destroy()

        w_last, eta_last = x, eta_try
        cur_ds = min(cur_ds * 1.2, ds_max)
        save_checkpoint(w_last, eta_last, dw_tan, deta_tan, eta_hist, I_hist,
                         smax_hist, ds_hist, cur_ds)

    return dict(eta=np.array(eta_hist), I_avg=np.array(I_hist), s_max=np.array(smax_hist),
                ds_used=np.array(ds_hist), converged=np.array(conv_hist), params=P)


_project_scalar_call_counter = [0]


def _project_scalar_standalone(expr, V, dx):
    """Standalone L2 projection helper for run_case_arclength (which
    doesn't have run_case's nested project_scalar closure available).
    Uses a fresh, unique PETSc options prefix every call (via a simple
    counter) -- reusing the same fixed prefix across many rapid calls
    was suspected of allowing PETSc-level state to leak between calls
    (confirmed symptom: s_max_at() appeared to return a value from a
    DIFFERENT, previous state than the one just requested, e.g. right
    after a tiny eta perturbation that should have left s_max nearly
    unchanged)."""
    _project_scalar_call_counter[0] += 1
    prefix = f"pemfc_arclen_proj_{_project_scalar_call_counter[0]}_"
    u_tr, v_te = ufl.TrialFunction(V), ufl.TestFunction(V)
    a_p = u_tr * v_te * dx
    L_p = expr * v_te * dx
    prob = LinearProblem(a_p, L_p, bcs=[],
                          petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                          petsc_options_prefix=prefix)
    return prob.solve().x.array.copy()


def diagnose_fold_point(checkpoint_path, overrides=None, reference_eta=0.30,
                         n_smallest=10, verbose=True, save_nullvec_path=None):
    """Diagnose the local bifurcation structure AT the point currently
    saved in checkpoint_path (from run_case_arclength), to distinguish a
    genuine SIMPLE FOLD (limit point) -- which pseudo-arclength
    continuation is designed to trace straight through -- from a higher-
    codimension degeneracy (a bifurcation point, where a second branch
    crosses the one being traced, or a cusp, where two folds coincide).
    Motivated by run_case_arclength stalling for 90+ steps at a fixed eta
    with no improving trend in Newton iteration counts, unlike the FIRST
    fold (near eta~0.36 for the base case), which showed a clear,
    monotonically-improving iteration-count trend and broke through in
    ~6 step-halvings. That qualitative difference is the reason to
    suspect this second stuck point is NOT a simple fold.

    METHODOLOGY (revised after an earlier version of this function gave
    an unreliable result): the Jacobian's singular value spectrum is
    computed at the stuck point AND at a fresh solve of a KNOWN-REGULAR
    reference point (`reference_eta`, default 0.30 -- comfortably far
    from any fold), using row-norm (not diagonal/Jacobi) equilibration,
    and the two are compared directly:

    1. CONDITION NUMBER RATIO (stuck / reference): a genuine singularity
       shows up as the stuck point being MANY orders of magnitude worse-
       conditioned than an ordinary point; a mild or absent gap says the
       stall is more likely algorithmic than mathematical. An earlier
       version of this function instead counted singular values below an
       absolute threshold (1e-3 x the largest) -- this was unreliable:
       FEM discretizations of elliptic-type PDEs naturally have singular
       value spectra spanning many orders of magnitude even at perfectly
       regular points (mesh-induced conditioning, unrelated to any
       bifurcation), so an uncalibrated absolute cutoff flagged most of
       the ~4500-dof spectrum as "small" regardless of what was actually
       happening. Comparing against a live reference point removes that
       problem. Row-norm scaling (rather than pure diagonal/Jacobi
       scaling) was also substituted for the same reason: diagonal
       scaling blows up if even a single dof has a near-zero DIAGONAL
       entry (not inherently meaningful -- can happen in an advection-
       dominated or weakly-coupled row), inflating that dof's entire
       row/column and producing a spuriously wide spectrum unrelated to
       genuine bifurcation structure; row-norm scaling only breaks down
       if a dof's entire row vanishes, a much stronger condition.

    2. TRANSVERSALITY: phi^T F_eta, where phi is the LEFT null vector
       (the left singular vector paired with the smallest singular
       value, mapped back through the row-norm scaling) and F_eta =
       dF/deta (via finite differencing, as in run_case_arclength). At a
       genuine simple fold this is transverse (bounded away from zero);
       collapsing toward zero indicates something more degenerate even
       if the conditioning gap from (1) alone looked fold-like.

    Returns a dict with the raw diagnostics (singular values, the
    transversality quantity) and a plain-language verdict string.
    """
    from petsc4py import PETSc
    import scipy.sparse as sp

    P = dict(DEFAULT_PARAMS)
    if overrides:
        P.update(overrides)

    F, R = P["F"], P["R"]
    K_perm, eps_p = P["K_perm"], P["eps_p"]
    p_in, p_out = P["p_in"], P["p_out"]
    mu_g, D_O2_g, D_H2O_g = P["mu_g"], P["D_O2_g"], P["D_H2O_g"]
    I0_ref, alpha_c, alpha_w = P["I0"], P["alpha_c"], P["alpha_w"]
    T_ref_I0, Ea_orr = P["T_ref_I0"], P["Ea_orr"]
    rho_l = P["rho_l"]
    M_H2O, M_O2, M_N2 = P["M_H2O"], P["M_O2"], P["M_N2"]
    mu_l = P["mu_l"]
    T_cell, C_O2_ref = P["T_cell"], P["C_O2_ref"]
    I0 = I0_ref * np.exp(-(Ea_orr / R) * (1.0 / T_cell - 1.0 / T_ref_I0))
    RH_in = P["RH_in"]
    theta_c_deg = P["theta_c_deg"]
    theta_c = theta_c_deg * np.pi / 180.0
    sigma_st = P["sigma_st"]
    X_O2_in, X_N2_in = P["X_O2_in"], P["X_N2_in"]
    w_ch, w_rb, Lx = P["w_ch"], P["w_rb"], P["Lx"]
    n_corey, tau_brug, s_max_cap = P["n_corey"], P["tau_brug"], P["s_max"]
    s_smooth_eps = P["s_smooth_eps"]
    single_phase = P["single_phase"]
    nx, ny = P["nx"], P["ny"]

    P_sat = Psat_atm(T_cell - 273.15) * 101325.0
    Ly = w_ch + w_rb
    y_inlet_end = 0.5 * w_ch
    y_rib_end = 0.5 * w_ch + w_rb

    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [Lx, Ly]], [nx, ny], mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    def on_membrane(x): return np.isclose(x[0], 0.0)
    def on_inlet(x):    return np.isclose(x[0], Lx) & (x[1] <= y_inlet_end + 1e-12)
    def on_rib(x):      return np.isclose(x[0], Lx) & (x[1] > y_inlet_end + 1e-12) & (x[1] <= y_rib_end + 1e-12)
    def on_outlet(x):   return np.isclose(x[0], Lx) & (x[1] > y_rib_end + 1e-12)

    TAG_MEMBRANE, TAG_INLET, TAG_RIB, TAG_OUTLET = 1, 2, 3, 4
    facet_indices, facet_markers = [], []
    for tag, locator in [(TAG_MEMBRANE, on_membrane), (TAG_INLET, on_inlet),
                          (TAG_RIB, on_rib), (TAG_OUTLET, on_outlet)]:
        idx = mesh.locate_entities_boundary(domain, fdim, locator)
        facet_indices.append(idx)
        facet_markers.append(np.full_like(idx, tag))
    facet_indices = np.concatenate(facet_indices)
    facet_markers = np.concatenate(facet_markers)
    order = np.argsort(facet_indices)
    facet_tags = mesh.meshtags(domain, fdim, facet_indices[order], facet_markers[order])

    ds_meas = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    dx = ufl.Measure("dx", domain=domain)
    nvec = ufl.FacetNormal(domain)

    V = fem.functionspace(domain, ("Lagrange", 1))
    P1e = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    ME = basix.ufl.mixed_element([P1e, P1e, P1e])
    W = fem.functionspace(domain, ME)

    w = fem.Function(W, name="w")
    v_te = ufl.TestFunction(W)
    p_hat_sol, Cw_sol, CgO2_sol = ufl.split(w)
    v_p, v_cw, v_cgo2 = ufl.split(v_te)

    P_scale = 1000.0
    p_sol = p_out + P_scale * p_hat_sol

    X_H2O_in = np.clip(RH_in * P_sat / p_in, 0.0, 0.98)
    dry_scale = 1.0 - X_H2O_in
    X_O2_eff = X_O2_in * dry_scale
    X_N2_eff = X_N2_in * dry_scale
    M_mix_in = X_O2_eff * M_O2 + X_N2_eff * M_N2 + X_H2O_in * M_H2O
    Y_O2_in = X_O2_eff * M_O2 / M_mix_in
    Y_H2O_in = X_H2O_in * M_H2O / M_mix_in
    M_dry_in = X_O2_in * M_O2 + X_N2_in * M_N2

    rho_g_const = p_out * M_mix_in / (R * T_cell)
    Cg_sat = (P_sat * M_H2O) / (P_sat * M_H2O + (p_in - P_sat) * M_dry_in)
    Cl_sat = 1.0

    inlet_facets = facet_tags.find(TAG_INLET)
    outlet_facets = facet_tags.find(TAG_OUTLET)
    dofs_p_in = fem.locate_dofs_topological(W.sub(0), fdim, inlet_facets)
    dofs_p_out = fem.locate_dofs_topological(W.sub(0), fdim, outlet_facets)
    dofs_cw_in = fem.locate_dofs_topological(W.sub(1), fdim, inlet_facets)
    dofs_cgo2_in = fem.locate_dofs_topological(W.sub(2), fdim, inlet_facets)
    bcs = [
        fem.dirichletbc(default_scalar_type((p_in - p_out) / P_scale), dofs_p_in, W.sub(0)),
        fem.dirichletbc(default_scalar_type(0.0), dofs_p_out, W.sub(0)),
        fem.dirichletbc(default_scalar_type(Y_H2O_in), dofs_cw_in, W.sub(1)),
        fem.dirichletbc(default_scalar_type(Y_O2_in), dofs_cgo2_in, W.sub(2)),
    ]
    bc_dofs_all = np.concatenate([dofs_p_in, dofs_p_out, dofs_cw_in, dofs_cgo2_in])

    def s_expr(Cw):
        if single_phase:
            return 0.0 * Cw
        raw = rho_g_const * (Cw - Cg_sat) / (rho_l * (Cl_sat - Cw) + rho_g_const * (Cw - Cg_sat) + 1e-12)
        s_smooth = 0.5 * (raw + ufl.sqrt(raw**2 + s_smooth_eps**2))
        s_smooth = ufl.max_value(s_smooth, 0.0)
        return ufl.min_value(s_smooth, s_max_cap)

    def min_smooth(a, b, eps=s_smooth_eps):
        return 0.5 * (a + b - ufl.sqrt((a - b) ** 2 + eps ** 2))

    hydrophobic = theta_c_deg >= 90.0
    s = s_expr(Cw_sol)
    krl = s ** n_corey
    krg = (1.0 - s) ** n_corey
    lam_l = (krl / mu_l) / (krl / mu_l + krg / mu_g + 1e-30)
    lam_g = 1.0 - lam_l
    rho_mix = rho_g_const * (1.0 - s) + rho_l * s
    nu_g_c, nu_l_c = mu_g / rho_g_const, mu_l / rho_l
    nu_mix = 1.0 / (krg / nu_g_c + krl / nu_l_c + 1e-30)
    mu_mix = nu_mix * rho_mix
    kappa = rho_mix * K_perm / mu_mix

    if hydrophobic:
        dJds = 1.417 - 4.240 * s + 3.789 * s**2
    else:
        u_hp = 1.0 - s
        dJds = -(1.417 - 4.240 * u_hp + 3.789 * u_hp**2)

    Dc = ((lam_l * lam_g * K_perm / (nu_mix + 1e-30))
          * sigma_st * np.cos(theta_c) * (eps_p / K_perm) ** 0.5 * dJds)
    Gamma = Dc * (1.0 - Cg_sat) / (rho_l - rho_g_const * Cg_sat)
    grads = ufl.grad(s)
    u_darcy = -(kappa / rho_mix) * ufl.grad(p_sol)
    CgH2O_local = min_smooth(Cw_sol, Cg_sat)
    Weff_w = lam_l + lam_g * CgH2O_local
    Deff_O2 = rho_g_const * (eps_p * (1.0 - s)) ** tau_brug * D_O2_g
    Diff_w = rho_g_const * (eps_p * (1.0 - s)) ** tau_brug * D_H2O_g + Gamma * rho_mix
    Jl_vec = Dc * grads
    F_conv_w = Weff_w * rho_mix * u_darcy
    W_O2 = rho_mix * lam_g * u_darcy

    eta_const = fem.Constant(domain, default_scalar_type(0.0))
    C_O2_molar = rho_g_const * CgO2_sol / M_O2
    I_local = (1.0 - s) * I0 * (C_O2_molar / C_O2_ref) * ufl.exp(alpha_c * F * eta_const / (R * T_cell))
    N_O2_expr = (M_O2 / (4.0 * F)) * I_local
    N_w_expr = (M_H2O * (1.0 + 2.0 * alpha_w) / (2.0 * F)) * I_local

    F_pres = (kappa * ufl.inner(ufl.grad(p_sol), ufl.grad(v_p)) * dx
              - (N_w_expr - N_O2_expr) * v_p * ds_meas(TAG_MEMBRANE))
    F_water = (Diff_w * ufl.inner(ufl.grad(Cw_sol), ufl.grad(v_cw)) * dx
               - ufl.inner(F_conv_w, ufl.grad(v_cw)) * dx
               + _smooth_outflow(ufl.inner(F_conv_w, nvec)) * v_cw * ds_meas(TAG_OUTLET)
               - N_w_expr * v_cw * ds_meas(TAG_MEMBRANE))
    F_oxy = (-CgO2_sol * ufl.inner(W_O2, ufl.grad(v_cgo2)) * dx
             + Deff_O2 * ufl.inner(ufl.grad(CgO2_sol), ufl.grad(v_cgo2)) * dx
             + CgO2_sol * ufl.inner(Jl_vec, ufl.grad(v_cgo2)) * dx
             + CgO2_sol * _smooth_outflow(ufl.inner(W_O2, nvec)) * v_cgo2 * ds_meas(TAG_OUTLET)
             + N_O2_expr * v_cgo2 * ds_meas(TAG_MEMBRANE))
    F_total = F_pres + F_water + F_oxy

    fc_opts = {"quadrature_degree": 4}
    F_form = fem.form(F_total, form_compiler_options=fc_opts)
    J_ufl = ufl.derivative(F_total, w)
    J_form = fem.form(J_ufl, form_compiler_options=fc_opts)

    def assemble_F_vec(x_arr, eta_val):
        w.x.array[:] = x_arr
        w.x.scatter_forward()
        eta_const.value = eta_val
        b = fem.petsc.assemble_vector(F_form)
        fem.petsc.apply_lifting(b, [J_form], bcs=[bcs], x0=[w.x.petsc_vec], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, bcs, w.x.petsc_vec, -1.0)
        return b

    def assemble_J_mat():
        A = fem.petsc.assemble_matrix(J_form, bcs=bcs)
        A.assemble()
        return A

    def assemble_Feta_vec(x_arr, eta_val, fd_eps=1e-6):
        b_plus = assemble_F_vec(x_arr, eta_val + fd_eps)
        b_minus = assemble_F_vec(x_arr, eta_val - fd_eps)
        b_plus.axpy(-1.0, b_minus)
        b_plus.scale(1.0 / (2.0 * fd_eps))
        b_minus.destroy()
        arr = b_plus.array_w
        arr[bc_dofs_all] = 0.0
        return b_plus

    def analyze_point(x_arr, eta_val, label):
        """Assemble J and dF/deta at (x_arr, eta_val) and return its
        row-norm-equilibrated singular value spectrum plus the
        transversality quantity. Row-norm scaling (rescale each row/col
        by 1/sqrt(its L2 norm)) is used instead of pure Jacobi/diagonal
        scaling -- diagonal scaling blows up if even a single dof has a
        near-zero DIAGONAL entry (common in an advection-dominated or
        near-decoupled row), which artificially inflates that dof's
        entire row/column and can produce a spurious wide spectrum with
        no relation to genuine bifurcation structure. Row-norm scaling
        only breaks down if a dof's entire row is near-zero, which is a
        much stronger (and more meaningful) degeneracy condition."""
        F_vec = assemble_F_vec(x_arr, eta_val)
        J_mat = assemble_J_mat()
        Feta_vec = assemble_Feta_vec(x_arr, eta_val)

        n = J_mat.getSize()[0]
        indptr, indices, data = J_mat.getValuesCSR()
        J_dense = sp.csr_matrix((data, indices, indptr), shape=(n, n)).toarray()

        row_norms = np.sqrt(np.sum(J_dense**2, axis=1))
        scale = 1.0 / np.sqrt(np.maximum(row_norms, 1e-30))
        J_scaled = (J_dense * scale[None, :]) * scale[:, None]

        U, svals, Vt = np.linalg.svd(J_scaled)
        order_idx = np.argsort(svals)
        svals_sorted = svals[order_idx]

        idx_min = order_idx[0]
        phi_scaled = U[:, idx_min]
        phi = scale * phi_scaled
        phi = phi / np.linalg.norm(phi)
        # right null vector (solution-space direction the branch is
        # degenerate along) -- for row-norm scaling, J_scaled = diag(scale)
        # @ J @ diag(scale), so the RIGHT singular vector maps back the
        # same way as the left one (both sides scaled identically here,
        # unlike a genuinely asymmetric row/column scaling)
        psi_scaled = Vt[idx_min, :]
        psi = scale * psi_scaled
        psi = psi / np.linalg.norm(psi)

        Feta_arr = Feta_vec.array.copy()
        transversality_raw = float(np.dot(phi, Feta_arr))
        transversality_normalized = transversality_raw / (
            np.linalg.norm(phi) * np.linalg.norm(Feta_arr) + 1e-300)

        J_mat.destroy(); F_vec.destroy(); Feta_vec.destroy()

        if verbose:
            print(f"[{label}] eta={eta_val:.5f}  smallest {n_smallest} scaled svals: "
                  f"{np.array2string(svals_sorted[:n_smallest], precision=3)}", flush=True)
            print(f"[{label}]   largest={svals_sorted[-1]:.4g}  "
                  f"cond={svals_sorted[-1]/max(svals_sorted[0], 1e-300):.4e}  "
                  f"transversality(normalized)={transversality_normalized:.4e}", flush=True)

        return dict(eta=eta_val, svals_scaled=svals_sorted,
                    transversality_normalized=transversality_normalized,
                    cond=svals_sorted[-1] / max(svals_sorted[0], 1e-300),
                    phi=phi, psi=psi)

    # --- stuck point, from the checkpoint ---
    ckpt = np.load(checkpoint_path)
    w_stuck = ckpt["w_last"]
    eta_stuck = float(ckpt["eta_last"])
    _progress_log(f"diagnose_fold_point: loaded checkpoint, eta={eta_stuck:.5f}")
    if verbose:
        print(f"Diagnosing stuck point at eta={eta_stuck:.5f} (from {checkpoint_path})\n", flush=True)
    stuck = analyze_point(w_stuck, eta_stuck, "STUCK")

    if save_nullvec_path:
        # Split phi (left null vector -- which equation/residual is
        # poorly constrained) and psi (right null vector -- which
        # solution-space direction the branch is degenerate along, i.e.
        # roughly "what a fold-crossing perturbation would look like")
        # into their p_hat/Cw/CgO2 components and save with mesh
        # coordinates, so their SPATIAL structure can be plotted: if the
        # near-singular direction is concentrated in a small region of
        # the domain (e.g. near a developing saturation front), that
        # points at mesh resolution rather than a genuine bifurcation;
        # if it's spread across the whole domain, that's more consistent
        # with a true higher-codimension point.
        w_phi = fem.Function(W)
        w_phi.x.array[:] = stuck["phi"]
        w_psi = fem.Function(W)
        w_psi.x.array[:] = stuck["psi"]
        phi_p = w_phi.sub(0).collapse()
        phi_cw = w_phi.sub(1).collapse()
        phi_cgo2 = w_phi.sub(2).collapse()
        psi_p = w_psi.sub(0).collapse()
        psi_cw = w_psi.sub(1).collapse()
        psi_cgo2 = w_psi.sub(2).collapse()
        coords_p = phi_p.function_space.tabulate_dof_coordinates()[:, :2]
        coords_cw = phi_cw.function_space.tabulate_dof_coordinates()[:, :2]
        coords_cgo2 = phi_cgo2.function_space.tabulate_dof_coordinates()[:, :2]
        np.savez(save_nullvec_path, eta=eta_stuck,
                 coords_p=coords_p, coords_Cw=coords_cw, coords_CgO2=coords_cgo2,
                 phi_p=phi_p.x.array.copy(), phi_Cw=phi_cw.x.array.copy(),
                 phi_CgO2=phi_cgo2.x.array.copy(),
                 psi_p=psi_p.x.array.copy(), psi_Cw=psi_cw.x.array.copy(),
                 psi_CgO2=psi_cgo2.x.array.copy())
        if verbose:
            print(f"Saved null-vector spatial data to {save_nullvec_path}", flush=True)

    # --- reference point: a fresh ordinary solve at a KNOWN-regular eta,
    # analyzed with the exact same method, so the comparison is apples-
    # to-apples rather than judged against an uncalibrated absolute
    # threshold (an earlier version of this function did that and it was
    # unreliable -- FEM discretizations naturally have singular value
    # spectra spanning many orders of magnitude even at perfectly regular
    # points, so an absolute cutoff flags most of the spectrum as "small"
    # regardless of whether anything unusual is actually happening) ---
    if verbose:
        print(f"\nSolving reference point at eta={reference_eta} for comparison...", flush=True)
    snes_petsc_options = {
        "snes_type": "newtonls", "snes_rtol": P["newton_rtol"], "snes_atol": P["newton_atol"],
        "snes_max_it": P["newton_max_it"], "snes_error_if_not_converged": False,
        "ksp_type": "preonly", "pc_type": "lu",
    }
    ref_problem = NonlinearProblem(F_total, w, bcs=bcs, petsc_options=snes_petsc_options,
                                    petsc_options_prefix="pemfc_diag_ref_",
                                    form_compiler_options=fc_opts)
    w.sub(0).interpolate(lambda x: np.full(x.shape[1], (p_in - p_out) / P_scale))
    w.sub(1).interpolate(lambda x: np.full(x.shape[1], Y_H2O_in))
    w.sub(2).interpolate(lambda x: np.full(x.shape[1], Y_O2_in))
    w.x.scatter_forward()

    # Warm-up ramp: solving directly at reference_eta (default 0.30) from
    # this naive/inlet-like guess is NOT reliable on its own (same issue
    # found and fixed in fold_svd_sweep() / run_case_arclength()'s
    # bootstrap -- SNES can fail to converge, leaving w in a divergent
    # state with Cw far outside its physical range, which would silently
    # corrupt every downstream condition-ratio comparison in this
    # function without this check).
    if reference_eta > 0.05:
        for eta_now in np.linspace(0.02, reference_eta, 8)[:-1]:
            eta_const.value = eta_now
            ref_problem.solve()
            if ref_problem.solver.getConvergedReason() <= 0:
                raise RuntimeError(
                    f"diagnose_fold_point: reference warm-up solve failed to converge "
                    f"at eta={eta_now:.4f}.")
    eta_const.value = reference_eta
    ref_problem.solve()
    if ref_problem.solver.getConvergedReason() <= 0:
        raise RuntimeError(
            f"diagnose_fold_point: reference point solve failed to converge at "
            f"eta={reference_eta} even after warm-up -- cannot produce a trustworthy "
            f"comparison. Try a different reference_eta.")
    w_ref = w.x.array.copy()
    reference = analyze_point(w_ref, reference_eta, "REFERENCE")

    smallest_ratio = reference["svals_scaled"][0] / max(stuck["svals_scaled"][0], 1e-300)
    cond_ratio = stuck["cond"] / max(reference["cond"], 1e-300)

    verdict_lines = [
        "",
        f"Stuck point (eta={stuck['eta']:.5f}): smallest scaled sval = "
        f"{stuck['svals_scaled'][0]:.4e}, cond = {stuck['cond']:.4e}",
        f"Reference point (eta={reference['eta']:.5f}): smallest scaled sval = "
        f"{reference['svals_scaled'][0]:.4e}, cond = {reference['cond']:.4e}",
        f"Smallest-singular-value ratio (reference/stuck): {smallest_ratio:.4e}",
        f"Condition-number ratio (stuck/reference): {cond_ratio:.4e}",
        f"Stuck-point transversality (normalized): {stuck['transversality_normalized']:.4e}",
    ]

    if cond_ratio < 20:
        verdict = ("NOT clearly singular relative to a normal point -- the stuck point's "
                   "conditioning is within roughly the same order of magnitude as an "
                   "ordinary, easily-converged reference point. This does NOT look like a "
                   "genuine fold, cusp, or bifurcation; the stall is more likely an "
                   "algorithmic/step-control issue (tangent update, tolerance, or step size) "
                   "rather than a true mathematical singularity.")
    elif abs(stuck["transversality_normalized"]) < 1e-3:
        verdict = ("LIKELY HIGHER-CODIMENSION POINT: substantially worse-conditioned than the "
                   "reference AND the transversality condition is collapsing toward zero -- "
                   "consistent with a cusp or bifurcation rather than a simple fold. "
                   "Branch-switching or a specialized unfolding would be needed to proceed "
                   "past this point with continuation.")
    else:
        verdict = ("Substantially worse-conditioned than the reference, but transversality is "
                   "still clearly nonzero -- consistent with being NEAR a simple fold (possibly "
                   "very close to, but not exactly at, a genuine limit point). Worth trying "
                   "smaller ds_min / relaxed newton_tol / more max_newton_it before concluding "
                   "this needs branch-switching.")

    verdict_lines.append("")
    verdict_lines.append("VERDICT: " + verdict)
    report = "\n".join(verdict_lines)
    if verbose:
        print(report, flush=True)
    _progress_log(f"diagnose_fold_point verdict: {verdict}")

    return dict(stuck=stuck, reference=reference, smallest_ratio=smallest_ratio,
                cond_ratio=cond_ratio, verdict=verdict, report=report)


def run_case_arclength_auto(overrides=None, eta_start=0.30, ds=0.01, ds_min=1e-4, ds_max=0.03,
                             checkpoint_path="/content/pemfc_m2_arclength_checkpoint.npz",
                             snapshot_dir=None, total_steps_budget=5000, steps_per_attempt=100,
                             svd_diagnostic_every=10, svd_out_path=None, diagnostics_dir=None,
                             refresh_step_ladder=(1e-4,),  # single entry: refresh_tangent is now a
                                                            # deterministic direct Jacobian computation,
                                                            # not parametrized by step size, so repeating
                                                            # it would just redo the same computation
                             newton_tol_ladder=(1e-5, 1e-4),
                             max_newton_it_ladder=(60, 100),
                             max_recovery_rounds=1, verbose=True, stats_out_path=None):
    """Fully automated, unattended pseudo-arclength continuation. Repeatedly
    calls run_case_arclength() in batches of `steps_per_attempt`, and
    whenever a batch makes literally zero progress (the classic "stuck"
    symptom seen manually several times: ds shrinks to ds_min repeatedly
    with eta frozen), automatically works through the SAME recovery
    ladder that was previously done by hand:

      1. refresh_tangent=True, trying refresh_step_ladder in order (each
         value guarded against branch-jumping by run_case_arclength's own
         built-in check, which raises rather than corrupting the
         checkpoint -- caught here and treated as "try the next, smaller
         refresh_step").
      2. If refresh doesn't help, relaxed (newton_tol, max_newton_it)
         pairs from the ladders, on the theory that a moderately (not
         extremely) ill-conditioned point just needs more/looser Newton
         iterations rather than a different tangent.

    Deliberately does NOT attempt rollback_checkpoint automatically --
    unlike the steps above (which either succeed or safely no-op),
    deciding how far to roll back a bad excursion requires judgment
    (see the manual sessions this was built from) and is too risky to
    automate blindly during an unattended multi-hour run. If the full
    ladder is exhausted `max_recovery_rounds` times without meaningful
    progress, this stops gracefully -- the checkpoint is left exactly as
    the last successful run_case_arclength() call left it (never mid-
    write) -- and AUTOMATICALLY runs diagnose_fold_point() (read-only,
    doesn't touch the checkpoint) before returning, logging its verdict
    so there's no separate manual step needed just to see whether the
    stall is a genuine higher-codimension point or something a bad
    excursion / wider recovery ladder could still fix. Actually acting on
    a genuine higher-codimension verdict (branch-switching) still needs a
    human -- that technique isn't implemented here.

    snapshot_dir, if given, is passed straight through to
    run_case_arclength() -- the full Fig. 7 field set (s, p, Cw, CgO2,
    CgH2O, C_O2_mix) is saved for EVERY accepted step (not just the
    checkpoint's latest one), so any point along the branch can be
    plotted after the fact via a fresh save_field_snapshot()-style load
    of the relevant step_*.npz file, without needing to re-trace anything.

    Returns the total number of accepted steps in the checkpoint when
    this returns (whether by reaching total_steps_budget or by stopping
    stuck).
    """
    def _count():
        if os.path.exists(checkpoint_path):
            try:
                return len(np.load(checkpoint_path)["eta_hist"])
            except Exception:
                return 0
        return 0

    # Recovery-ladder usage statistics -- how often each strategy was
    # actually needed, for a "practical robustness" discussion in the
    # paper. Persists across resumed sessions by loading whatever's
    # already at stats_out_path.
    stats = dict(n_plain_attempts=0, n_plain_success=0, n_stuck_events=0,
                 n_refresh_success=0, n_tolerance_success=0, n_give_up=0)
    if stats_out_path and os.path.exists(stats_out_path):
        try:
            import json
            with open(stats_out_path) as f:
                stats.update(json.load(f))
            _progress_log(f"  loaded prior recovery-ladder stats from {stats_out_path}")
        except Exception as e:
            _progress_log(f"  could not load prior stats ({e}), starting fresh")

    def _save_stats():
        if stats_out_path:
            try:
                import json
                with open(stats_out_path, "w") as f:
                    json.dump(stats, f, indent=2)
            except Exception as e:
                _progress_log(f"  stats save failed (continuing anyway): {e}")

    total_accepted = _count()
    _progress_log(f"run_case_arclength_auto: starting, {total_accepted} steps already in checkpoint")

    def _eta_now():
        if os.path.exists(checkpoint_path):
            try:
                return float(np.load(checkpoint_path)["eta_last"])
            except Exception:
                return None
        return None

    # Minimum cumulative |eta| advance over a whole attempt (up to
    # steps_per_attempt steps) to count as REAL progress -- not just
    # "some new steps got accepted". Without this, a region where the
    # tangent's eta-component (deta_tan) is near-zero (confirmed to
    # happen at the second stuck point) can keep accepting steps
    # indefinitely -- each a genuine, converged, non-branch-jumping step,
    # advancing true arc-length -- while eta itself stays essentially
    # frozen to 4-5 decimal places, so total_accepted keeps climbing and
    # the old check (total_accepted > prev_total) never detects the stall.
    min_eta_progress = 20 * ds_min

    recovery_rounds = 0
    while total_accepted < total_steps_budget:
        resume = os.path.exists(checkpoint_path)
        prev_total = total_accepted
        prev_eta = _eta_now()
        stats["n_plain_attempts"] += 1
        try:
            result = run_case_arclength(
                overrides=overrides, eta_start=eta_start, ds=ds, ds_min=ds_min, ds_max=ds_max,
                n_steps=steps_per_attempt, resume=resume, checkpoint_path=checkpoint_path,
                snapshot_dir=snapshot_dir, verbose=verbose,
                svd_diagnostic_every=svd_diagnostic_every, svd_out_path=svd_out_path)
            total_accepted = len(result["eta"])
        except Exception as e:
            _progress_log(f"run_case_arclength_auto: plain attempt raised {type(e).__name__}: {e}")
            total_accepted = _count()

        eta_progress = None
        if prev_eta is not None:
            eta_now = _eta_now()
            if eta_now is not None:
                eta_progress = abs(eta_now - prev_eta)

        if total_accepted > prev_total and (eta_progress is None or eta_progress >= min_eta_progress):
            stats["n_plain_success"] += 1
            _save_stats()
            recovery_rounds = 0  # making real progress -- reset the stuck counter
            continue

        if total_accepted > prev_total and eta_progress is not None and eta_progress < min_eta_progress:
            _progress_log(f"run_case_arclength_auto: {total_accepted - prev_total} step(s) accepted but "
                          f"eta only advanced {eta_progress:.2e} (< {min_eta_progress:.2e} threshold) -- "
                          f"treating as STUCK despite nominal progress (near-zero deta_tan region).")

        # --- stuck: zero (or negligible) progress in a plain attempt. ---
        # --- Work through the recovery ladder before giving up.       ---
        stats["n_stuck_events"] += 1
        _save_stats()
        _progress_log(f"run_case_arclength_auto: STUCK at {total_accepted} accepted steps "
                      f"-- starting recovery ladder (round {recovery_rounds + 1}/{max_recovery_rounds})")
        recovered_this_round = False
        eta_before_ladder = _eta_now()

        for refresh_step_try in refresh_step_ladder:
            _progress_log(f"  [ladder: refresh] now attempting refresh_step={refresh_step_try:.1e} "
                          f"(n_steps={steps_per_attempt})")
            try:
                result = run_case_arclength(
                    overrides=overrides, eta_start=eta_start, ds=ds, ds_min=ds_min, ds_max=ds_max,
                    n_steps=steps_per_attempt, resume=True, checkpoint_path=checkpoint_path,
                    refresh_tangent=True, refresh_step=refresh_step_try,
                    snapshot_dir=snapshot_dir, verbose=verbose,
                svd_diagnostic_every=svd_diagnostic_every, svd_out_path=svd_out_path)
                new_total = len(result["eta"])
            except Exception as e:
                _progress_log(f"  refresh_step={refresh_step_try:.1e} raised "
                              f"{type(e).__name__}: {e} -- trying next")
                new_total = _count()
                continue
            eta_progress_ladder = None
            if eta_before_ladder is not None:
                eta_now_ladder = _eta_now()
                if eta_now_ladder is not None:
                    eta_progress_ladder = abs(eta_now_ladder - eta_before_ladder)
            decisive = new_total > total_accepted + 1 and (
                eta_progress_ladder is None or eta_progress_ladder >= min_eta_progress)
            if decisive:
                # +1 tolerance: the refresh bootstrap itself always adds
                # exactly one accepted point even when the main loop
                # after it can't proceed -- only count this as real
                # recovery if it went further than just that AND eta
                # actually advanced meaningfully (not just more steps
                # accepted in a near-zero-deta_tan region).
                _progress_log(f"  recovered via refresh_step={refresh_step_try:.1e}: "
                              f"{total_accepted} -> {new_total}")
                total_accepted = new_total
                recovered_this_round = True
                stats["n_refresh_success"] += 1
                _save_stats()
                break
            if new_total > total_accepted + 1 and eta_progress_ladder is not None:
                _progress_log(f"  refresh_step={refresh_step_try:.1e}: {total_accepted} -> {new_total} "
                              f"steps accepted, but eta only advanced {eta_progress_ladder:.2e} "
                              f"(< {min_eta_progress:.2e}) -- NOT counting as real recovery")
            _progress_log(f"  refresh_step={refresh_step_try:.1e} did not decisively recover "
                          f"({total_accepted} -> {new_total}, only the refresh's own bootstrap "
                          f"point) -- trying next")
            total_accepted = new_total

        if recovered_this_round:
            recovery_rounds = 0
            continue

        _progress_log("  [ladder: refresh] exhausted with no decisive recovery -- "
                      "moving to relaxed-tolerance ladder")
        for tol_try, max_it_try in zip(newton_tol_ladder, max_newton_it_ladder):
            _progress_log(f"  [ladder: tolerance] now attempting newton_tol={tol_try:.1e}, "
                          f"max_newton_it={max_it_try} (n_steps={steps_per_attempt})")
            try:
                result = run_case_arclength(
                    overrides=overrides, eta_start=eta_start, ds=ds, ds_min=ds_min, ds_max=ds_max,
                    n_steps=steps_per_attempt, resume=True, checkpoint_path=checkpoint_path,
                    newton_tol=tol_try, max_newton_it=max_it_try,
                    snapshot_dir=snapshot_dir, verbose=verbose,
                svd_diagnostic_every=svd_diagnostic_every, svd_out_path=svd_out_path)
                new_total = len(result["eta"])
            except Exception as e:
                _progress_log(f"  newton_tol={tol_try:.1e}/max_it={max_it_try} raised "
                              f"{type(e).__name__}: {e} -- trying next")
                new_total = _count()
                continue
            eta_progress_ladder = None
            if eta_before_ladder is not None:
                eta_now_ladder = _eta_now()
                if eta_now_ladder is not None:
                    eta_progress_ladder = abs(eta_now_ladder - eta_before_ladder)
            decisive = new_total > total_accepted and (
                eta_progress_ladder is None or eta_progress_ladder >= min_eta_progress)
            if decisive:
                _progress_log(f"  progress via newton_tol={tol_try:.1e}, max_it={max_it_try}: "
                              f"{total_accepted} -> {new_total}")
                total_accepted = new_total
                recovered_this_round = True
                stats["n_tolerance_success"] += 1
                _save_stats()
                break
            if new_total > total_accepted and eta_progress_ladder is not None:
                _progress_log(f"  newton_tol={tol_try:.1e}, max_it={max_it_try}: "
                              f"{total_accepted} -> {new_total} steps accepted, but eta only "
                              f"advanced {eta_progress_ladder:.2e} (< {min_eta_progress:.2e}) "
                              f"-- NOT counting as real recovery")
            else:
                _progress_log(f"  newton_tol={tol_try:.1e}/max_it={max_it_try} made no progress "
                              f"({total_accepted} -> {new_total}) -- trying next")
            total_accepted = new_total
            total_accepted = new_total

        if recovered_this_round:
            recovery_rounds = 0
            continue

        recovery_rounds += 1
        if recovery_rounds >= max_recovery_rounds:
            stats["n_give_up"] += 1
            _save_stats()
            eta_stuck = float(np.load(checkpoint_path)["eta_last"])
            _progress_log(
                f"run_case_arclength_auto: recovery ladder exhausted {recovery_rounds} time(s) "
                f"in a row with no progress -- STOPPING at {total_accepted} accepted steps "
                f"(eta={eta_stuck:.5f}). Checkpoint is safe and unmodified beyond this point. "
                f"Running diagnose_fold_point() automatically to characterize the stuck point...")
            if verbose:
                print(f"\n[run_case_arclength_auto] STOPPED, stuck after exhausting recovery "
                      f"{recovery_rounds} time(s). {total_accepted} accepted steps in checkpoint. "
                      f"Running diagnose_fold_point() automatically...", flush=True)

            nullvec_path = None
            outlet_path = None
            if diagnostics_dir:
                os.makedirs(diagnostics_dir, exist_ok=True)
                nullvec_path = os.path.join(diagnostics_dir, f"nullvec_eta_{eta_stuck:.5f}.npz")
                outlet_path = os.path.join(diagnostics_dir, f"outletflow_eta_{eta_stuck:.5f}.npz")

            try:
                diag = diagnose_fold_point(checkpoint_path, overrides=overrides, verbose=verbose,
                                            save_nullvec_path=nullvec_path)
                _progress_log(f"run_case_arclength_auto: automatic diagnosis verdict -- "
                              f"{diag['verdict']}")
                if verbose:
                    print(f"\n[run_case_arclength_auto] Automatic diagnosis: {diag['verdict']}",
                          flush=True)
                    if diag["cond_ratio"] < 20:
                        print("[run_case_arclength_auto] -> NOT a genuine singularity by this "
                              "check. A bad excursion may have slipped through undetected -- "
                              "consider rollback_checkpoint() before retrying, or widen the "
                              "recovery ladders (refresh_step_ladder / newton_tol_ladder) and "
                              "call run_case_arclength_auto() again.", flush=True)
                    else:
                        print("[run_case_arclength_auto] -> Looks like a genuine higher-"
                              "codimension point (fold/cusp/bifurcation). This automated "
                              "continuation cannot proceed past it -- branch-switching or a "
                              "specialized unfolding would be needed, which isn't implemented "
                              "here.", flush=True)
            except Exception as e:
                _progress_log(f"run_case_arclength_auto: automatic diagnosis itself failed "
                              f"({type(e).__name__}: {e}) -- run diagnose_fold_point() manually.")
                if verbose:
                    print(f"[run_case_arclength_auto] Automatic diagnosis failed ({e}) -- "
                          f"run diagnose_fold_point() manually.", flush=True)

            try:
                check_outlet_flow(checkpoint_path, overrides=overrides, verbose=verbose,
                                   save_path=outlet_path)
            except Exception as e:
                _progress_log(f"run_case_arclength_auto: automatic check_outlet_flow failed "
                              f"({type(e).__name__}: {e}) -- run check_outlet_flow() manually.")
                if verbose:
                    print(f"[run_case_arclength_auto] Automatic check_outlet_flow failed ({e}) "
                          f"-- run check_outlet_flow() manually.", flush=True)
            break

    _progress_log(f"run_case_arclength_auto: returning with {total_accepted} total accepted steps")
    _save_stats()
    if verbose:
        print(f"\n[run_case_arclength_auto] Recovery-ladder usage summary: {stats}", flush=True)
    return total_accepted



def rollback_checkpoint(checkpoint_path, overrides=None, n_pop=1, verbose=True):
    """Undo the last `n_pop` accepted step(s) recorded in checkpoint_path's
    history (e.g. a bad excursion caused by refresh_tangent bootstrapping
    in the wrong eta direction -- confirmed to happen when the branch is
    locally decreasing in eta but the refresh always tried "+ds" for its
    second point, landing on a discontinuous, off-branch solution rather
    than a valid nearby point).

    Since only the MOST RECENT solution vector is kept in the checkpoint
    (not one per history entry), rolling back can't just restore an old
    w array -- it re-SOLVES (ordinary Newton, warm-started from whatever
    is currently in the checkpoint) at the eta value that becomes the new
    "last" entry after popping, and overwrites w_last/eta_last with that
    fresh solution. The old (possibly-bad) tangent is left in the
    checkpoint as-is; follow this call with resume=True,
    refresh_tangent=True before continuing, since a rolled-back point's
    saved tangent is not trustworthy.
    """
    P = dict(DEFAULT_PARAMS)
    if overrides:
        P.update(overrides)

    F, R = P["F"], P["R"]
    K_perm, eps_p = P["K_perm"], P["eps_p"]
    p_in, p_out = P["p_in"], P["p_out"]
    mu_g, D_O2_g, D_H2O_g = P["mu_g"], P["D_O2_g"], P["D_H2O_g"]
    I0_ref, alpha_c, alpha_w = P["I0"], P["alpha_c"], P["alpha_w"]
    T_ref_I0, Ea_orr = P["T_ref_I0"], P["Ea_orr"]
    rho_l = P["rho_l"]
    M_H2O, M_O2, M_N2 = P["M_H2O"], P["M_O2"], P["M_N2"]
    mu_l = P["mu_l"]
    T_cell, C_O2_ref = P["T_cell"], P["C_O2_ref"]
    I0 = I0_ref * np.exp(-(Ea_orr / R) * (1.0 / T_cell - 1.0 / T_ref_I0))
    RH_in = P["RH_in"]
    theta_c_deg = P["theta_c_deg"]
    theta_c = theta_c_deg * np.pi / 180.0
    sigma_st = P["sigma_st"]
    X_O2_in, X_N2_in = P["X_O2_in"], P["X_N2_in"]
    w_ch, w_rb, Lx = P["w_ch"], P["w_rb"], P["Lx"]
    n_corey, tau_brug, s_max_cap = P["n_corey"], P["tau_brug"], P["s_max"]
    s_smooth_eps = P["s_smooth_eps"]
    single_phase = P["single_phase"]
    nx, ny = P["nx"], P["ny"]

    P_sat = Psat_atm(T_cell - 273.15) * 101325.0
    Ly = w_ch + w_rb
    y_inlet_end = 0.5 * w_ch
    y_rib_end = 0.5 * w_ch + w_rb

    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [Lx, Ly]], [nx, ny], mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    def on_membrane(x): return np.isclose(x[0], 0.0)
    def on_inlet(x):    return np.isclose(x[0], Lx) & (x[1] <= y_inlet_end + 1e-12)
    def on_rib(x):      return np.isclose(x[0], Lx) & (x[1] > y_inlet_end + 1e-12) & (x[1] <= y_rib_end + 1e-12)
    def on_outlet(x):   return np.isclose(x[0], Lx) & (x[1] > y_rib_end + 1e-12)

    TAG_MEMBRANE, TAG_INLET, TAG_RIB, TAG_OUTLET = 1, 2, 3, 4
    facet_indices, facet_markers = [], []
    for tag, locator in [(TAG_MEMBRANE, on_membrane), (TAG_INLET, on_inlet),
                          (TAG_RIB, on_rib), (TAG_OUTLET, on_outlet)]:
        idx = mesh.locate_entities_boundary(domain, fdim, locator)
        facet_indices.append(idx)
        facet_markers.append(np.full_like(idx, tag))
    facet_indices = np.concatenate(facet_indices)
    facet_markers = np.concatenate(facet_markers)
    order = np.argsort(facet_indices)
    facet_tags = mesh.meshtags(domain, fdim, facet_indices[order], facet_markers[order])

    ds_meas = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    dx = ufl.Measure("dx", domain=domain)
    nvec = ufl.FacetNormal(domain)

    V = fem.functionspace(domain, ("Lagrange", 1))
    P1e = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    ME = basix.ufl.mixed_element([P1e, P1e, P1e])
    W = fem.functionspace(domain, ME)

    w = fem.Function(W, name="w")
    v_te = ufl.TestFunction(W)
    p_hat_sol, Cw_sol, CgO2_sol = ufl.split(w)
    v_p, v_cw, v_cgo2 = ufl.split(v_te)

    P_scale = 1000.0
    p_sol = p_out + P_scale * p_hat_sol

    X_H2O_in = np.clip(RH_in * P_sat / p_in, 0.0, 0.98)
    dry_scale = 1.0 - X_H2O_in
    X_O2_eff = X_O2_in * dry_scale
    X_N2_eff = X_N2_in * dry_scale
    M_mix_in = X_O2_eff * M_O2 + X_N2_eff * M_N2 + X_H2O_in * M_H2O
    Y_O2_in = X_O2_eff * M_O2 / M_mix_in
    Y_H2O_in = X_H2O_in * M_H2O / M_mix_in
    M_dry_in = X_O2_in * M_O2 + X_N2_in * M_N2

    rho_g_const = p_out * M_mix_in / (R * T_cell)
    Cg_sat = (P_sat * M_H2O) / (P_sat * M_H2O + (p_in - P_sat) * M_dry_in)
    Cl_sat = 1.0

    inlet_facets = facet_tags.find(TAG_INLET)
    outlet_facets = facet_tags.find(TAG_OUTLET)
    dofs_p_in = fem.locate_dofs_topological(W.sub(0), fdim, inlet_facets)
    dofs_p_out = fem.locate_dofs_topological(W.sub(0), fdim, outlet_facets)
    dofs_cw_in = fem.locate_dofs_topological(W.sub(1), fdim, inlet_facets)
    dofs_cgo2_in = fem.locate_dofs_topological(W.sub(2), fdim, inlet_facets)
    bcs = [
        fem.dirichletbc(default_scalar_type((p_in - p_out) / P_scale), dofs_p_in, W.sub(0)),
        fem.dirichletbc(default_scalar_type(0.0), dofs_p_out, W.sub(0)),
        fem.dirichletbc(default_scalar_type(Y_H2O_in), dofs_cw_in, W.sub(1)),
        fem.dirichletbc(default_scalar_type(Y_O2_in), dofs_cgo2_in, W.sub(2)),
    ]

    def s_expr(Cw):
        if single_phase:
            return 0.0 * Cw
        raw = rho_g_const * (Cw - Cg_sat) / (rho_l * (Cl_sat - Cw) + rho_g_const * (Cw - Cg_sat) + 1e-12)
        s_smooth = 0.5 * (raw + ufl.sqrt(raw**2 + s_smooth_eps**2))
        s_smooth = ufl.max_value(s_smooth, 0.0)
        return ufl.min_value(s_smooth, s_max_cap)

    def min_smooth(a, b, eps=s_smooth_eps):
        return 0.5 * (a + b - ufl.sqrt((a - b) ** 2 + eps ** 2))

    hydrophobic = theta_c_deg >= 90.0
    s = s_expr(Cw_sol)
    krl = s ** n_corey
    krg = (1.0 - s) ** n_corey
    lam_l = (krl / mu_l) / (krl / mu_l + krg / mu_g + 1e-30)
    lam_g = 1.0 - lam_l
    rho_mix = rho_g_const * (1.0 - s) + rho_l * s
    nu_g_c, nu_l_c = mu_g / rho_g_const, mu_l / rho_l
    nu_mix = 1.0 / (krg / nu_g_c + krl / nu_l_c + 1e-30)
    mu_mix = nu_mix * rho_mix
    kappa = rho_mix * K_perm / mu_mix

    if hydrophobic:
        dJds = 1.417 - 4.240 * s + 3.789 * s**2
    else:
        u_hp = 1.0 - s
        dJds = -(1.417 - 4.240 * u_hp + 3.789 * u_hp**2)

    Dc = ((lam_l * lam_g * K_perm / (nu_mix + 1e-30))
          * sigma_st * np.cos(theta_c) * (eps_p / K_perm) ** 0.5 * dJds)
    Gamma = Dc * (1.0 - Cg_sat) / (rho_l - rho_g_const * Cg_sat)
    grads = ufl.grad(s)
    u_darcy = -(kappa / rho_mix) * ufl.grad(p_sol)
    CgH2O_local = min_smooth(Cw_sol, Cg_sat)
    Weff_w = lam_l + lam_g * CgH2O_local
    Deff_O2 = rho_g_const * (eps_p * (1.0 - s)) ** tau_brug * D_O2_g
    Diff_w = rho_g_const * (eps_p * (1.0 - s)) ** tau_brug * D_H2O_g + Gamma * rho_mix
    Jl_vec = Dc * grads
    F_conv_w = Weff_w * rho_mix * u_darcy
    W_O2 = rho_mix * lam_g * u_darcy

    eta_const = fem.Constant(domain, default_scalar_type(0.0))
    C_O2_molar = rho_g_const * CgO2_sol / M_O2
    I_local = (1.0 - s) * I0 * (C_O2_molar / C_O2_ref) * ufl.exp(alpha_c * F * eta_const / (R * T_cell))
    N_O2_expr = (M_O2 / (4.0 * F)) * I_local
    N_w_expr = (M_H2O * (1.0 + 2.0 * alpha_w) / (2.0 * F)) * I_local

    F_pres = (kappa * ufl.inner(ufl.grad(p_sol), ufl.grad(v_p)) * dx
              - (N_w_expr - N_O2_expr) * v_p * ds_meas(TAG_MEMBRANE))
    F_water = (Diff_w * ufl.inner(ufl.grad(Cw_sol), ufl.grad(v_cw)) * dx
               - ufl.inner(F_conv_w, ufl.grad(v_cw)) * dx
               + _smooth_outflow(ufl.inner(F_conv_w, nvec)) * v_cw * ds_meas(TAG_OUTLET)
               - N_w_expr * v_cw * ds_meas(TAG_MEMBRANE))
    F_oxy = (-CgO2_sol * ufl.inner(W_O2, ufl.grad(v_cgo2)) * dx
             + Deff_O2 * ufl.inner(ufl.grad(CgO2_sol), ufl.grad(v_cgo2)) * dx
             + CgO2_sol * ufl.inner(Jl_vec, ufl.grad(v_cgo2)) * dx
             + CgO2_sol * _smooth_outflow(ufl.inner(W_O2, nvec)) * v_cgo2 * ds_meas(TAG_OUTLET)
             + N_O2_expr * v_cgo2 * ds_meas(TAG_MEMBRANE))
    F_total = F_pres + F_water + F_oxy

    fc_opts = {"quadrature_degree": 4}
    snes_petsc_options = {
        "snes_type": "newtonls", "snes_rtol": P["newton_rtol"], "snes_atol": P["newton_atol"],
        "snes_max_it": P["newton_max_it"], "snes_error_if_not_converged": False,
        "ksp_type": "preonly", "pc_type": "lu",
    }
    solve_problem = NonlinearProblem(F_total, w, bcs=bcs, petsc_options=snes_petsc_options,
                                      petsc_options_prefix="pemfc_rollback_",
                                      form_compiler_options=fc_opts)

    d = np.load(checkpoint_path)
    eta_hist = list(d["eta_hist"])
    I_hist = list(d["I_hist"])
    smax_hist = list(d["smax_hist"])
    ds_hist = list(d["ds_hist"])
    w_current = d["w_last"]

    dropped = []
    for _ in range(n_pop):
        dropped.append((eta_hist.pop(), I_hist.pop(), smax_hist.pop(), ds_hist.pop()))
    if verbose:
        for de, dI, dsm, _ in dropped:
            print(f"Dropped: eta={de:.5f}  I={dI/1e4:.4f} A/cm^2  s_max={dsm:.3f}", flush=True)

    good_eta = eta_hist[-1]
    if verbose:
        print(f"\nRolling back to eta={good_eta:.5f} (re-solving fresh)...", flush=True)
    _progress_log(f"rollback_checkpoint: popped {n_pop}, re-solving at eta={good_eta:.5f}")

    eta_const.value = good_eta
    w.x.array[:] = w_current
    w.x.scatter_forward()
    solve_problem.solve()
    converged = solve_problem.solver.getConvergedReason() > 0
    if not converged:
        if verbose:
            print("  warm-started solve did not converge, retrying from a naive initial "
                  "guess...", flush=True)
        w.sub(0).interpolate(lambda x: np.full(x.shape[1], (p_in - p_out) / P_scale))
        w.sub(1).interpolate(lambda x: np.full(x.shape[1], Y_H2O_in))
        w.sub(2).interpolate(lambda x: np.full(x.shape[1], Y_O2_in))
        w.x.scatter_forward()
        solve_problem.solve()
        converged = solve_problem.solver.getConvergedReason() > 0

    if not converged:
        raise RuntimeError(f"rollback_checkpoint: could not re-solve at eta={good_eta} "
                            f"either warm-started or from a naive guess -- checkpoint left "
                            f"UNCHANGED on disk, nothing was overwritten.")

    w_good = w.x.array.copy()
    if verbose:
        print(f"  re-solve converged. Saving repaired checkpoint at eta={good_eta:.5f}.",
              flush=True)

    if np.all(np.isfinite(w_good)):
        np.savez(checkpoint_path, w_last=w_good, eta_last=good_eta,
                 dw_tan=d["dw_tan"], deta_tan=d["deta_tan"],
                 eta_hist=np.array(eta_hist), I_hist=np.array(I_hist),
                 smax_hist=np.array(smax_hist), ds_hist=np.array(ds_hist),
                 cur_ds=float(d["cur_ds"]))
        _progress_log(f"rollback_checkpoint: repaired checkpoint saved at eta={good_eta:.5f}")
        if verbose:
            print("\nDone. The saved tangent is now STALE (it was for the point you just "
                  "dropped) -- call run_case_arclength(resume=True, refresh_tangent=True, "
                  "...) next, not a plain resume.", flush=True)
    else:
        raise RuntimeError("rollback_checkpoint: re-solved state contains NaN/Inf -- "
                            "refusing to save, checkpoint left UNCHANGED on disk.")

    return dict(eta=good_eta, w=w_good)
def save_field_snapshot(checkpoint_path, out_path, overrides=None, verbose=True):
    """Extract and save the full 2D field snapshot (all four Fig. 7
    panels -- water mixture C^{H2O}, water vapor Cg^{H2O}, oxygen mixture
    C^{O2}, gas oxygen mass fraction Cg^{O2} -- plus s and p for
    completeness) at whatever point is currently saved in checkpoint_path
    (from run_case_arclength). Read-only: does not modify the checkpoint.

    Field definitions match the paper (Table 1 / Sec. 2.1): water vapor
    Cg^{H2O} = min(C^{H2O}, Cg_sat) (paper's own two-phase equilibrium
    condition -- gas-phase water saturates once liquid forms, same
    min_smooth() expression used internally for the model's Weff_w term);
    oxygen mixture C^{O2} = (rho_g/rho_mix) * Cg^{O2} (from rho*C^{O2} =
    rho_g*Cg^{O2}, Eq. 11 with oxygen existing only in the gas phase).
    """
    P = dict(DEFAULT_PARAMS)
    if overrides:
        P.update(overrides)

    F, R = P["F"], P["R"]
    K_perm, eps_p = P["K_perm"], P["eps_p"]
    p_in, p_out = P["p_in"], P["p_out"]
    mu_g, D_O2_g, D_H2O_g = P["mu_g"], P["D_O2_g"], P["D_H2O_g"]
    I0_ref, alpha_c, alpha_w = P["I0"], P["alpha_c"], P["alpha_w"]
    T_ref_I0, Ea_orr = P["T_ref_I0"], P["Ea_orr"]
    rho_l = P["rho_l"]
    M_H2O, M_O2, M_N2 = P["M_H2O"], P["M_O2"], P["M_N2"]
    mu_l = P["mu_l"]
    T_cell, C_O2_ref = P["T_cell"], P["C_O2_ref"]
    RH_in = P["RH_in"]
    theta_c_deg = P["theta_c_deg"]
    sigma_st = P["sigma_st"]
    X_O2_in, X_N2_in = P["X_O2_in"], P["X_N2_in"]
    w_ch, w_rb, Lx = P["w_ch"], P["w_rb"], P["Lx"]
    n_corey, tau_brug, s_max_cap = P["n_corey"], P["tau_brug"], P["s_max"]
    s_smooth_eps = P["s_smooth_eps"]
    single_phase = P["single_phase"]
    nx, ny = P["nx"], P["ny"]

    P_sat = Psat_atm(T_cell - 273.15) * 101325.0
    Ly = w_ch + w_rb
    y_inlet_end = 0.5 * w_ch
    y_rib_end = 0.5 * w_ch + w_rb

    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [Lx, Ly]], [nx, ny], mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    def on_membrane(x): return np.isclose(x[0], 0.0)
    def on_inlet(x):    return np.isclose(x[0], Lx) & (x[1] <= y_inlet_end + 1e-12)
    def on_rib(x):      return np.isclose(x[0], Lx) & (x[1] > y_inlet_end + 1e-12) & (x[1] <= y_rib_end + 1e-12)
    def on_outlet(x):   return np.isclose(x[0], Lx) & (x[1] > y_rib_end + 1e-12)

    TAG_MEMBRANE, TAG_INLET, TAG_RIB, TAG_OUTLET = 1, 2, 3, 4
    facet_indices, facet_markers = [], []
    for tag, locator in [(TAG_MEMBRANE, on_membrane), (TAG_INLET, on_inlet),
                          (TAG_RIB, on_rib), (TAG_OUTLET, on_outlet)]:
        idx = mesh.locate_entities_boundary(domain, fdim, locator)
        facet_indices.append(idx)
        facet_markers.append(np.full_like(idx, tag))
    facet_indices = np.concatenate(facet_indices)
    facet_markers = np.concatenate(facet_markers)
    order = np.argsort(facet_indices)
    facet_tags = mesh.meshtags(domain, fdim, facet_indices[order], facet_markers[order])
    dx = ufl.Measure("dx", domain=domain)

    V = fem.functionspace(domain, ("Lagrange", 1))
    P1e = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    ME = basix.ufl.mixed_element([P1e, P1e, P1e])
    W = fem.functionspace(domain, ME)

    w = fem.Function(W, name="w")
    p_hat_sol, Cw_sol, CgO2_sol = ufl.split(w)

    P_scale = 1000.0
    p_sol = p_out + P_scale * p_hat_sol

    X_H2O_in = np.clip(RH_in * P_sat / p_in, 0.0, 0.98)
    dry_scale = 1.0 - X_H2O_in
    X_O2_eff = X_O2_in * dry_scale
    X_N2_eff = X_N2_in * dry_scale
    M_mix_in = X_O2_eff * M_O2 + X_N2_eff * M_N2 + X_H2O_in * M_H2O
    M_dry_in = X_O2_in * M_O2 + X_N2_in * M_N2
    rho_g_const = p_out * M_mix_in / (R * T_cell)
    Cg_sat = (P_sat * M_H2O) / (P_sat * M_H2O + (p_in - P_sat) * M_dry_in)
    Cl_sat = 1.0

    def s_expr(Cw):
        if single_phase:
            return 0.0 * Cw
        raw = rho_g_const * (Cw - Cg_sat) / (rho_l * (Cl_sat - Cw) + rho_g_const * (Cw - Cg_sat) + 1e-12)
        s_smooth = 0.5 * (raw + ufl.sqrt(raw**2 + s_smooth_eps**2))
        s_smooth = ufl.max_value(s_smooth, 0.0)
        return ufl.min_value(s_smooth, s_max_cap)

    def min_smooth(a, b, eps=s_smooth_eps):
        return 0.5 * (a + b - ufl.sqrt((a - b) ** 2 + eps ** 2))

    s = s_expr(Cw_sol)
    krl = s ** n_corey
    krg = (1.0 - s) ** n_corey
    lam_l = (krl / mu_l) / (krl / mu_l + krg / mu_g + 1e-30)
    lam_g = 1.0 - lam_l
    rho_mix = rho_g_const * (1.0 - s) + rho_l * s
    CgH2O_local = min_smooth(Cw_sol, Cg_sat)          # Fig. 7b: water vapor
    C_O2_mix = rho_g_const * CgO2_sol / rho_mix         # Fig. 7c: oxygen mixture

    ckpt = np.load(checkpoint_path)
    w.x.array[:] = ckpt["w_last"]
    w.x.scatter_forward()
    eta_val = float(ckpt["eta_last"])
    if verbose:
        print(f"Extracting field snapshot at eta={eta_val:.5f} from {checkpoint_path}", flush=True)

    coords = V.tabulate_dof_coordinates()[:, :2]
    p_arr = _project_scalar_standalone(p_sol, V, dx)
    Cw_arr = _project_scalar_standalone(Cw_sol, V, dx)
    CgO2_arr = _project_scalar_standalone(CgO2_sol, V, dx)
    CgH2O_arr = _project_scalar_standalone(CgH2O_local, V, dx)
    CO2mix_arr = _project_scalar_standalone(C_O2_mix, V, dx)
    s_arr = _project_scalar_standalone(s, V, dx)
    deta_sign = float(ckpt["deta_tan"]) if "deta_tan" in ckpt.files else 1.0
    branch_state = _classify_branch_state(eta_val, float(s_arr.max()), deta_sign)

    np.savez(out_path, coords=coords, eta=eta_val, p=p_arr, Cw=Cw_arr, CgO2=CgO2_arr,
             CgH2O=CgH2O_arr, C_O2_mix=CO2mix_arr, s=s_arr, branch_state=branch_state)
    if verbose:
        print(f"Saved to {out_path}", flush=True)
        print(f"  branch_state: {branch_state}")
        print(f"  s range: {s_arr.min():.3f} - {s_arr.max():.3f}")
        print(f"  Cw range: {Cw_arr.min():.3f} - {Cw_arr.max():.3f}")
        print(f"  CgO2 range: {CgO2_arr.min():.4f} - {CgO2_arr.max():.4f}")
        print(f"  CgH2O (vapor) range: {CgH2O_arr.min():.4f} - {CgH2O_arr.max():.4f}")
        print(f"  C_O2_mix range: {CO2mix_arr.min():.4f} - {CO2mix_arr.max():.4f}")

    return dict(eta=eta_val, coords=coords, p=p_arr, Cw=Cw_arr, CgO2=CgO2_arr,
                CgH2O=CgH2O_arr, C_O2_mix=CO2mix_arr, s=s_arr)
def check_outlet_flow(checkpoint_path, overrides=None, y_band=None, verbose=True, save_path=None):
    """Diagnostic for the "degenerate outflow BC" hypothesis: the outlet
    boundary condition in F_oxy (and F_water) is a natural (Neumann-type)
    ADVECTIVE outflow condition -- CgO2*inner(W_O2, n)*v on ds(OUTLET) --
    which implicitly assumes a genuine outflow there. If the local normal
    velocity approaches zero or changes sign as flooding intensifies, that
    boundary condition's own well-posedness degrades, which would show up
    exactly as a spatially-localized Jacobian null direction concentrated
    at the outlet -- matching what psi_CgO2's spatial plot showed (top-10%
    energy concentrated at y=1.77-2.00mm, squarely inside the outlet
    segment y in [1.6mm, 2.0mm] for the default geometry, spanning the
    FULL GDL thickness rather than being membrane-localized).

    Projects the mixture Darcy velocity u_darcy and the gas-phase oxygen
    advective velocity W_O2 onto the scalar space, and reports statistics
    (min/max/sign) of their y-component (the component that matters for
    the outlet's normal flux, since the outlet is the y=Ly edge) within
    `y_band` (defaults to the outlet segment itself, [y_rib_end, Ly]).
    """
    P = dict(DEFAULT_PARAMS)
    if overrides:
        P.update(overrides)

    F, R = P["F"], P["R"]
    K_perm, eps_p = P["K_perm"], P["eps_p"]
    p_in, p_out = P["p_in"], P["p_out"]
    mu_g, D_O2_g, D_H2O_g = P["mu_g"], P["D_O2_g"], P["D_H2O_g"]
    I0_ref, alpha_c, alpha_w = P["I0"], P["alpha_c"], P["alpha_w"]
    T_ref_I0, Ea_orr = P["T_ref_I0"], P["Ea_orr"]
    rho_l = P["rho_l"]
    M_H2O, M_O2, M_N2 = P["M_H2O"], P["M_O2"], P["M_N2"]
    mu_l = P["mu_l"]
    T_cell, C_O2_ref = P["T_cell"], P["C_O2_ref"]
    RH_in = P["RH_in"]
    theta_c_deg = P["theta_c_deg"]
    sigma_st = P["sigma_st"]
    X_O2_in, X_N2_in = P["X_O2_in"], P["X_N2_in"]
    w_ch, w_rb, Lx = P["w_ch"], P["w_rb"], P["Lx"]
    n_corey, tau_brug, s_max_cap = P["n_corey"], P["tau_brug"], P["s_max"]
    s_smooth_eps = P["s_smooth_eps"]
    single_phase = P["single_phase"]
    nx, ny = P["nx"], P["ny"]

    P_sat = Psat_atm(T_cell - 273.15) * 101325.0
    Ly = w_ch + w_rb
    y_inlet_end = 0.5 * w_ch
    y_rib_end = 0.5 * w_ch + w_rb
    if y_band is None:
        y_band = (y_rib_end, Ly)

    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [Lx, Ly]], [nx, ny], mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    def on_membrane(x): return np.isclose(x[0], 0.0)
    def on_inlet(x):    return np.isclose(x[0], Lx) & (x[1] <= y_inlet_end + 1e-12)
    def on_rib(x):      return np.isclose(x[0], Lx) & (x[1] > y_inlet_end + 1e-12) & (x[1] <= y_rib_end + 1e-12)
    def on_outlet(x):   return np.isclose(x[0], Lx) & (x[1] > y_rib_end + 1e-12)

    TAG_MEMBRANE, TAG_INLET, TAG_RIB, TAG_OUTLET = 1, 2, 3, 4
    facet_indices, facet_markers = [], []
    for tag, locator in [(TAG_MEMBRANE, on_membrane), (TAG_INLET, on_inlet),
                          (TAG_RIB, on_rib), (TAG_OUTLET, on_outlet)]:
        idx = mesh.locate_entities_boundary(domain, fdim, locator)
        facet_indices.append(idx)
        facet_markers.append(np.full_like(idx, tag))
    facet_indices = np.concatenate(facet_indices)
    facet_markers = np.concatenate(facet_markers)
    order = np.argsort(facet_indices)
    facet_tags = mesh.meshtags(domain, fdim, facet_indices[order], facet_markers[order])
    dx = ufl.Measure("dx", domain=domain)

    V = fem.functionspace(domain, ("Lagrange", 1))
    P1e = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    ME = basix.ufl.mixed_element([P1e, P1e, P1e])
    W = fem.functionspace(domain, ME)

    w = fem.Function(W, name="w")
    p_hat_sol, Cw_sol, CgO2_sol = ufl.split(w)

    P_scale = 1000.0
    p_sol = p_out + P_scale * p_hat_sol

    X_H2O_in = np.clip(RH_in * P_sat / p_in, 0.0, 0.98)
    dry_scale = 1.0 - X_H2O_in
    X_O2_eff = X_O2_in * dry_scale
    X_N2_eff = X_N2_in * dry_scale
    M_mix_in = X_O2_eff * M_O2 + X_N2_eff * M_N2 + X_H2O_in * M_H2O
    M_dry_in = X_O2_in * M_O2 + X_N2_in * M_N2
    rho_g_const = p_out * M_mix_in / (R * T_cell)
    Cg_sat = (P_sat * M_H2O) / (P_sat * M_H2O + (p_in - P_sat) * M_dry_in)
    Cl_sat = 1.0

    def s_expr(Cw):
        if single_phase:
            return 0.0 * Cw
        raw = rho_g_const * (Cw - Cg_sat) / (rho_l * (Cl_sat - Cw) + rho_g_const * (Cw - Cg_sat) + 1e-12)
        s_smooth = 0.5 * (raw + ufl.sqrt(raw**2 + s_smooth_eps**2))
        s_smooth = ufl.max_value(s_smooth, 0.0)
        return ufl.min_value(s_smooth, s_max_cap)

    theta_c = theta_c_deg * np.pi / 180.0
    hydrophobic = theta_c_deg >= 90.0
    s = s_expr(Cw_sol)
    krl = s ** n_corey
    krg = (1.0 - s) ** n_corey
    lam_l = (krl / mu_l) / (krl / mu_l + krg / mu_g + 1e-30)
    lam_g = 1.0 - lam_l
    rho_mix = rho_g_const * (1.0 - s) + rho_l * s
    nu_g_c, nu_l_c = mu_g / rho_g_const, mu_l / rho_l
    nu_mix = 1.0 / (krg / nu_g_c + krl / nu_l_c + 1e-30)
    mu_mix = nu_mix * rho_mix
    kappa = rho_mix * K_perm / mu_mix
    u_darcy = -(kappa / rho_mix) * ufl.grad(p_sol)
    W_O2 = rho_mix * lam_g * u_darcy

    ckpt = np.load(checkpoint_path)
    w.x.array[:] = ckpt["w_last"]
    w.x.scatter_forward()
    eta_val = float(ckpt["eta_last"])

    coords = V.tabulate_dof_coordinates()[:, :2]
    uy = _project_scalar_standalone(u_darcy[1], V, dx)
    Wo2y = _project_scalar_standalone(W_O2[1], V, dx)
    s_arr = _project_scalar_standalone(s, V, dx)
    CgO2_arr = _project_scalar_standalone(CgO2_sol, V, dx)

    band_mask = (coords[:, 1] >= y_band[0]) & (coords[:, 1] <= y_band[1])
    n_band = int(band_mask.sum())

    if verbose:
        print(f"eta = {eta_val:.5f}, checking y in [{y_band[0]*1e3:.3f}, {y_band[1]*1e3:.3f}] mm "
              f"({n_band} dofs)", flush=True)
        print(f"  u_darcy_y in band: min={uy[band_mask].min():.4e}  max={uy[band_mask].max():.4e}  "
              f"(sign change: {uy[band_mask].min() < 0 < uy[band_mask].max()})")
        print(f"  W_O2_y (gas oxygen advective velocity) in band: "
              f"min={Wo2y[band_mask].min():.4e}  max={Wo2y[band_mask].max():.4e}  "
              f"(sign change: {Wo2y[band_mask].min() < 0 < Wo2y[band_mask].max()})")
        print(f"  s in band: min={s_arr[band_mask].min():.4f}  max={s_arr[band_mask].max():.4f}")
        print(f"  CgO2 in band: min={CgO2_arr[band_mask].min():.4f}  max={CgO2_arr[band_mask].max():.4f}")
        print(f"  (for context, over the WHOLE domain) u_darcy_y: "
              f"min={uy.min():.4e} max={uy.max():.4e}   W_O2_y: min={Wo2y.min():.4e} max={Wo2y.max():.4e}")
        if Wo2y[band_mask].min() < 0 < Wo2y[band_mask].max():
            print("\n  -> W_O2's normal (y) component CHANGES SIGN inside the outlet band -- "
                  "directly supports the degenerate-outflow-BC hypothesis.")
        elif abs(Wo2y[band_mask]).max() < 0.05 * abs(Wo2y).max():
            print("\n  -> W_O2's normal (y) component is much smaller in this band than "
                  "elsewhere in the domain (near-stagnant) -- also consistent with the "
                  "outflow BC becoming numerically marginal there.")
        else:
            print("\n  -> No sign change or clear near-stagnation found in this band -- "
                  "the degenerate-outflow-BC hypothesis is not directly supported by this check.")

    if save_path:
        np.savez(save_path, eta=eta_val, coords=coords, u_darcy_y=uy, W_O2_y=Wo2y,
                 s=s_arr, CgO2=CgO2_arr, y_band=np.array(y_band))
        if verbose:
            print(f"Saved to {save_path}", flush=True)

    return dict(eta=eta_val, coords=coords, u_darcy_y=uy, W_O2_y=Wo2y, s=s_arr, CgO2=CgO2_arr,
                band_mask=band_mask)
def fold_svd_sweep(eta_list, overrides=None, out_path=None, verbose=True,
                    max_newton_it=50, resume=True):
    """Sweep eta (ordinary Newton continuation, warm-started from the
    previous point) and record the row-norm-equilibrated Jacobian's
    smallest singular value and condition number at each converged
    point. Dedicated, standalone re-implementation of the ad hoc sweep
    originally used to first discover the genuine fold near eta~0.36 (at
    the time, only logged via _progress_log, never saved as reusable
    data) -- built for reproducibly regenerating that figure.

    eta_list should be sorted and start comfortably below the fold (e.g.
    0.30) with fine spacing near where the fold is expected (e.g. 0.005
    steps dropping to 0.001 above ~0.34) -- ordinary Newton continuation
    cannot converge AT or PAST a genuine fold, so the sweep should stop
    naturally (remaining points just won't converge) once it's close
    enough; this is fine, expected behavior, not a bug.

    max_newton_it (default 50): worst-case iteration cap per point.
    Lower this (e.g. to 15-20) on expensive meshes (refined) where a
    single difficult point near the fold -- or near the second,
    higher-codimension obstruction further down the branch, which this
    sweep can also run into if eta_list extends that far -- can otherwise
    take many tens of minutes for one point alone.

    resume (default True): if out_path already has saved data, resume
    from there instead of starting over from eta_list[0] -- skips any
    eta_list values already covered (within 1e-6) and warm-starts from
    the last existing point. Saves incrementally (after every point, not
    just at the end) specifically so an interrupted run doesn't lose
    already-computed points -- this sweep can take hours near a
    difficult region, and re-running it from scratch after an
    interruption previously wasted all of that time.

    Returns dict with eta, sv_smallest, sv_largest, cond, I_avg, s_max
    (arrays, one entry per CONVERGED eta -- non-converged points are
    silently skipped) and saves the same to out_path if given. I_avg/
    s_max are recorded alongside the SVD diagnostics specifically so
    this ALSO doubles as a standard-continuation (eta, I) trace directly
    comparable against the arclength-traced branch on the same plot --
    showing exactly where ordinary continuation stops relative to how
    far pseudo-arclength continuation is able to go.
    """
    P = dict(DEFAULT_PARAMS)
    if overrides:
        P.update(overrides)

    F, R = P["F"], P["R"]
    K_perm, eps_p = P["K_perm"], P["eps_p"]
    p_in, p_out = P["p_in"], P["p_out"]
    mu_g, D_O2_g, D_H2O_g = P["mu_g"], P["D_O2_g"], P["D_H2O_g"]
    I0_ref, alpha_c, alpha_w = P["I0"], P["alpha_c"], P["alpha_w"]
    T_ref_I0, Ea_orr = P["T_ref_I0"], P["Ea_orr"]
    rho_l = P["rho_l"]
    M_H2O, M_O2, M_N2 = P["M_H2O"], P["M_O2"], P["M_N2"]
    mu_l = P["mu_l"]
    T_cell, C_O2_ref = P["T_cell"], P["C_O2_ref"]
    I0 = I0_ref * np.exp(-(Ea_orr / R) * (1.0 / T_cell - 1.0 / T_ref_I0))
    RH_in = P["RH_in"]
    theta_c_deg = P["theta_c_deg"]
    theta_c = theta_c_deg * np.pi / 180.0
    sigma_st = P["sigma_st"]
    X_O2_in, X_N2_in = P["X_O2_in"], P["X_N2_in"]
    w_ch, w_rb, Lx = P["w_ch"], P["w_rb"], P["Lx"]
    n_corey, tau_brug, s_max_cap = P["n_corey"], P["tau_brug"], P["s_max"]
    s_smooth_eps = P["s_smooth_eps"]
    single_phase = P["single_phase"]
    nx, ny = P["nx"], P["ny"]

    P_sat = Psat_atm(T_cell - 273.15) * 101325.0
    Ly = w_ch + w_rb
    y_inlet_end = 0.5 * w_ch
    y_rib_end = 0.5 * w_ch + w_rb

    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [Lx, Ly]], [nx, ny], mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    def on_membrane(x): return np.isclose(x[0], 0.0)
    def on_inlet(x):    return np.isclose(x[0], Lx) & (x[1] <= y_inlet_end + 1e-12)
    def on_rib(x):      return np.isclose(x[0], Lx) & (x[1] > y_inlet_end + 1e-12) & (x[1] <= y_rib_end + 1e-12)
    def on_outlet(x):   return np.isclose(x[0], Lx) & (x[1] > y_rib_end + 1e-12)

    TAG_MEMBRANE, TAG_INLET, TAG_RIB, TAG_OUTLET = 1, 2, 3, 4
    facet_indices, facet_markers = [], []
    for tag, locator in [(TAG_MEMBRANE, on_membrane), (TAG_INLET, on_inlet),
                          (TAG_RIB, on_rib), (TAG_OUTLET, on_outlet)]:
        idx = mesh.locate_entities_boundary(domain, fdim, locator)
        facet_indices.append(idx)
        facet_markers.append(np.full_like(idx, tag))
    facet_indices = np.concatenate(facet_indices)
    facet_markers = np.concatenate(facet_markers)
    order = np.argsort(facet_indices)
    facet_tags = mesh.meshtags(domain, fdim, facet_indices[order], facet_markers[order])

    ds_meas = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    dx = ufl.Measure("dx", domain=domain)
    nvec = ufl.FacetNormal(domain)

    V = fem.functionspace(domain, ("Lagrange", 1))
    membrane_dofs_V = fem.locate_dofs_topological(V, fdim, facet_tags.find(TAG_MEMBRANE))
    P1e = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    ME = basix.ufl.mixed_element([P1e, P1e, P1e])
    W = fem.functionspace(domain, ME)

    w = fem.Function(W, name="w")
    v_te = ufl.TestFunction(W)
    p_hat_sol, Cw_sol, CgO2_sol = ufl.split(w)
    v_p, v_cw, v_cgo2 = ufl.split(v_te)
    P_scale = 1000.0
    p_sol = p_out + P_scale * p_hat_sol

    X_H2O_in = np.clip(RH_in * P_sat / p_in, 0.0, 0.98)
    dry_scale = 1.0 - X_H2O_in
    X_O2_eff = X_O2_in * dry_scale
    X_N2_eff = X_N2_in * dry_scale
    M_mix_in = X_O2_eff * M_O2 + X_N2_eff * M_N2 + X_H2O_in * M_H2O
    Y_O2_in = X_O2_eff * M_O2 / M_mix_in
    Y_H2O_in = X_H2O_in * M_H2O / M_mix_in
    M_dry_in = X_O2_in * M_O2 + X_N2_in * M_N2
    rho_g_const = p_out * M_mix_in / (R * T_cell)
    Cg_sat = (P_sat * M_H2O) / (P_sat * M_H2O + (p_in - P_sat) * M_dry_in)
    Cl_sat = 1.0

    inlet_facets = facet_tags.find(TAG_INLET)
    outlet_facets = facet_tags.find(TAG_OUTLET)
    dofs_p_in = fem.locate_dofs_topological(W.sub(0), fdim, inlet_facets)
    dofs_p_out = fem.locate_dofs_topological(W.sub(0), fdim, outlet_facets)
    dofs_cw_in = fem.locate_dofs_topological(W.sub(1), fdim, inlet_facets)
    dofs_cgo2_in = fem.locate_dofs_topological(W.sub(2), fdim, inlet_facets)
    bcs = [
        fem.dirichletbc(default_scalar_type((p_in - p_out) / P_scale), dofs_p_in, W.sub(0)),
        fem.dirichletbc(default_scalar_type(0.0), dofs_p_out, W.sub(0)),
        fem.dirichletbc(default_scalar_type(Y_H2O_in), dofs_cw_in, W.sub(1)),
        fem.dirichletbc(default_scalar_type(Y_O2_in), dofs_cgo2_in, W.sub(2)),
    ]

    def s_expr(Cw):
        if single_phase:
            return 0.0 * Cw
        raw = rho_g_const * (Cw - Cg_sat) / (rho_l * (Cl_sat - Cw) + rho_g_const * (Cw - Cg_sat) + 1e-12)
        s_smooth = 0.5 * (raw + ufl.sqrt(raw**2 + s_smooth_eps**2))
        s_smooth = ufl.max_value(s_smooth, 0.0)
        return ufl.min_value(s_smooth, s_max_cap)

    def min_smooth(a, b, eps=s_smooth_eps):
        return 0.5 * (a + b - ufl.sqrt((a - b) ** 2 + eps ** 2))

    hydrophobic = theta_c_deg >= 90.0
    s = s_expr(Cw_sol)
    krl = s ** n_corey
    krg = (1.0 - s) ** n_corey
    lam_l = (krl / mu_l) / (krl / mu_l + krg / mu_g + 1e-30)
    lam_g = 1.0 - lam_l
    rho_mix = rho_g_const * (1.0 - s) + rho_l * s
    nu_g_c, nu_l_c = mu_g / rho_g_const, mu_l / rho_l
    nu_mix = 1.0 / (krg / nu_g_c + krl / nu_l_c + 1e-30)
    mu_mix = nu_mix * rho_mix
    kappa = rho_mix * K_perm / mu_mix
    if hydrophobic:
        dJds = 1.417 - 4.240 * s + 3.789 * s**2
    else:
        u_hp = 1.0 - s
        dJds = -(1.417 - 4.240 * u_hp + 3.789 * u_hp**2)
    Dc = ((lam_l * lam_g * K_perm / (nu_mix + 1e-30))
          * sigma_st * np.cos(theta_c) * (eps_p / K_perm) ** 0.5 * dJds)
    Gamma = Dc * (1.0 - Cg_sat) / (rho_l - rho_g_const * Cg_sat)
    grads = ufl.grad(s)
    u_darcy = -(kappa / rho_mix) * ufl.grad(p_sol)
    CgH2O_local = min_smooth(Cw_sol, Cg_sat)
    Weff_w = lam_l + lam_g * CgH2O_local
    Deff_O2 = rho_g_const * (eps_p * (1.0 - s)) ** tau_brug * D_O2_g
    Diff_w = rho_g_const * (eps_p * (1.0 - s)) ** tau_brug * D_H2O_g + Gamma * rho_mix
    Jl_vec = Dc * grads
    F_conv_w = Weff_w * rho_mix * u_darcy
    W_O2 = rho_mix * lam_g * u_darcy

    eta_const = fem.Constant(domain, default_scalar_type(0.0))
    C_O2_molar = rho_g_const * CgO2_sol / M_O2
    I_local = (1.0 - s) * I0 * (C_O2_molar / C_O2_ref) * ufl.exp(alpha_c * F * eta_const / (R * T_cell))
    N_O2_expr = (M_O2 / (4.0 * F)) * I_local
    N_w_expr = (M_H2O * (1.0 + 2.0 * alpha_w) / (2.0 * F)) * I_local

    F_pres = (kappa * ufl.inner(ufl.grad(p_sol), ufl.grad(v_p)) * dx
              - (N_w_expr - N_O2_expr) * v_p * ds_meas(TAG_MEMBRANE))
    F_water = (Diff_w * ufl.inner(ufl.grad(Cw_sol), ufl.grad(v_cw)) * dx
               - ufl.inner(F_conv_w, ufl.grad(v_cw)) * dx
               + _smooth_outflow(ufl.inner(F_conv_w, nvec)) * v_cw * ds_meas(TAG_OUTLET)
               - N_w_expr * v_cw * ds_meas(TAG_MEMBRANE))
    F_oxy = (-CgO2_sol * ufl.inner(W_O2, ufl.grad(v_cgo2)) * dx
             + Deff_O2 * ufl.inner(ufl.grad(CgO2_sol), ufl.grad(v_cgo2)) * dx
             + CgO2_sol * ufl.inner(Jl_vec, ufl.grad(v_cgo2)) * dx
             + CgO2_sol * _smooth_outflow(ufl.inner(W_O2, nvec)) * v_cgo2 * ds_meas(TAG_OUTLET)
             + N_O2_expr * v_cgo2 * ds_meas(TAG_MEMBRANE))
    F_total = F_pres + F_water + F_oxy

    snes_petsc_options = {
        "snes_type": "newtonls", "snes_rtol": 1e-7, "snes_atol": 1e-9,
        "snes_max_it": max_newton_it, "snes_error_if_not_converged": False,
        "ksp_type": "preonly", "pc_type": "lu",
    }
    problem = NonlinearProblem(F_total, w, bcs=bcs, petsc_options=snes_petsc_options,
                                petsc_options_prefix="pemfc_svdsweep_",
                                form_compiler_options={"quadrature_degree": 4})

    w.sub(0).interpolate(lambda x: np.full(x.shape[1], (p_in - p_out) / P_scale))
    w.sub(1).interpolate(lambda x: np.full(x.shape[1], Y_H2O_in))
    w.sub(2).interpolate(lambda x: np.full(x.shape[1], Y_O2_in))
    w.x.scatter_forward()

    # Warm-up ramp: jumping straight to eta_list[0] (e.g. 0.30) from this
    # naive/inlet-like guess is NOT reliable (confirmed: SNES_DIVERGED_
    # LINE_SEARCH) -- every other place in this codebase that starts a
    # naive-guess solve (run_case's default eta_list) ramps up gradually
    # from a much smaller eta first. Reuse that same warm-up here,
    # silently (not recorded in the output), before the requested sweep.
    eta_list = list(eta_list)
    eta_out, sv_small_out, sv_large_out, cond_out, I_out, smax_out = [], [], [], [], [], []

    if resume and out_path and os.path.exists(out_path):
        try:
            _prior = np.load(out_path)
            eta_out = list(_prior["eta"])
            sv_small_out = list(_prior["sv_smallest"])
            sv_large_out = list(_prior["sv_largest"])
            cond_out = list(_prior["cond"])
            I_out = list(_prior["I_avg"]) if "I_avg" in _prior.files else [np.nan] * len(eta_out)
            smax_out = list(_prior["s_max"]) if "s_max" in _prior.files else [np.nan] * len(eta_out)
            covered = set(np.round(eta_out, 6))
            n_before = len(eta_list)
            eta_list = [e for e in eta_list if round(e, 6) not in covered]
            if verbose:
                for i in range(len(eta_out)):
                    print(f"eta={eta_out[i]:.5f}: smallest_sval={sv_small_out[i]:.4e}  "
                          f"cond={cond_out[i]:.4e}  I_avg={I_out[i]/1e4:.4f} A/cm^2  "
                          f"s_max={smax_out[i]:.4f}  (loaded from prior run)", flush=True)
                print(f"Resuming: {len(eta_out)} point(s) already in {out_path}, "
                      f"{n_before - len(eta_list)} requested point(s) already covered "
                      f"(skipping), {len(eta_list)} remaining.", flush=True)
        except Exception as e:
            if verbose:
                print(f"Could not load prior sweep data ({e}) -- starting fresh.", flush=True)
            eta_out, sv_small_out, sv_large_out, cond_out, I_out, smax_out = [], [], [], [], [], []

    if not eta_list:
        if verbose:
            print("Nothing left to do -- all requested points already covered.", flush=True)
        return dict(eta=np.array(eta_out), sv_smallest=np.array(sv_small_out),
                    sv_largest=np.array(sv_large_out), cond=np.array(cond_out),
                    I_avg=np.array(I_out), s_max=np.array(smax_out))

    def _save_progress():
        if out_path:
            np.savez(out_path, eta=np.array(eta_out), sv_smallest=np.array(sv_small_out),
                     sv_largest=np.array(sv_large_out), cond=np.array(cond_out),
                     I_avg=np.array(I_out), s_max=np.array(smax_out))

    # Warm-up ramp: jumping straight to the first remaining target from a
    # naive/inlet-like guess is NOT reliable (confirmed: SNES_DIVERGED_
    # LINE_SEARCH) -- every other place in this codebase that starts a
    # naive-guess solve (run_case's default eta_list) ramps up gradually
    # from a much smaller eta first. Reuse that same warm-up here,
    # silently (not recorded in the output), before the requested sweep.
    # If resuming, ramp all the way up to the LAST already-covered point
    # instead of from scratch -- we don't save full field state in this
    # sweep's output, only scalar diagnostics, so a valid warm-start
    # state has to be reconstructed either way.
    first_target = eta_out[-1] if eta_out else eta_list[0]
    eta_prev = None  # tracks wherever `problem`'s current solved state actually is,
                      # right before the main loop -- used as the step-halving base for
                      # the very first target point (previously left as None here, which
                      # silently made eta_start_for_step == eta_target for that point,
                      # collapsing the halving formula into a no-op: eta_now never
                      # actually got smaller, so a failed first attempt looked like "even
                      # the smallest sub-step failed" after only ONE real attempt)
    if first_target > 0.05:
        warmup = list(np.linspace(0.02, first_target, 8))[:-1]  # exclude the
                                                                   # last point,
                                                                   # first_target
                                                                   # covers it
        if verbose:
            print(f"Warming up from eta=0.02 to eta={first_target:.4f} "
                  f"before the requested sweep...", flush=True)
        _progress_log(f"fold_svd_sweep: warming up from eta=0.02 to eta={first_target:.4f} "
                      f"({len(warmup)} steps) before the requested sweep")
        for eta_now in warmup:
            eta_const.value = eta_now
            problem.solve()
            if problem.solver.getConvergedReason() <= 0:
                _progress_log(f"fold_svd_sweep: warm-up FAILED at eta={eta_now:.4f}")
                raise RuntimeError(
                    f"fold_svd_sweep: warm-up solve failed at eta={eta_now:.4f} -- "
                    f"cannot proceed to the requested sweep.")
            _progress_log(f"fold_svd_sweep: warm-up step converged at eta={eta_now:.4f}")
        if warmup:
            eta_prev = warmup[-1]  # the actual eta problem is currently solved at
    if eta_out:
        # Solve AT the resume point itself, to get a genuinely valid
        # warm-start state for the next (new) target below -- needed
        # regardless of the warmup ramp above (this sweep doesn't save
        # full field state, only scalar diagnostics, so this
        # reconstruction step is unconditional when resuming).
        _progress_log(f"fold_svd_sweep: re-solving at the resume point eta={first_target:.4f}")
        eta_const.value = first_target
        problem.solve()
        if problem.solver.getConvergedReason() <= 0:
            _progress_log(f"fold_svd_sweep: FAILED to re-solve at resume point eta={first_target:.4f}")
            raise RuntimeError(
                f"fold_svd_sweep: could not re-solve at the resume point "
                f"eta={first_target:.4f} -- cannot safely continue.")
        eta_prev = first_target  # overrides warmup's eta_prev -- this solve is exactly
                                  # at first_target, an even better base than warmup's

    for eta_target in eta_list:
        # Adaptive step-halving toward eta_target: if the full step fails
        # to converge, retry with progressively smaller steps from the
        # last successfully converged point, so the sweep can get as
        # close as possible to the fold before truly giving up. Added
        # because the first version of this sweep stopped well short of
        # the fold -- eta=0.355 vs the fold at ~0.36 -- missing almost
        # all of the dramatic singular-value collapse.
        eta_start_for_step = eta_prev if eta_prev is not None else eta_target
        eta_now = eta_target
        min_substep = 1e-5
        converged_this_target = False
        while True:
            eta_const.value = eta_now
            problem.solve()
            reason = problem.solver.getConvergedReason()
            if reason > 0:
                converged_this_target = (eta_now == eta_target)
                break
            step_size = abs(eta_now - eta_start_for_step)
            if step_size < min_substep:
                if verbose:
                    print(f"eta={eta_now:.5f}: did NOT converge (reason={reason}) even at "
                          f"the smallest sub-step tried -- stopping sweep here (expected "
                          f"once at/past a fold)", flush=True)
                _progress_log(f"fold_svd_sweep: stuck at eta={eta_now:.5f}, stopping")
                break
            eta_now = eta_start_for_step + 0.5 * (eta_now - eta_start_for_step)
            if verbose:
                print(f"  (target eta={eta_target:.5f} did not converge, retrying at "
                      f"eta={eta_now:.5f})", flush=True)
            _progress_log(f"fold_svd_sweep: target eta={eta_target:.5f} did not converge "
                          f"(reason={reason}), retrying at eta={eta_now:.5f}")

        if reason <= 0:
            break  # smallest sub-step also failed -- genuinely stuck, stop the whole sweep

        sv_small_raw, sv_large_raw, sv_small_sc, sv_large_sc = _jacobian_singular_values(problem)
        cond_sc = sv_large_sc[-1] / max(sv_small_sc[0], 1e-300)
        Jloc_arr = _project_scalar_standalone(I_local, V, dx)
        I_avg_now = float(np.mean(Jloc_arr[membrane_dofs_V]))
        s_arr = _project_scalar_standalone(s, V, dx)
        smax_now = float(s_arr.max())
        eta_out.append(eta_now)
        sv_small_out.append(sv_small_sc[0])
        sv_large_out.append(sv_large_sc[-1])
        cond_out.append(cond_sc)
        I_out.append(I_avg_now)
        smax_out.append(smax_now)
        eta_prev = eta_now
        _save_progress()
        if verbose:
            print(f"eta={eta_now:.5f}: smallest_sval={sv_small_sc[0]:.4e}  cond={cond_sc:.4e}  "
                  f"I_avg={I_avg_now/1e4:.4f} A/cm^2  s_max={smax_now:.4f}",
                  flush=True)
        _progress_log(f"fold_svd_sweep: eta={eta_now:.5f} smallest_sval={sv_small_sc[0]:.4e} "
                      f"cond={cond_sc:.4e} I_avg={I_avg_now/1e4:.4f} s_max={smax_now:.4f}")
        if not converged_this_target and verbose:
            print(f"Target eta={eta_target:.5f} not reached directly -- recorded the "
                  f"closest convergeable point instead (eta={eta_now:.5f}), continuing "
                  f"to the next target.", flush=True)

    result = dict(eta=np.array(eta_out), sv_smallest=np.array(sv_small_out),
                  sv_largest=np.array(sv_large_out), cond=np.array(cond_out),
                  I_avg=np.array(I_out), s_max=np.array(smax_out))
    if out_path:
        np.savez(out_path, **result)
        if verbose:
            print(f"\nSaved to {out_path}", flush=True)
    return result


def collect_diagnostic_summary(entries, out_path=None, verbose=True):
    """Run diagnose_fold_point() over a labeled list of checkpoints and
    compile the key comparison numbers into one clean summary table --
    e.g. the confirmed genuine fold vs. the second (bistable) stuck
    point, at coarse and refined mesh resolution -- ready for direct use
    building a paper table without hunting through progress logs.

    entries: list of dicts, each with:
        "label"      -- short name for this row (e.g. "second point, coarse mesh")
        "checkpoint" -- path to the checkpoint .npz to diagnose
        "overrides"  -- (optional) dict, e.g. {"nx": 36, "ny": 90}; must
                        match whatever mesh that checkpoint was generated with

    Saves (if out_path given) a JSON file (human-readable, easy to turn
    into a LaTeX/Word table directly) with one row per entry:
        label, eta, cond_ratio, smallest_sv_ratio, transversality, verdict

    Returns the same list of row-dicts.
    """
    rows = []
    for entry in entries:
        label = entry["label"]
        ckpt = entry["checkpoint"]
        overrides = entry.get("overrides")
        if verbose:
            print(f"\n{'='*70}\nDiagnosing: {label}  ({ckpt})\n{'='*70}", flush=True)
        try:
            diag = diagnose_fold_point(ckpt, overrides=overrides, verbose=verbose)
            rows.append(dict(
                label=label,
                checkpoint=ckpt,
                eta=diag["stuck"]["eta"],
                cond_ratio=diag["cond_ratio"],
                smallest_sv_ratio=diag["smallest_ratio"],
                transversality=diag["stuck"]["transversality_normalized"],
                verdict=diag["verdict"],
            ))
        except Exception as e:
            if verbose:
                print(f"  FAILED to diagnose {label}: {type(e).__name__}: {e}", flush=True)
            rows.append(dict(label=label, checkpoint=ckpt, error=str(e)))

    if verbose:
        print(f"\n{'='*70}\nSummary\n{'='*70}")
        for r in rows:
            if "error" in r:
                print(f"{r['label']}: FAILED ({r['error']})")
            else:
                print(f"{r['label']}: eta={r['eta']:.5f}  cond_ratio={r['cond_ratio']:.2f}x  "
                      f"transversality={r['transversality']:.4f}")

    if out_path:
        import json
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)
        if verbose:
            print(f"\nSaved to {out_path}", flush=True)

    return rows


def run_case_to_target_eta(overrides=None, target_eta=0.55, eta_ramp_start=0.30,
                             ds=0.02, ds_min=1e-4, ds_max=0.1, steps_per_batch=5,
                             max_arclength_steps=300, eta_tol=1e-3, verbose=True):
    """Lightweight utility for parameter studies: reach target_eta (default
    0.55, matching this paper's own parametric-study comparison point --
    see e.g. "at eta=0.55" in every Fig. 11/12/14/16/17/19/20/22/24/25/27
    caption) as cheaply as possible, WITHOUT the heavy paper-backing-data
    infrastructure (no SVD tracking, no per-step field snapshots).

    Two-path strategy, since whether target_eta needs arclength at all
    depends on the parameter value -- the fold's location is NOT fixed,
    it can shift well above or below target_eta depending on the swept
    parameter (e.g. porosity, wettability):

    1. Try ordinary (eta-fixed) Newton continuation straight to
       target_eta first. Fast (a normal run_case() sweep), and works
       whenever this parameter combination's fold (if any) sits above
       target_eta.
    2. If that stalls before reaching target_eta, fall back to
       pseudo-arclength continuation (run_case_arclength), resuming from
       wherever ordinary Newton got to, stepping in small batches and
       checking after each whether eta has come within eta_tol of
       target_eta. Stops as soon as it does, or after
       max_arclength_steps, whichever first -- does NOT guarantee
       reaching target_eta (the traced branch may turn around and never
       come back to it, as confirmed for the base case's own branch,
       which peaks near eta~0.36 then decreases). Returns the CLOSEST
       approach either way, clearly flagged via target_reached.

    Returns dict with:
        eta_hist, I_hist, s_max_hist -- full accepted-step trace (ordinary
                                         Newton portion + arclength portion
                                         if used)
        target_reached  -- bool, True only if the closest approach was
                            within eta_tol of target_eta
        best_eta         -- actual eta of the returned field snapshot
        gap              -- |best_eta - target_eta|
        fields           -- dict of field arrays at best_eta (same layout
                             as save_field_snapshot's output)
        method           -- "newton" or "arclength", which path supplied
                             the returned fields
    """
    import tempfile

    newton_eta_list = np.concatenate([np.linspace(0.02, eta_ramp_start, 8),
                                       np.linspace(eta_ramp_start, target_eta, 6)])
    if verbose:
        print(f"[run_case_to_target_eta] Trying ordinary Newton up to eta={target_eta}...",
              flush=True)
    result = run_case(overrides=overrides, eta_list=newton_eta_list, verbose=False,
                       save_fields_at=(target_eta,))
    tag = f"eta{target_eta:.3f}"

    if tag in result["fields"] and abs(result["eta"][-1] - target_eta) < eta_tol:
        if verbose:
            print(f"[run_case_to_target_eta] Reached target directly via ordinary Newton "
                  f"(no fold before eta={target_eta} for this parameter set).", flush=True)
        return dict(eta_hist=result["eta"], I_hist=result["I_avg"], s_max_hist=result["s_max"],
                    target_reached=True, best_eta=float(result["eta"][-1]), gap=0.0,
                    fields=result["fields"][tag], method="newton")

    eta_reached_by_newton = float(result["eta"][-1]) if len(result["eta"]) else eta_ramp_start
    if verbose:
        print(f"[run_case_to_target_eta] Ordinary Newton stopped at eta={eta_reached_by_newton:.4f} "
              f"(target {target_eta}) -- falling back to arclength continuation...", flush=True)

    scratch_ckpt = tempfile.mktemp(suffix="_scratch_arclength.npz")
    best_eta, best_gap = eta_reached_by_newton, abs(eta_reached_by_newton - target_eta)
    total_steps, resumed = 0, False
    try:
        while total_steps < max_arclength_steps:
            r = run_case_arclength(overrides=overrides, eta_start=eta_reached_by_newton, ds=ds,
                                    ds_min=ds_min, ds_max=ds_max, n_steps=steps_per_batch,
                                    resume=resumed, checkpoint_path=scratch_ckpt, verbose=False)
            resumed = True
            total_steps += steps_per_batch
            eta_now = float(r["eta"][-1])
            gap = abs(eta_now - target_eta)
            if gap < best_gap:
                best_eta, best_gap = eta_now, gap
            if gap < eta_tol:
                if verbose:
                    print(f"[run_case_to_target_eta] Reached target via arclength "
                          f"(eta={eta_now:.4f}) after ~{total_steps} step(s).", flush=True)
                break
        else:
            if verbose:
                print(f"[run_case_to_target_eta] Exhausted ~{max_arclength_steps} arclength "
                      f"steps without reaching eta={target_eta} (closest: {best_eta:.4f}, "
                      f"gap={best_gap:.4f}). Returning closest approach.", flush=True)

        tmp_field_path = tempfile.mktemp(suffix="_scratch_fields.npz")
        raw_fields = save_field_snapshot(scratch_ckpt, tmp_field_path, overrides=overrides,
                                          verbose=False)
        d = np.load(scratch_ckpt)
        eta_hist_full = np.concatenate([result["eta"], d["eta_hist"][1:]])
        I_hist_full = np.concatenate([result["I_avg"], d["I_hist"][1:]])
        smax_hist_full = np.concatenate([result["s_max"], d["smax_hist"][1:]])

        # Post-process into the same membrane_y/s/I/CgO2 + interface_y/x
        # format run_case's save_fields_at produces (save_field_snapshot
        # only gives raw 2D fields, no membrane-line extraction or local
        # current density) -- makes this drop-in compatible with
        # run_parametric_study()'s expected per-value result format.
        P_local = dict(DEFAULT_PARAMS)
        if overrides:
            P_local.update(overrides)
        I0_arc = P_local["I0"] * np.exp(
            -(P_local["Ea_orr"] / P_local["R"]) * (1.0 / P_local["T_cell"] - 1.0 / P_local["T_ref_I0"]))
        rho_g_const_arc = P_local["p_out"] * (
            P_local["X_O2_in"] * P_local["M_O2"] + P_local["X_N2_in"] * P_local["M_N2"]
        ) / (P_local["R"] * P_local["T_cell"])
        C_O2_molar_arr = rho_g_const_arc * raw_fields["CgO2"] / P_local["M_O2"]
        I_local_arr = ((1.0 - raw_fields["s"]) * I0_arc * (C_O2_molar_arr / P_local["C_O2_ref"])
                        * np.exp(P_local["alpha_c"] * P_local["F"] * best_eta
                                 / (P_local["R"] * P_local["T_cell"])))
        mem = extract_line(raw_fields["coords"], {"s": raw_fields["s"], "CgO2": raw_fields["CgO2"],
                                                   "I": I_local_arr}, x_target=0.0)
        interface_y, interface_x = extract_interface(raw_fields["coords"], raw_fields["s"])
        fields = dict(membrane_y=mem["y"], membrane_s=mem["s"], membrane_I=mem["I"],
                      membrane_CgO2=mem["CgO2"], interface_y=interface_y, interface_x=interface_x)
    finally:
        for p in [scratch_ckpt, locals().get("tmp_field_path")]:
            if p:
                try:
                    os.remove(p)
                except Exception:
                    pass

    return dict(eta_hist=eta_hist_full, I_hist=I_hist_full, s_max_hist=smax_hist_full,
                target_reached=best_gap < eta_tol, best_eta=best_eta, gap=best_gap,
                fields=fields, method="arclength")


def run_parametric_study_to_target_eta(param_key, values, target_eta=0.55,
                                         eta_ramp_start=0.30, verbose=True, **base_overrides):
    """Like run_parametric_study(), but aims for target_eta (default 0.55,
    matching this paper's own parametric-study comparison point) using
    run_case_to_target_eta() per value -- ordinary Newton first, falling
    back to lightweight pseudo-arclength continuation only for whichever
    parameter values actually need it to get near target_eta. Much
    cheaper than the full paper-backing-data arclength infrastructure
    (no SVD tracking, no per-step snapshots) since this only needs ONE
    good field snapshot per parameter value, not the whole traced branch.

    Returns the same {value: {eta, I_avg, s_max, iters, membrane_y,
    membrane_s, membrane_I, membrane_CgO2, interface_y, interface_x}}
    format as run_parametric_study(), so it's a drop-in replacement for
    save_study_npz() and the existing plotting cells -- plus each
    per-value dict also carries `target_reached` (bool) and `gap` (float)
    so the plots/tables can flag which parameter values didn't quite
    make it to target_eta.
    """
    results = {}
    for v in values:
        if verbose:
            print(f"\n{'='*70}\n{param_key} = {v}\n{'='*70}", flush=True)
        overrides = dict(base_overrides)
        overrides[param_key] = v
        r = run_case_to_target_eta(overrides=overrides, target_eta=target_eta,
                                    eta_ramp_start=eta_ramp_start, verbose=verbose)
        fld = r["fields"] or {}
        results[v] = dict(
            eta=r["eta_hist"], I_avg=r["I_hist"], s_max=r["s_max_hist"],
            iters=np.ones_like(r["eta_hist"]),  # placeholder: not tracked uniformly across both
                                                 # paths (see method/gap instead), but MUST match
                                                 # eta_hist's length for existing plot cells'
                                                 # `iters < MAX_ITERS_CAP` filters to work correctly
            membrane_y=fld.get("membrane_y"), membrane_s=fld.get("membrane_s"),
            membrane_I=fld.get("membrane_I"), membrane_CgO2=fld.get("membrane_CgO2"),
            interface_y=fld.get("interface_y"), interface_x=fld.get("interface_x"),
            target_reached=np.array([r["target_reached"]]), gap=np.array([r["gap"]]),
            method=np.array([r["method"]]))
        if verbose:
            status = "reached" if r["target_reached"] else f"NOT reached (gap={r['gap']:.4f})"
            print(f"[{param_key}={v}] target_eta={target_eta} {status} via {r['method']}", flush=True)
    return results
