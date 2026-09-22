"""Local constitutive laws independent of geometry, FEM and PETSc.

State rows: precursor/fast, slow-site, blocked-byproduct coverage.
All reaction fluxes are mol m^-2 s^-1; gas concentrations mol m^-3.
"""
import numpy as np
from config import thermal_speed


def rate_constants(T, parameters):
    p = parameters
    vp = thermal_speed(p.precursor_molar_mass, T)/4
    vb = thermal_speed(p.byproduct_molar_mass, T)/4
    return vp*p.beta0, vp*p.beta_slow, vb*p.beta_byproduct


def closure_multiplier(c, theta, T, parameters):
    """Extension hook g_phi >= 0. Fixed to one in this baseline.

    Replace with a frozen, positive learned correction. The same callback is
    used by both the boundary flux and surface update, in 2D and 3D.
    """
    return np.ones_like(c)


def coverage(state, parameters):
    if parameters.kinetics_model == "soft_saturation":
        f = parameters.slow_fraction
        return (1-f)*state[0] + f*state[1]
    return state[0]


def coefficients(c, state, T, parameters):
    """Return molar Robin velocities (precursor, byproduct), both m/s."""
    kp, ks, kb = rate_constants(T, parameters)
    g = closure_multiplier(c, coverage(state,parameters), T, parameters)
    if np.any(~np.isfinite(g)) or np.any(g < 0):
        raise ValueError("Surface closure must be finite and nonnegative")
    model = parameters.kinetics_model
    if model == "ideal":
        return kp*g*(1-state[0]), np.zeros_like(c)
    if model == "soft_saturation":
        f = parameters.slow_fraction
        return g*((1-f)*kp*(1-state[0])+f*ks*(1-state[1])), np.zeros_like(c)
    empty = 1-state[0]-state[2]
    return kp*g*empty, kb*empty


def reaction_rate(c, theta, T, parameters, byproduct=None):
    """Closure interface. theta is the complete three-row local state."""
    ap, ab = coefficients(c, theta, T, parameters)
    cb = np.zeros_like(c) if byproduct is None else byproduct
    return ap*c, ab*cb


def implicit_update(old, c, byproduct, dt, parameters, guess=None):
    """Backward Euler, analytically eliminated surface unknowns.

    For nonconstant learned g, evaluate at the nonlinear outer iterate.
    No concentration or coverage clipping is performed here.
    """
    p = parameters
    kp, ks, kb = rate_constants(p.temperature, p)
    g = closure_multiplier(c, coverage(old if guess is None else guess,p),p.temperature,p)
    a, b = dt*kp*g*c/p.gamma, dt*kb*byproduct/p.gamma
    new = old.copy()
    if p.kinetics_model == "competitive_adsorption":
        empty = (1-old[0]-old[2])/(1+a+b)
        new[0] = old[0]+a*empty
        new[2] = old[2]+b*empty
    else:
        new[0] = (old[0]+a)/(1+a)
        if p.kinetics_model == "soft_saturation":
            slow = dt*ks*g*c/p.gamma
            new[1] = (old[1]+slow)/(1+slow)
    return new
