"""Paper Eqs.24-26 and 28-29; stable even for the literal Table-I site area."""
import numpy as np
from config import thermal_speed


def scales(cfg, volume_to_reactive_area, velocity=None):
    u = cfg.mean_velocity if velocity is None else velocity
    k = thermal_speed(cfg.precursor_molar_mass,cfg.temperature)*cfg.beta0/4
    return cfg.gamma/(k*cfg.c0), volume_to_reactive_area*u/k


def final_coverage(x, dose, cfg, volume_to_reactive_area, velocity=None):
    tau, length = scales(cfg,volume_to_reactive_area,velocity)
    a = np.asarray(dose)/tau
    # log(expm1(a)), without exp(a) overflow or cancellation at a=0.
    with np.errstate(divide="ignore",invalid="ignore"):
        log_numerator = a + np.log(-np.expm1(-a))
    return np.exp(log_numerator-np.logaddexp(np.asarray(x)/length,log_numerator))


def transient_coverage(x,t,cfg,volume_to_reactive_area,velocity=None):
    """Rectangular pulse, initially clean and empty, no axial diffusion.

    Eq.28 with retarded time; cap exposure at td after the trailing front.
    Not an instantaneous benchmark for a ramped inlet.
    """
    u = cfg.mean_velocity if velocity is None else velocity
    exposure = np.minimum(np.maximum(t-np.asarray(x)/u,0),cfg.pulse_duration)
    return final_coverage(x,exposure,cfg,volume_to_reactive_area,u)
