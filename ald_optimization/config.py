"""SI parameters. Every physical choice has provenance in parameter_manifest()."""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import argparse
import math

R = 8.31446261815324
NA = 6.02214076e23
TORR = 133.32236842105263


@dataclass(frozen=True)
class Config:
    temperature: float = 473.0
    pressure: float = TORR
    precursor_molar_mass: float = 0.150  # kg/mol; paper 150 amu
    byproduct_molar_mass: float = 0.016  # ASSUMPTION: methane-like surrogate
    carrier_molar_mass: float = 0.0280134  # ASSUMPTION: nitrogen
    diffusivity: float = 0.01
    byproduct_diffusivity: float = 0.05
    kinematic_viscosity: float = 0.05  # m²/s, dimensional interpretation of Table I
    grad_div_factor: float = 10.0  # numerical stabilization multiplier, not a material property
    beta0: float = 1e-2
    beta_slow: float = 1e-4
    slow_fraction: float = 0.1
    beta_byproduct: float = 1e-3
    byproduct_yield: float = 1.0
    # Explicit working assumption, NOT a verified correction of the paper.
    site_area: float = 24e-20  # 24 Å² = 0.24 nm²; literal table is 24e-18 m²
    site_area_mode: str = "angstrom24"
    precursor_pressure: float = 20e-3 * TORR
    length: float = 0.50
    height: float = 0.020
    wafer_start: float = 0.10
    wafer_end: float = 0.40
    mean_velocity: float = 0.50
    chamber_radius: float = 0.250
    wafer_radius: float = 0.150
    port_radius: float = 0.015
    port_offset: float = 0.220  # ASSUMPTION: bottom-facing port centers
    flow_sccm: float = 300.0
    standard_temperature: float = 273.15  # ASSUMPTION: definition of sccm
    standard_pressure: float = 101325.0
    pulse_duration: float = 0.20  # equivalent rectangular dose
    rise_time: float = 0.02  # ASSUMPTION, pulse ends at dose + rise
    purge_duration: float = 1.0  # starts AFTER the falling ramp
    dt: float = 0.005
    nx: int = 120
    ny: int = 12
    mesh_size_3d: float = 0.012
    port_mesh_size: float = 0.004
    kinetics_model: str = "ideal"
    flow_model: str = "navier_stokes"
    inlet_condition: str = "flux"  # conservative Danckwerts; see README
    max_coupling_iterations: int = 100
    coupling_tolerance: float = 2e-9
    relaxation: float = 0.8
    min_dt: float = 1e-7
    linear_rtol: float = 1e-11
    negative_tolerance: float = 1e-9  # relative to inlet concentration
    output_root: str = "outputs"

    @property
    def c0(self):
        return self.precursor_pressure / (R * self.temperature)

    @property
    def gamma(self):
        """Molar capacity [mol sites/m²], one precursor molecule/site."""
        return 1.0 / (NA * self.site_area)

    @property
    def end_time(self):
        return self.pulse_duration + self.rise_time + self.purge_duration

    @property
    def actual_volume_flow(self):
        return (self.flow_sccm * 1e-6 / 60 * self.temperature
                / self.standard_temperature * self.standard_pressure / self.pressure)

    @property
    def density(self):
        return self.pressure * self.carrier_molar_mass / (R * self.temperature)

    def validate(self):
        positive = (self.temperature, self.pressure, self.site_area, self.dt,
                    self.height, self.length, self.diffusivity, self.mean_velocity)
        if min(positive) <= 0 or not 0 <= self.slow_fraction <= 1:
            raise ValueError("Positive dimensional parameters and 0 <= f <= 1 required")
        if not 0 <= self.rise_time <= self.pulse_duration or self.purge_duration < 0:
            raise ValueError("Require 0 <= rise_time <= pulse_duration and purge >= 0")
        if self.kinetics_model not in ("ideal", "soft_saturation", "competitive_adsorption"):
            raise ValueError("Unknown kinetics_model")
        if self.inlet_condition not in ("flux", "dirichlet"):
            raise ValueError("Unknown inlet_condition")
        if any(not 0 <= b <= 1 for b in (self.beta0,self.beta_slow,self.beta_byproduct)):
            raise ValueError("Sticking probabilities must lie in [0,1]")
        if min(self.nx,self.ny)<2 or min(self.mesh_size_3d,self.port_mesh_size)<=0:
            raise ValueError("Invalid mesh resolution")
        if self.port_offset+self.port_radius>=self.chamber_radius or self.port_offset-self.port_radius<=self.wafer_radius:
            raise ValueError("3D ports must lie inside the chamber and outside the wafer")
        return self


def thermal_speed(molar_mass, temperature):
    return math.sqrt(8 * R * temperature / (math.pi * molar_mass))


def pulse_integral(t, cfg):
    """Integral of unit-height trapezoid: area exactly pulse_duration [s]."""
    t = max(0.0, float(t))
    tr, td = cfg.rise_time, cfg.pulse_duration
    if tr == 0:
        return min(t, td)
    if t < tr:
        return t*t / (2*tr)
    if t < td:
        return t - tr/2
    if t < td+tr:
        a = t-td
        return td-tr/2 + a-a*a/(2*tr)
    return td


def pulse_average(t0, t1, cfg):
    return cfg.c0 * (pulse_integral(t1, cfg)-pulse_integral(t0, cfg))/(t1-t0)


def parameter_manifest(cfg):
    paper = {
        "temperature": "Table I", "pressure": "III.A: 1 Torr",
        "precursor_molar_mass": "Table I: 150 amu -> 0.150 kg/mol",
        "diffusivity": "Table I", "byproduct_diffusivity": "Table I",
        "kinematic_viscosity": "Table I: value 0.05; printed m/s corrected dimensionally to m²/s",
        "chamber_radius": "Fig.1 / III.A: diameter 50 cm",
        "height": "III.A: wafer reactor height 2 cm (also used for planar surrogate)",
        "wafer_radius": "III.A: diameter 300 mm", "port_radius": "III.A: diameter 30 mm",
        "flow_sccm": "III.A", "beta0": "Fig.3 baseline / selected reaction probability",
        "beta_slow": "III.C.1 / Fig.11", "slow_fraction": "Fig.11",
        "beta_byproduct": "III.C.2: one of the explored values",
        "byproduct_yield": "III.C.2: one ligand per precursor",
        "precursor_pressure": "scenario-selected: Fig.3 20, Fig.5 75, Fig.10/13 50 mTorr",
    }
    entries = {}
    for key, val in asdict(cfg).items():
        entries[key] = {"value": val, "source": paper.get(key, "ASSUMPTION / numerical or scenario choice")}
    entries["site_area"]["source"] = (
        "Table I literal 24 nm²" if cfg.site_area_mode == "table"
        else "ASSUMPTION: interpret printed 24 as Å²; unverified, 100x smaller than Table I")
    entries["derived"] = {"c0_mol_m3": cfg.c0, "capacity_mol_m2": cfg.gamma,
        "thermal_speed_m_s": thermal_speed(cfg.precursor_molar_mass, cfg.temperature),
        "actual_flow_m3_s": cfg.actual_volume_flow, "density_kg_m3": cfg.density}
    return entries


def arguments(description, defaults=None):
    cfg = defaults or Config()
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quick", action="store_true", help="Coarse smoke run; not mesh-converged")
    p.add_argument("--site-area", choices=["angstrom24", "table"], default=cfg.site_area_mode)
    p.add_argument("--model", choices=["ideal", "soft_saturation", "competitive_adsorption"], default=cfg.kinetics_model)
    p.add_argument("--dt", type=float, default=cfg.dt)
    p.add_argument("--dose", type=float, default=cfg.pulse_duration)
    p.add_argument("--purge", type=float, default=cfg.purge_duration)
    p.add_argument("--pressure-mtorr", type=float, default=cfg.precursor_pressure/TORR*1000)
    p.add_argument("--nx", type=int, default=cfg.nx)
    p.add_argument("--ny", type=int, default=cfg.ny)
    p.add_argument("--output", default=cfg.output_root)
    p.add_argument("--inlet", choices=["flux", "dirichlet"], default=cfg.inlet_condition)
    p.add_argument("--flow", choices=["navier_stokes", "stokes"], default=cfg.flow_model)
    p.add_argument("--mesh-size", type=float, default=cfg.mesh_size_3d)
    args = p.parse_args()
    cfg = replace(cfg, site_area_mode=args.site_area,
        site_area=24e-18 if args.site_area == "table" else 24e-20,
        kinetics_model=args.model, dt=args.dt, pulse_duration=args.dose,
        rise_time=min(cfg.rise_time, args.dose), purge_duration=args.purge,
        precursor_pressure=args.pressure_mtorr*1e-3*TORR,
        nx=args.nx, ny=args.ny, output_root=args.output,
        inlet_condition=args.inlet, flow_model=args.flow, mesh_size_3d=args.mesh_size)
    if args.quick:
        cfg = replace(cfg, nx=min(cfg.nx,60), ny=min(cfg.ny,6), dt=max(cfg.dt,0.02),
                      mesh_size_3d=max(cfg.mesh_size_3d,0.055), port_mesh_size=0.012)
    return cfg.validate(), args
