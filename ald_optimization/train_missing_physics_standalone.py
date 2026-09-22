#!/usr/bin/env python3
"""Explicit hidden-state ALD FEML, standalone learner (two-file project).

GT generator remains unchanged. This file loads its saved sparse FEM operators.
Only cp, cb, theta_p supervise training. Predicted theta_b_hat is an internal
state; GT theta_b is opened only in separate --oracle-theta-b mode or after
training is frozen. No coordinates, condition IDs, or time enter the networks.

Examples (from the directory containing both standalone files):
  python gt_competitive_standalone.py --preset multi --quick
  python train_missing_physics_standalone.py --self-test
  python train_missing_physics_standalone.py --oracle-theta-b
  python train_missing_physics_standalone.py --epochs 5 --output outputs/09_hidden_state_smoke
  python train_missing_physics_standalone.py --epochs 200

The default training output is outputs/09_hidden_state_feml. Existing memoryless
outputs are protected. --self-test exercises manufactured algebra, not ALD data.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
try:
    import torch
    from torch import nn
    from torch.nn.utils import parameters_to_vector, vector_to_parameters
    TORCH_IMPORT_ERROR = None
except (ImportError, OSError) as exc:
    torch = nn = None
    TORCH_IMPORT_ERROR = str(exc)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========================= CONFIG / PHYSICS =========================
"""SI parameters. Every physical choice has provenance in parameter_manifest()."""

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
        if self.kinetics_model not in ("ideal", "soft_saturation", "competitive_adsorption", "learned_closure"):
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


def stage5_config():
    """Single source of truth for the existing non-ideal comparison case."""
    return replace(Config(), precursor_pressure=50e-3*TORR,
                   pulse_duration=0.8, purge_duration=1.0)


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


# ========================= GT DATA STRUCTURES =========================

@dataclass
class Operators:
    B: sparse.csr_matrix
    Bb: sparse.csr_matrix
    mass: np.ndarray
    surface: np.ndarray
    wafer: np.ndarray
    inlet: np.ndarray
    outlet: np.ndarray
    coordinates: np.ndarray


@dataclass
class Observation:
    name: str
    cfg: Config
    times: np.ndarray
    cp: np.ndarray
    cb: np.ndarray
    theta: np.ndarray
    active: np.ndarray
    role: str

    def training_mask(self):
        i = np.arange(len(self.times))
        return (i > 0) & (i % 5 != 0)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_operators(path: Path) -> Operators:
    with np.load(path) as d:
        B = sparse.csr_matrix(
            (d["B_data"], d["B_indices"], d["B_indptr"]),
            shape=tuple(d["B_shape"].tolist()),
        )
        Bb = sparse.csr_matrix(
            (d["Bb_data"], d["Bb_indices"], d["Bb_indptr"]),
            shape=tuple(d["Bb_shape"].tolist()),
        )
        return Operators(
            B=B,
            Bb=Bb,
            mass=d["mass"],
            surface=d["surface"],
            wafer=d["wafer"],
            inlet=d["inlet"],
            outlet=d["outlet"],
            coordinates=d["coordinates"],
        )


def load_case(folder: Path) -> Observation:
    meta = json.loads((folder / "config.json").read_text())
    cfg = Config(**meta["config"])
    with np.load(folder / "fields.npz") as d:
        return Observation(
            name=meta["name"],
            cfg=cfg,
            times=d["times"],
            cp=d["cp_mol_m3"],
            cb=d["cb_mol_m3"],
            theta=d["theta_p"],
            active=d["surface_dofs"].astype(int),
            role=meta["role"],
        )


def require_torch():
    if torch is None:
        raise RuntimeError("Training requires CPU PyTorch: " + str(TORCH_IMPORT_ERROR)
                           + ". Oracle and --self-test can run without Torch.")


def known_kp(cfg):
    # The hidden GT beta_byproduct is deliberately not used by the learner.
    return cfg.beta0 * thermal_speed(cfg.precursor_molar_mass, cfg.temperature) / 4


class HiddenRateNet(nn.Module if nn is not None else object):
    """Two small positive rate networks, no GT adsorption parameters or states."""
    def __init__(self, concentration_ref, time_ref=1., width=32, depth=3):
        require_torch()
        super().__init__()
        self.width, self.depth = width, depth
        self.register_buffer("concentration_ref",torch.tensor(float(concentration_ref),dtype=torch.float64))
        self.register_buffer("time_ref",torch.tensor(float(time_ref),dtype=torch.float64))
        def network(initial_positive_output):
            layers=[]
            for i in range(depth):
                layers += [nn.Linear(4 if i == 0 else width,width),nn.Tanh()]
            layers += [nn.Linear(width,1)]
            result=nn.Sequential(*layers).double()
            nn.init.normal_(result[-1].weight,std=.01)
            nn.init.constant_(result[-1].bias,math.log(math.expm1(initial_positive_output)))
            return result
        # Generic initial guesses: v_ads=0.1*kp, removal=0.01/time_ref.
        # Neither is derived from GT beta_byproduct or hidden theta_b.
        self.ads=network(.1)
        self.rem=network(.01)

    def forward(self,x):
        return torch.cat([torch.nn.functional.softplus(self.ads(x)),
                          torch.nn.functional.softplus(self.rem(x))],dim=-1)

    def specification(self):
        return dict(inputs=["cp/c_ref","cb/c_ref","theta_p","predicted_theta_b_hat"],
            outputs=["positive_adsorption_multiplier","positive_removal_multiplier"],
            architecture=f"two independent 4-{'-'.join([str(self.width)]*self.depth)}-1 MLPs",
            width=self.width,depth=self.depth,activation="tanh",output_activation="softplus",
            concentration_ref_mol_m3=float(self.concentration_ref),time_ref_s=float(self.time_ref),
            rate_definition="f=(kp*c_ref/Gamma)*(cb/c_ref)*free*q_ads - theta_b_hat*q_rem/time_ref",
            initial_ads_multiplier=.1,initial_rem_multiplier=.01,dtype="float64",device="cpu")


class RateAdapter:
    def __init__(self,net):
        self.net=net

    def value_tangent(self,xi):
        x=torch.tensor(xi,dtype=torch.float64,requires_grad=True)
        q=self.net(x)
        derivative=[]
        for j in range(2):
            tangent,=torch.autograd.grad(q[:,j].sum(),x,retain_graph=(j==0))
            derivative.append(tangent.detach().numpy())
        return q.detach().numpy(),np.stack(derivative,axis=1)

    def value(self,xi):
        with torch.no_grad():
            return self.net(torch.as_tensor(xi,dtype=torch.float64)).numpy()

    def vjp(self,xi,cotangent):
        self.net(torch.as_tensor(xi,dtype=torch.float64)).backward(
            torch.as_tensor(cotangent,dtype=torch.float64))


@dataclass
class Trajectory:
    times: np.ndarray
    states: np.ndarray
    factors: list
    iterations: list
    residuals: list
    diagnostics: dict


class TransientFEM:
    """BE on the exported conservative P1 FEM with two coupled surface states.

    u=(cp/c_ref, cb/c_ref, theta_p, theta_b_hat). Residuals are divided by
    their storage coefficients, so F_(u_previous)=-I. Hp multiplies Rp/c_ref
    [m/s]; Hf multiplies d(theta_b_hat)/dt [1/s]. Gas cb receives +Gamma*f
    on the residual side: adsorption is a sink and removal is a source.
    """
    def __init__(self,ops,cfg,concentration_ref,adapter=None,time_ref=1.,
                 mode="hidden",tolerance=2e-11):
        if cfg.inlet_condition != "flux":
            raise ValueError("This inverse model uses the GT Danckwerts flux inlet")
        if mode not in ("hidden","ideal","oracle"):
            raise ValueError("Unknown solver mode")
        if mode=="hidden" and adapter is None:
            raise ValueError("Hidden dynamics require a rate adapter")
        self.ops,self.cfg,self.ref,self.adapter=ops,cfg,float(concentration_ref),adapter
        self.mode,self.time_ref,self.tolerance=mode,float(time_ref),float(tolerance)
        self.n=len(ops.mass)
        self.active=np.flatnonzero(ops.surface>0)
        self.s=len(self.active)
        self.size=2*self.n+2*self.s
        self.ps=slice(2*self.n,2*self.n+self.s)
        self.zs=slice(2*self.n+self.s,self.size)
        self.kp=known_kp(cfg)
        self.kappa=self.kp*self.ref/cfg.gamma
        self.rem_scale=1/self.time_ref
        if not self.s or np.any(ops.mass<=0) or min(self.ref,self.time_ref,cfg.gamma)<=0:
            raise ValueError("Positive weights, references, capacity, and reactive surface required")
        a=self.active; j=np.arange(self.s); w=ops.surface[a]/ops.mass[a]
        self.Hp=sparse.csc_matrix((np.r_[w,-cfg.byproduct_yield*w,
                np.full(self.s,-self.ref/cfg.gamma)],
                (np.r_[a,self.n+a,2*self.n+j],np.tile(j,3))),shape=(self.size,self.s))
        self.Hf=sparse.csc_matrix((np.r_[w*cfg.gamma/self.ref,-np.ones(self.s)],
                (np.r_[self.n+a,2*self.n+self.s+j],np.tile(j,2))),shape=(self.size,self.s))
        self.G=sparse.block_diag([sparse.diags(1/ops.mass)@ops.B,
                sparse.diags(1/ops.mass)@ops.Bb,sparse.csr_matrix((2*self.s,2*self.s))],format="csc")
        self.I=sparse.eye(self.size,format="csc")
        self._cache={}

    def local_state(self,u):
        return np.stack([u[self.active],u[self.n+self.active],u[self.ps],u[self.zs]],axis=1)

    def local_rates(self,xi,jacobian=True,oracle_rate=None):
        a,b,p,z=xi.T
        free=1-p-z
        rp=self.kp*a*free
        dr=np.column_stack([self.kp*free,np.zeros(self.s),-self.kp*a,-self.kp*a])
        if self.mode in ("ideal","oracle"):
            f=np.zeros(self.s) if self.mode=="ideal" else np.asarray(oracle_rate)
            if f.shape != (self.s,):
                raise ValueError("Oracle requires the GT hidden-state increment at each step")
            return rp,f,dr,np.zeros((self.s,4))
        if jacobian:
            q,dq=self.adapter.value_tangent(xi)
        else:
            q=self.adapter.value(xi)
        if q.shape != (self.s,2) or np.any(~np.isfinite(q)) or np.any(q<0):
            raise ArithmeticError("Rate network returned invalid positive multipliers")
        ads=self.kappa*b*free
        rem=self.rem_scale*z
        f=ads*q[:,0]-rem*q[:,1]
        if not jacobian:
            return rp,f,None,None
        df=ads[:,None]*dq[:,0,:]-rem[:,None]*dq[:,1,:]
        df[:,1]+=self.kappa*free*q[:,0]
        df[:,2]-=self.kappa*b*q[:,0]
        df[:,3]-=self.kappa*b*q[:,0]+self.rem_scale*q[:,1]
        return rp,f,dr,df

    def derivative_matrix(self,local_derivative):
        columns=np.r_[self.active,self.n+self.active,
                      np.arange(self.ps.start,self.ps.stop),np.arange(self.zs.start,self.zs.stop)]
        return sparse.csc_matrix((local_derivative.T.ravel(),
                (np.tile(np.arange(self.s),4),columns)),shape=(self.s,self.size))

    def residual_jacobian(self,u,previous,t0,t1,jacobian=True,oracle_rate=None):
        dt=t1-t0
        rp,f,dr,df=self.local_rates(self.local_state(u),jacobian,oracle_rate)
        # Exact dt key avoids silently reusing a different discrete step.
        if dt not in self._cache:
            self._cache[dt]=self.I+dt*self.G
        A=self._cache[dt]
        residual=A@u-previous+dt*(self.Hp@rp+self.Hf@f)
        residual[:self.n]-=dt*self.ops.inlet/self.ops.mass*pulse_average(t0,t1,self.cfg)/self.ref
        if not jacobian:
            return residual
        K=A+dt*(self.Hp@self.derivative_matrix(dr)+self.Hf@self.derivative_matrix(df))
        return residual,K.tocsc()

    def admissible(self,u):
        return (np.all(np.isfinite(u)) and np.min(u)>=-1e-10
                and np.max(u[self.ps]+u[self.zs])<=1+1e-10)

    def step(self,previous,t0,t1,retain_factor,oracle_rate=None):
        u=previous.copy()
        for iteration in range(50):
            residual,K=self.residual_jacobian(u,previous,t0,t1,True,oracle_rate)
            norm=float(np.linalg.norm(residual,np.inf))
            if not np.isfinite(norm):
                raise ArithmeticError(f"Nonfinite nonlinear residual at t={t1:g}")
            if norm<self.tolerance:
                if not self.admissible(u):
                    raise ArithmeticError(f"Surface simplex/concentration bounds failed at t={t1:g}")
                return u,splu(K) if retain_factor else None,iteration,norm
            try:
                increment=splu(K).solve(-residual)
            except RuntimeError as exc:
                raise ArithmeticError(f"Jacobian factorization failed at t={t1:g}") from exc
            alpha=1.
            for _ in range(32):
                candidate=u+alpha*increment
                if self.admissible(candidate):
                    trial=float(np.linalg.norm(self.residual_jacobian(
                        candidate,previous,t0,t1,False,oracle_rate),np.inf))
                    if trial<self.tolerance or trial<=(1-1e-4*alpha)*norm:
                        u=candidate
                        break
                alpha*=.5
            else:
                raise ArithmeticError(f"Bounded Newton line search failed at t={t1:g}, residual={norm:.3e}")
        raise ArithmeticError(f"Newton did not converge at t={t1:g}")

    def forward(self,times,retain_factors=False,oracle_hidden=None):
        times=np.asarray(times,dtype=float)
        if len(times)<2 or times[0]!=0 or np.any(np.diff(times)<=0):
            raise ValueError("A fixed increasing time grid beginning at zero is required")
        if self.mode=="oracle":
            if oracle_hidden is None or oracle_hidden.shape!=(len(times),self.s):
                raise ValueError("Oracle hidden-state array is missing or misaligned")
            if np.max(abs(oracle_hidden[0]))>1e-12:
                raise ValueError("This GT schema assumes an initially empty surface")
        elif oracle_hidden is not None:
            raise ValueError("GT hidden state is forbidden outside the explicit oracle mode")
        states=np.zeros((len(times),self.size))
        factors=[]; iterations=[]; residuals=[]
        for i in range(1,len(times)):
            oracle_rate=(oracle_hidden[i]-oracle_hidden[i-1])/(times[i]-times[i-1]) if self.mode=="oracle" else None
            u,factor,nit,res=self.step(states[i-1],times[i-1],times[i],retain_factors,oracle_rate)
            states[i]=u; factors.append(factor); iterations.append(nit); residuals.append(res)
        diagnostics=self.mass_diagnostics(times,states)
        diagnostics.update(max_newton_iterations=max(iterations),total_newton_iterations=sum(iterations),
                           maximum_residual=max(residuals),min_dt=float(np.diff(times).min()),
                           max_dt=float(np.diff(times).max()),clipping=False,solver_mode=self.mode)
        if max(diagnostics["precursor_relative"],diagnostics["byproduct_relative"],
               diagnostics["byproduct_relative_to_generated"])>2e-5:
            raise ArithmeticError(f"Mass balance failed: {diagnostics}")
        return Trajectory(times,states,factors,iterations,residuals,diagnostics)

    def mass_diagnostics(self,times,states):
        cp=states[:,:self.n]*self.ref; cb=states[:,self.n:2*self.n]*self.ref
        p=states[:,self.ps]; z=states[:,self.zs]
        dt=np.diff(times)
        incoming=np.r_[0,np.cumsum([self.ops.inlet.sum()*pulse_average(a,b,self.cfg)*(b-a)
                                   for a,b in zip(times[:-1],times[1:])])]
        outp=np.r_[0,np.cumsum(dt*(cp[1:]@self.ops.outlet))]
        outb=np.r_[0,np.cumsum(dt*(cb[1:]@self.ops.outlet))]
        cap=self.cfg.gamma*self.ops.surface[self.active]
        rp=cp@self.ops.mass+p@cap+outp-incoming
        rb=cb@self.ops.mass+z@cap+outb-self.cfg.byproduct_yield*(p@cap)
        scale=max(float(incoming[-1]),1e-30)
        generated=max(float(self.cfg.byproduct_yield*(p[-1]@cap)),1e-30)
        return dict(precursor_relative=float(np.max(abs(rp))/scale),
            byproduct_relative=float(np.max(abs(rb))/scale),mass_normalizer_mol=scale,
            byproduct_relative_to_generated=float(np.max(abs(rb))/generated),
            generated_byproduct_mol=generated,
            mass_normalizer="total injected precursor; 1 m out-of-plane depth",
            precursor_absolute_mol=float(np.max(abs(rp))),byproduct_absolute_mol=float(np.max(abs(rb))),
            precursor_min_mol_m3=float(cp.min()),precursor_max_mol_m3=float(cp.max()),
            byproduct_min_mol_m3=float(cb.min()),byproduct_max_mol_m3=float(cb.max()),
            theta_p_min=float(p.min()),theta_p_max=float(p.max()),
            theta_b_min=float(z.min()),theta_b_max=float(z.max()),
            maximum_total_occupancy=float((p+z).max()))

    def adjoint(self,tr,dLdu):
        if self.mode!="hidden":
            raise ValueError("Adjoint training is available only for predicted hidden dynamics")
        lam_next=np.zeros(self.size)
        xall=[]; qall=[]; worst=0.
        for i in range(len(tr.times)-1,0,-1):
            rhs=dLdu[i]+lam_next
            _,K=self.residual_jacobian(tr.states[i],tr.states[i-1],tr.times[i-1],tr.times[i])
            factor=tr.factors[i-1]
            lam=(splu(K) if factor is None else factor).solve(rhs,trans="T")
            worst=max(worst,float(np.linalg.norm(K.T@lam-rhs)/max(np.linalg.norm(rhs),1e-30)))
            x=self.local_state(tr.states[i]); free=1-x[:,2]-x[:,3]
            # No direct phi dependence in Rp. Its dependence on z is in K.
            cot=-(tr.times[i]-tr.times[i-1])*(self.Hf.T@lam)
            qbar=np.column_stack([cot*self.kappa*x[:,1]*free,-cot*self.rem_scale*x[:,3]])
            xall.append(x); qall.append(qbar); lam_next=lam
        self.adapter.vjp(np.concatenate(xall),np.concatenate(qall))
        return worst


@dataclass
class Prediction:
    observation: Observation
    theta_b_hat: np.ndarray
    diagnostics: dict


def to_prediction(obs,tr,solver):
    u=tr.states
    data=Observation(obs.name,obs.cfg,tr.times,u[:,:solver.n]*solver.ref,
        u[:,solver.n:2*solver.n]*solver.ref,u[:,solver.ps],solver.active,obs.role)
    return Prediction(data,u[:,solver.zs],tr.diagnostics)


class FieldLoss:
    """Only observable columns are supervised; dL/dtheta_b_hat is initially zero."""
    def __init__(self,obs,solver,weights=(1.,.2,.2),split="train"):
        self.target=np.concatenate([obs.cp/solver.ref,obs.cb/solver.ref,obs.theta],axis=1)
        n=solver.n
        self.slices=[solver.ps,slice(0,n),slice(n,2*n)]
        measure=[solver.ops.wafer[solver.active],solver.ops.mass,solver.ops.mass]
        mask=obs.training_mask() if split=="train" else ~obs.training_mask()
        if split=="all": mask[:]=True
        mask[0]=False
        dt=np.r_[0.,np.diff(obs.times)]
        self.q=[dt[:,None]*s[None,:]*mask[:,None] for s in measure]
        self.norm=[max(float(np.sum(w*self.target[:,sl]**2)),1e-30) for sl,w in zip(self.slices,self.q)]
        self.weights=weights

    def __call__(self,states,gradient=True):
        derivative=np.zeros_like(states) if gradient else None
        components=[]
        for sl,q,normalizer,weight in zip(self.slices,self.q,self.norm,self.weights):
            error=states[:,sl]-self.target[:,sl]
            components.append(float(np.sum(q*error**2)/normalizer))
            if gradient: derivative[:,sl]=2*weight*q*error/normalizer
        return float(np.dot(self.weights,components)),derivative,components


def regularization_anchors(observations,cref,count=512,seed=41):
    rng=np.random.default_rng(seed); chunks=[]
    for obs in observations:
        if obs.role!="train": raise ValueError("Held-out regularization is forbidden")
        m=obs.training_mask()
        x=np.stack([obs.cp[m][:,obs.active]/cref,obs.cb[m][:,obs.active]/cref,obs.theta[m]],axis=-1).reshape(-1,3)
        chunks.append(x)
    points=np.concatenate(chunks)
    points=points[rng.choice(len(points),min(len(points),count),replace=False)]
    # Independent samples of admissible z, not GT z or labels for its value.
    z=rng.uniform(size=len(points))*(1-points[:,2])
    return np.column_stack([points,z])


def smoothness_penalty(net,anchors,coefficient,backward):
    if coefficient==0: return 0.
    x=torch.tensor(anchors,dtype=torch.float64,requires_grad=True)
    q=net(x); penalty=0.
    for j in range(2):
        dq,=torch.autograd.grad(q[:,j].sum(),x,create_graph=backward,retain_graph=(backward or j==0))
        penalty=penalty+coefficient*torch.mean(torch.sum(dq*dq,dim=1))
    if backward: penalty.backward()
    return float(penalty.detach())


def objective(net,observations,ops,gradient=True,smoothness=0.,anchors=None,tolerance=2e-11):
    if gradient: net.zero_grad(set_to_none=True)
    total=0.; components=np.zeros(3); adjres=0.; diagnostics={}
    if not observations: raise ValueError("No training observations")
    for obs in observations:
        if obs.role!="train": raise ValueError("Held-out fields cannot enter the training objective")
        solver=TransientFEM(ops,obs.cfg,float(net.concentration_ref),RateAdapter(net),
                            float(net.time_ref),tolerance=tolerance)
        tr=solver.forward(obs.times,retain_factors=gradient)
        value,dLdu,terms=FieldLoss(obs,solver)(tr.states,gradient)
        total+=value/len(observations); components+=np.asarray(terms)/len(observations)
        if gradient: adjres=max(adjres,solver.adjoint(tr,dLdu/len(observations)))
        diagnostics[obs.name]=tr.diagnostics
    reg=smoothness_penalty(net,anchors,smoothness,gradient) if smoothness else 0.
    total+=reg
    if not np.isfinite(total): raise ArithmeticError("Nonfinite field loss")
    if gradient and any(p.grad is None or not torch.isfinite(p.grad).all() for p in net.parameters()):
        raise ArithmeticError("Missing/nonfinite parameter gradient")
    return dict(loss=total,theta_loss=float(components[0]),cp_loss=float(components[1]),
        cb_loss=float(components[2]),regularization_loss=reg,maximum_adjoint_residual=adjres,
        diagnostics=diagnostics)


def directional_gradcheck(net,observations,ops,seed=43,smoothness=0.,anchors=None):
    result=objective(net,observations,ops,True,smoothness,anchors,tolerance=2e-12)
    parameters=list(net.parameters())
    original=parameters_to_vector(parameters).detach().clone()
    gradient=torch.cat([p.grad.ravel() for p in parameters]).detach()
    ads_count=sum(p.numel() for p in net.ads.parameters())
    rng=np.random.default_rng(seed); rows=[]
    try:
        for name in ("all_parameters","adsorption_head","removal_head"):
            d=rng.normal(size=len(original))
            if name=="adsorption_head": d[ads_count:]=0
            if name=="removal_head": d[:ads_count]=0
            d/=np.linalg.norm(d)
            direction=torch.tensor(d,dtype=torch.float64)
            exact=float(gradient@direction)
            for epsilon in (1e-4,3e-5):
                with torch.no_grad(): vector_to_parameters(original+epsilon*direction,parameters)
                plus=objective(net,observations,ops,False,smoothness,anchors,tolerance=2e-12)["loss"]
                with torch.no_grad(): vector_to_parameters(original-epsilon*direction,parameters)
                minus=objective(net,observations,ops,False,smoothness,anchors,tolerance=2e-12)["loss"]
                fd=(plus-minus)/(2*epsilon)
                absolute=abs(fd-exact); relative=absolute/max(abs(fd),abs(exact),1e-12)
                rows.append(dict(direction=name,epsilon=epsilon,adjoint=exact,central_difference=fd,
                    relative_error=relative,absolute_error=absolute,
                    passed=bool(relative<3e-3 or absolute<5e-8),center_loss=result["loss"]))
    finally:
        with torch.no_grad(): vector_to_parameters(original,parameters)
        net.zero_grad(set_to_none=True)
    return rows


def state_hash(net):
    digest=hashlib.sha256()
    for name,value in net.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train(net,observations,ops,out,epochs,lr,smoothness,seed,patience=40):
    anchors=regularization_anchors(observations,float(net.concentration_ref),seed=seed)
    optimizer=torch.optim.Adam(net.parameters(),lr=lr)
    best=float("inf"); best_epoch=-1; best_state=None; last_significant=0
    feasible=deepcopy(net.state_dict()); opt_feasible=deepcopy(optimizer.state_dict())
    history=[]; failures=[]; start=time.monotonic(); consecutive=0
    for epoch in range(epochs+1):
        try:
            item=objective(net,observations,ops,True,smoothness,anchors)
        except (ArithmeticError,RuntimeError) as exc:
            failures.append(dict(epoch=epoch,error=str(exc),action="restore last feasible parameters and optimizer; halve LR"))
            write_json(out/"failed_steps.json",failures)
            net.load_state_dict(feasible); optimizer.load_state_dict(opt_feasible)
            consecutive+=1
            for group in optimizer.param_groups: group["lr"]*=.5**consecutive
            if consecutive>=4: raise RuntimeError("Repeated solver failure; no successful recovery is claimed") from exc
            continue
        consecutive=0
        feasible=deepcopy(net.state_dict()); opt_feasible=deepcopy(optimizer.state_dict())
        norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),2.))
        row={key:value for key,value in item.items() if key!="diagnostics"}
        row.update(epoch=epoch,gradient_norm=norm,learning_rate=optimizer.param_groups[0]["lr"],elapsed_s=time.monotonic()-start)
        history.append(row)
        if item["loss"]<best:
            if best==float("inf") or best-item["loss"]>1e-5*max(best,1e-8): last_significant=epoch
            best=item["loss"]; best_epoch=epoch; best_state=deepcopy(net.state_dict())
            # Checkpoint immediately, not only after a long run finishes.
            torch.save(dict(state_dict=best_state,specification=net.specification(),best_epoch=epoch,
                best_loss=best,training_conditions=[o.name for o in observations],
                hidden_gt_used=False,selection="training observable loss + smoothness",
                seed=seed,optimizer_state=optimizer.state_dict()),out/"best_model.pt")
            write_json(out/"best_solver_diagnostics.json",item["diagnostics"])
        write_rows(out/"training_history.csv",history)
        if epoch%10==0 or epoch==epochs:
            print(f"epoch={epoch:4d} L={item['loss']:.6g} theta={item['theta_loss']:.4g} "
                  f"cp={item['cp_loss']:.4g} cb={item['cb_loss']:.4g}",flush=True)
        if patience>0 and epoch-last_significant>=patience:
            print(f"Early stop: {patience} epochs without a relative 1e-5 objective improvement",flush=True)
            break
        if epoch<epochs: optimizer.step()
    if best_state is None: raise RuntimeError("No feasible training checkpoint")
    net.load_state_dict(best_state)
    write_json(out/"failed_steps.json",failures)
    return history,best_epoch,best


EXPECTED_TRAIN=((20.,.2),(50.,.4),(50.,.8),(75.,.8))
EXPECTED_HELD=((35.,.6),)
REPORTED_MEMORYLESS_G={(20.,.2):69.60,(50.,.4):183.22,(50.,.8):231.64,
                       (75.,.8):254.54,(35.,.6):187.31}


def condition_key(obs):
    return round(obs.cfg.precursor_pressure/TORR*1000,6),round(obs.cfg.pulse_duration,6)


def case_label(obs):
    pressure,dose=condition_key(obs)
    return f"{pressure:g} mTorr / {dose:g} s"


def validate_dataset(observations,ops):
    if len(observations)!=5 or any(o.role not in ("train","held_out") for o in observations):
        raise ValueError("This experiment requires exactly four train cases and one held-out case")
    train_obs=[o for o in observations if o.role=="train"]
    held=[o for o in observations if o.role=="held_out"]
    if sorted(condition_key(o) for o in train_obs)!=sorted(EXPECTED_TRAIN):
        raise ValueError("Need exactly four training conditions: 20/.2, 50/.4, 50/.8, 75/.8. "
                         "Generate GT with --preset multi; single-condition fallback is disabled.")
    if [condition_key(o) for o in held]!=list(EXPECTED_HELD):
        raise ValueError("Need exactly the held-out 35 mTorr / 0.6 s condition")
    n=len(ops.mass); active=np.flatnonzero(ops.surface>0)
    if ops.B.shape!=(n,n) or ops.Bb.shape!=(n,n) or not len(active) or np.any(ops.mass<=0):
        raise ValueError("Malformed FEM operators")
    if ops.wafer.sum()<=0 or ops.outlet.sum()<=0 or ops.inlet.sum()<=0:
        raise ValueError("Missing wafer, inlet, or outlet measure")
    for B in (ops.B,ops.Bb):
        mismatch=np.max(abs(np.asarray(B.sum(axis=0)).ravel()-ops.outlet))
        if mismatch>1e-8*max(ops.outlet.sum(),1e-30):
            raise ValueError("FEM column sums do not reproduce the exported outflow vector")
    # These settings affect the operator or hidden law and must remain fixed.
    fixed=["temperature","precursor_molar_mass","byproduct_molar_mass","site_area","beta0",
           "beta_byproduct","byproduct_yield","diffusivity","byproduct_diffusivity",
           "mean_velocity","length","height","wafer_start","wafer_end","nx","ny","inlet_condition"]
    reference=observations[0].cfg
    for obs in observations:
        if not np.array_equal(obs.active,active): raise ValueError(f"Surface DOF order mismatch: {obs.name}")
        if any(getattr(obs.cfg,k)!=getattr(reference,k) for k in fixed):
            raise ValueError("Conditions must share the same physics and FEM operator layout")
        nt=len(obs.times)
        if obs.cp.shape!=(nt,n) or obs.cb.shape!=(nt,n) or obs.theta.shape!=(nt,len(active)):
            raise ValueError(f"Invalid field shape in {obs.name}")
        if obs.times[0]!=0 or np.any(np.diff(obs.times)<=0): raise ValueError("Invalid time grid")
        if abs(obs.times[-1]-obs.cfg.end_time)>1e-9: raise ValueError("Incomplete GT trajectory")
        if not all(np.all(np.isfinite(a)) for a in (obs.cp,obs.cb,obs.theta)): raise ValueError("Nonfinite GT fields")
        if min(obs.cp.min()/obs.cfg.c0,obs.cb.min()/obs.cfg.c0,obs.theta.min()) < -1e-8 or obs.theta.max()>1+1e-8:
            raise ValueError("GT concentration/coverage bounds failed")
        if max(np.max(abs(obs.cp[0]))/obs.cfg.c0,np.max(abs(obs.cb[0]))/obs.cfg.c0,np.max(abs(obs.theta[0])))>1e-10:
            raise ValueError("GT and learner must start with an empty reactor and surface")
    return sorted(train_obs,key=condition_key),held


def dataset_fingerprint(root):
    # Hash public data only. Never read hidden files to fingerprint the training input.
    files=[root/"operators.npz"]+sorted(root.glob("*/fields.npz"))+sorted(root.glob("*/config.json"))
    digest=hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def load_hidden_for_diagnostics(root,obs,*,access):
    if access not in ("explicit_oracle_debug","after_frozen_training"):
        raise ValueError("Opening GT hidden state is forbidden in training")
    with np.load(root/obs.name/"hidden_surface_state.npz") as d:
        if not np.array_equal(d["times"],obs.times): raise ValueError("Hidden-state times differ from observations")
        z=d["theta_b"].copy()
    if z.shape!=obs.theta.shape or not np.all(np.isfinite(z)):
        raise ValueError("Hidden-state diagnostic array is invalid")
    if z.min() < -1e-8 or (z+obs.theta).max()>1+1e-8:
        raise ValueError("GT surface simplex check failed")
    return z


def relative_l2(pred,true,weight):
    numerator=float(np.sum(weight*(pred-true)**2))
    denominator=float(np.sum(weight*true**2))
    if denominator<=1e-30:
        return None  # A relative error is undefined for an identically zero target.
    return 100*math.sqrt(numerator/denominator)


def metrics(pred,gt,ops,split="all"):
    dt=np.r_[0.,np.diff(gt.times)]
    if split=="withheld_times": dt*=~gt.training_mask()
    wv=dt[:,None]*ops.mass[None,:]; ws=dt[:,None]*ops.wafer[gt.active][None,:]
    outlet=ops.outlet/ops.outlet.sum()
    return dict(theta_p_relative_L2_pct=relative_l2(pred.theta,gt.theta,ws),
        cp_relative_L2_pct=relative_l2(pred.cp,gt.cp,wv),cb_relative_L2_pct=relative_l2(pred.cb,gt.cb,wv),
        final_theta_p_relative_L2_pct=relative_l2(pred.theta[-1],gt.theta[-1],ops.wafer[gt.active]),
        outlet_cp_relative_L2_pct=relative_l2(pred.cp@outlet,gt.cp@outlet,dt),
        outlet_cb_relative_L2_pct=relative_l2(pred.cb@outlet,gt.cb@outlet,dt))


def error_reduction(ideal,learned):
    return {key:(1-learned[key]/ideal[key] if ideal[key] is not None and ideal[key]>1e-12
                 and learned[key] is not None else None) for key in ideal}


def predict(net,obs,ops,ideal=False):
    solver=TransientFEM(ops,obs.cfg,float(net.concentration_ref),None if ideal else RateAdapter(net),
        float(net.time_ref),mode="ideal" if ideal else "hidden")
    return to_prediction(obs,solver.forward(obs.times),solver)


def save_prediction(path,pred,ops):
    obs=pred.observation
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,times=obs.times,cp_mol_m3=obs.cp,cb_mol_m3=obs.cb,
        theta_p=obs.theta,theta_b_hat=pred.theta_b_hat,surface_dofs=obs.active,
        coordinates=ops.coordinates,wafer_weight=ops.wafer)
    write_json(path.with_suffix(".json"),pred.diagnostics)


def posthoc_hidden(pred,gt,ztrue,ops,cref):
    dt=np.r_[0.,np.diff(gt.times)]
    weight=dt[:,None]*ops.wafer[gt.active][None,:]
    hidden_error=relative_l2(pred.theta_b_hat,ztrue,weight)
    denominator=1-gt.theta
    # Common observation-side denominator. Negative reconstructed free fractions
    # are retained here as a diagnostic, never clipped to look accurate.
    gtrue=np.divide(1-gt.theta-ztrue,denominator,out=np.full_like(denominator,np.nan),where=denominator>1e-4)
    greconstructed=np.divide(1-gt.theta-pred.theta_b_hat,denominator,
                              out=np.full_like(denominator,np.nan),where=denominator>1e-4)
    # Legacy code used its first training condition's c0; that reference is not
    # known from the supplied error list. A fixed comparison mask is declared.
    mask=(gt.cp[:,gt.active]/cref>1e-4)&(denominator>1e-4)&(weight>0)
    gerror=relative_l2(greconstructed[mask],gtrue[mask],weight[mask]) if np.any(mask) else None
    return dict(theta_b_hidden_relative_L2_pct=hidden_error,
        theta_b_absolute_weighted_RMS=float(np.sqrt(np.sum(weight*(pred.theta_b_hat-ztrue)**2)/weight.sum())),
        final_theta_b_relative_L2_pct=relative_l2(pred.theta_b_hat[-1],ztrue[-1],ops.wafer[gt.active]),
        reconstructed_g_relative_L2_pct=gerror,
        g_definition="(1 - theta_p_GT - theta_b_hat)/ (1 - theta_p_GT), post-hoc only; unclipped",
        g_comparison_note="Historical memoryless diagnostic evaluated its network on GT local states; new diagnostic uses the learned open-loop hidden trajectory. Mask references can differ; no strict same-protocol improvement claim.",
        g_mask="cp/c_ref>1e-4, 1-theta_p_GT>1e-4; positive dt*wafer weight",
        g_identifiable_sample_count=int(mask.sum()))


def oracle_test(observations,ops,gt_root,out,cref):
    """Explicit debug only: prescribe z_GT and its discrete increment consistently.

    Using z_GT only in Rp while omitting Gamma*Delta(z_GT)/dt from gas cb is
    NOT an oracle of the competitive model. Both pathways are included here.
    No NN is constructed, trained, or checkpointed in this mode.
    """
    folder=out/"oracle_debug"; folder.mkdir(parents=True,exist_ok=True)
    results={}; passed=True
    for obs in observations:
        z=load_hidden_for_diagnostics(gt_root,obs,access="explicit_oracle_debug")
        solver=TransientFEM(ops,obs.cfg,cref,mode="oracle",tolerance=2e-12)
        tr=solver.forward(obs.times,oracle_hidden=z)
        pred=to_prediction(obs,tr,solver)
        m=metrics(pred.observation,obs,ops)
        absolute=max(float(np.max(abs(pred.observation.cp-obs.cp))/obs.cfg.c0),
                     float(np.max(abs(pred.observation.cb-obs.cb))/obs.cfg.c0),
                     float(np.max(abs(pred.observation.theta-obs.theta))))
        zerror=float(np.max(abs(pred.theta_b_hat-z)))
        case_pass=absolute<2e-5 and zerror<1e-8 and max(tr.diagnostics["precursor_relative"],tr.diagnostics["byproduct_relative"])<2e-5
        results[obs.name]=dict(passed=bool(case_pass),field_errors=m,
            maximum_normalized_field_difference=absolute,maximum_hidden_difference=zerror,
            mass_balance=tr.diagnostics)
        save_prediction(folder/f"{obs.name}_oracle.npz",pred,ops)
        passed=passed and case_pass
    report=dict(mode="ORACLE DEBUG ONLY; not an inverse-learning result",passed=bool(passed),
        hidden_GT_supplied=True,network_constructed=False,dataset_sha256=dataset_fingerprint(gt_root),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=results,
        tolerance="max cp/c0, cb/c0, theta difference < 2e-5; max hidden difference < 1e-8; mass error < 2e-5")
    write_json(folder/"report.json",report)
    print(json.dumps(report,indent=2),flush=True)
    if not passed: raise ArithmeticError("Oracle failed: fix coupling/data/units before ML training")
    return report


def clean_training_output(out,gt_root):
    """Clear only this learner's known outputs, preserving oracle/verification records."""
    if out.is_symlink() or out==gt_root or out in gt_root.parents or gt_root in out.parents:
        raise ValueError("Training output and input GT must be separate, nonsymlink trees")
    if out.name in ("09_missing_physics","08_competitive_feml","08_gt_competitive"):
        raise ValueError("Refusing to overwrite the GT or historical memoryless output")
    if out.exists():
        manifest=out/"manifest.json"
        recognized=manifest.is_file() and json.loads(manifest.read_text()).get("learner_id")=="explicit_hidden_state_feml"
        harmless={"verification","oracle_debug"}
        if not recognized and any(p.name not in harmless for p in out.iterdir()):
            raise ValueError("Existing output is not owned by this hidden-state learner; choose its own output directory")
    out.mkdir(parents=True,exist_ok=True)
    for name in ("predictions","figures","best_model.pt","manifest.json","metrics.json","training_history.csv",
                 "SUMMARY.md","failed_steps.json","RUN_FAILED.json","gradient_check.csv","best_solver_diagnostics.json"):
        path=out/name
        if path.is_symlink(): raise ValueError(f"Refusing to clear symlink {path}")
        if path.is_dir(): shutil.rmtree(path)
        elif path.exists(): path.unlink()
    (out/"predictions").mkdir(); (out/"figures").mkdir()


def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt",default="outputs/08_gt_competitive")
    p.add_argument("--output",default="outputs/09_hidden_state_feml")
    p.add_argument("--epochs",type=int,default=200)
    p.add_argument("--learning-rate",type=float,default=.003)
    p.add_argument("--smoothness",type=float,default=1e-6)
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--patience",type=int,default=40,help="0 disables early stopping; otherwise train-objective only")
    p.add_argument("--memoryless-metrics",default="outputs/09_missing_physics/metrics.json")
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--oracle-theta-b",action="store_true",help="Separate debug; supplies GT z and its gas sink, never trains")
    mode.add_argument("--self-test",action="store_true",help="Manufactured algebra verification; works without Torch or GT files")
    p.add_argument("--use-all-train",action="store_true",help="Compatibility flag; four training conditions are always required")
    return p.parse_args()


def load_dataset(root):
    if not (root/"operators.npz").is_file():
        raise FileNotFoundError(f"Missing {root/'operators.npz'}. Run gt_competitive_standalone.py --preset multi first, or pass --gt to existing exported GT.")
    folders=sorted(p.parent for p in root.glob("*/config.json"))
    observations=[load_case(p) for p in folders]
    ops=load_operators(root/"operators.npz")
    train_obs,held=validate_dataset(observations,ops)
    for obs in observations:
        if not (root/obs.name/"hidden_surface_state.npz").is_file():
            raise FileNotFoundError(f"Missing post-hoc hidden-state file for {obs.name}; its contents are not needed for training")
    return ops,train_obs,held


def training_summary(out,observations,results,legacy,manifest,oracle_report):
    lines=["# Explicit Hidden-State ALD FEML","",
        f"Status: {manifest['status']}; best epoch: {manifest['best_epoch']}; best train objective: {manifest['best_loss']:.6g}.",
        "","Training observations: cp, cb, theta_p only. GT theta_b was opened only after all updates ended and weights were frozen.",
        "Held-out fields never affected gradients, normalization, checkpoint selection, early stopping, or architecture.","",
        "Known physics: exported DOLFINx transport, precursor sticking velocity kp, molar capacity Gamma, site balance and stoichiometry.",
        "Hidden state: predicted adsorbed-byproduct site fraction z = theta_b_hat, initially zero.",
        "NN inputs: (cp/c_ref, cb/c_ref, theta_p, predicted z); outputs: positive q_ads, q_rem.","",
        "Equations:","",
        "```text",
        "e = 1 - theta_p - z",
        "Rp = kp * cp * e",
        "f  = (kp * cb / Gamma) * e * softplus(N_ads) - (z/t_ref) * softplus(N_rem)",
        "Gamma * d(theta_p)/dt = Rp",
        "dz/dt = f",
        "Rb = Gamma * f  (positive: adsorption sink; negative: release to gas)",
        "gas precursor boundary residual: +Rp",
        "gas byproduct boundary residual: -n_bp*Rp + Rb",
        "```","",
        "The cb availability factor prevents adsorption from an empty gas phase. Removal is explicitly assumed to return one gas byproduct molecule per freed site, consistent with this model's molar capacity.","",
        "| Case | Role | Ideal theta_p % | New theta_p % | Ideal cp % | New cp % | Ideal cb % | New cb % | Hidden theta_b % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for obs in observations:
        r=results[obs.name]; a=r["ideal"]; b=r["hidden_state_model"]
        lines.append(f"| {case_label(obs)} | {obs.role} | {fmt(a['theta_p_relative_L2_pct'])} | {fmt(b['theta_p_relative_L2_pct'])} | {fmt(a['cp_relative_L2_pct'])} | {fmt(b['cp_relative_L2_pct'])} | {fmt(a['cb_relative_L2_pct'])} | {fmt(b['cb_relative_L2_pct'])} | {fmt(r['posthoc']['theta_b_hidden_relative_L2_pct'])} |")
    lines += ["","All errors above are relative L2 percentages using BE time weights and FEM volume/wafer quadrature.","",
        "| Case | Precursor mass residual | Byproduct mass residual | Old g L2 % (reported) | New z-based g diagnostic % |",
        "|---|---:|---:|---:|---:|"]
    for obs in observations:
        r=results[obs.name]; d=r["mass_balance"]["hidden_state_model"]
        lines.append(f"| {case_label(obs)} | {d['precursor_relative']:.3e} | {d['byproduct_relative']:.3e} | {fmt(legacy[obs.name]['reported_closure_relative_L2_pct'])} | {fmt(r['posthoc']['reconstructed_g_relative_L2_pct'])} |")
    lines += ["","Mass residuals are fractions of total injected precursor; they are not percentages.",
        "Historical memoryless errors are user-reported, not rerun results. New g diagnostics use predicted z and GT theta_p after training. Old normalization/mask may differ; this is not a strictly identical-protocol error-reduction estimate.","",
        "Previous failure has at least two plausible structural causes: missing history state and an omitted byproduct adsorption sink. High g errors alone do not prove mathematical non-single-valuedness. The old code also defaulted to the first training case unless --use-all-train was supplied; the previous run's actual flag is not available here.","",
        f"Actual exported-GT oracle status: {('passed' if oracle_report.get('passed') else oracle_report.get('status','not run'))}.",
        f"Torch/FEM directional check: passed; max relative error {manifest['gradient_check_max_relative']:.3e}.",
        "The discrete adjoint includes lambda_(n+1) and sensitivities of both the byproduct gas sink and hidden-state ODE.","",
        "Technical assessment: do not advance directly to a claimed 2D-to-3D zero-shot result. First inspect the actual errors, oracle, mass balance, time-step/mesh sensitivity, and support of hidden-state dynamics. This run does not establish mesh independence or uniquely identify the adsorption and removal laws separately.",
        "If all observable errors improve, this is evidence of field reconstruction. Hidden-state error must be judged separately; lower training loss alone is not hidden-physics recovery.",
        "With this exact competitive GT and known site balance, one byproduct state is structurally sufficient. A failure of the new learner should first trigger checks of implementation, optimization, observation support, and stiffness; it is not by itself evidence that another latent state is needed.","",
        "The prior 24 Å² site-area assumption is retained. This remains synthetic recovery, not experimental validation.","",
        "Foreground:","```bash","python train_missing_physics_standalone.py --epochs 200","```",
        "Background:","```bash","nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\",
        "python train_missing_physics_standalone.py --epochs 200 \\","> train_hidden_state.log 2>&1 &","```","",
        "Main figure: figures/00_main_result.png. All metrics: metrics.json. Raw predictions: predictions/."]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n")


def run_training(args,ops,training,held,gt_root,out):
    require_torch()
    torch.set_num_threads(1); torch.manual_seed(args.seed); np.random.seed(args.seed)
    # Fixed physical maximum inlet concentration of the four training cases.
    cref=max(o.cfg.c0 for o in training)
    net=HiddenRateNet(cref)
    fingerprint=dataset_fingerprint(gt_root)
    oracle_path=out/"oracle_debug"/"report.json"
    oracle_report={"status":"not run on this dataset/source"}
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if oracle_path.is_file():
        saved=json.loads(oracle_path.read_text())
        if saved.get("dataset_sha256")==fingerprint and saved.get("source_sha256")==source_sha:
            oracle_report=saved
            if not saved.get("passed"): raise RuntimeError("The explicit oracle failed; training is blocked")
    clean_training_output(out,gt_root)
    manifest=dict(learner_id="explicit_hidden_state_feml",status="running",
        started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source_sha,
        dataset_sha256=fingerprint,arguments=vars(args),seed=args.seed,
        network=net.specification(),normalization_source="maximum inlet concentration across four training cases only",
        training_cases=[dict(name=o.name,config=asdict(o.cfg)) for o in training],
        heldout_cases=[dict(name=o.name,config=asdict(o.cfg)) for o in held],
        GT_hidden_state_used_in_training=False,GT_byproduct_rate_parameter_used_by_learner=False,
        hidden_file_access_policy="separate explicit oracle flag OR after frozen training",
        observation_loss_weights={"theta_p":1.,"cp":.2,"cb":.2},
        time_split="every fifth noninitial training-case snapshot excluded from loss",
        smoothness_anchors="training cp/c_ref, cb/c_ref, theta_p plus independently sampled admissible z; no GT z",
        optimizer="Adam; gradient norm capped at 2; feasible-state/LR backtracking",
        checkpoint_selection="training observable-field objective plus smoothness only",
        early_stopping="patience epochs without relative 1e-5 training objective improvement; no held-out selection",
        solver="exported DOLFINx lumped P1 FEM operators; bounded Newton; SciPy SuperLU",
        integrator="backward Euler on exactly the exported GT time grid; no clipping or adaptive grid during optimization",
        units={"cp_cb":"mol/m3","Gamma":"mol/m2","Rp_Rb":"mol/m2/s","f":"1/s","theta_p_z":"dimensionless site fraction"},
        byproduct_coupling="gas residual includes Gamma*f, same f as dz/dt; removal releases gas",
        oracle_status="passed" if oracle_report.get("passed") else "not run",
        versions={"python":sys.version.split()[0],"numpy":np.__version__,"torch":str(torch.__version__)},
        known_limits=["no mesh/time refinement executed by training entry point","adsorption/removal laws need not be separately identifiable","no 3D test"])
    write_json(out/"manifest.json",manifest)
    try:
        anchors=regularization_anchors(training,cref,seed=args.seed)
        print("Checking full transient gradient for both rate heads...",flush=True)
        checks=directional_gradcheck(net,training,ops,seed=args.seed+2,smoothness=args.smoothness,anchors=anchors)
        write_rows(out/"gradient_check.csv",checks)
        if not all(r["passed"] for r in checks):
            raise ArithmeticError("Directional gradient check failed; optimization is blocked")
        manifest["gradient_check_max_relative"]=max(r["relative_error"] for r in checks)
        print(f"Gradient gate passed: max relative error {manifest['gradient_check_max_relative']:.3e}",flush=True)
        history,best_epoch,best_loss=train(net,training,ops,out,args.epochs,args.learning_rate,args.smoothness,args.seed,args.patience)
        net.eval()
        for parameter in net.parameters(): parameter.requires_grad_(False)
        frozen=state_hash(net)
        observations=training+held
        ideal={}; learned={}; hidden={}; results={}
        # The first GT hidden-state content read in the normal training process
        # is BELOW this point, after all optimizer updates are complete.
        for obs in observations:
            ideal[obs.name]=predict(net,obs,ops,ideal=True)
            learned[obs.name]=predict(net,obs,ops)
            hidden[obs.name]=load_hidden_for_diagnostics(gt_root,obs,access="after_frozen_training")
            save_prediction(out/"predictions"/f"{obs.name}_ideal.npz",ideal[obs.name],ops)
            save_prediction(out/"predictions"/f"{obs.name}_hidden_state.npz",learned[obs.name],ops)
            a=metrics(ideal[obs.name].observation,obs,ops)
            b=metrics(learned[obs.name].observation,obs,ops)
            results[obs.name]=dict(role=obs.role,ideal=a,hidden_state_model=b,
                improvement_fraction=error_reduction(a,b),
                withheld_time_errors=metrics(learned[obs.name].observation,obs,ops,"withheld_times"),
                posthoc=posthoc_hidden(learned[obs.name],obs,hidden[obs.name],ops,cref),
                mass_balance={"ideal":ideal[obs.name].diagnostics,"hidden_state_model":learned[obs.name].diagnostics})
        if state_hash(net)!=frozen: raise RuntimeError("Evaluation changed frozen weights or normalization")
        legacy=legacy_results(Path(args.memoryless_metrics),observations)
        for obs in observations: results[obs.name]["historical_memoryless"]=legacy[obs.name]
        write_json(out/"metrics.json",results)
        make_plots(out/"figures",observations,ideal,learned,hidden,results,legacy,history,ops)
        manifest.update(status="completed_not_automatically_a_recovery_success",best_epoch=best_epoch,best_loss=best_loss,
            completed_utc=datetime.now(timezone.utc).isoformat(),frozen_state_sha256=frozen,
            all_primary_fields_improved=all(
                results[o.name]["improvement_fraction"][k] is not None and results[o.name]["improvement_fraction"][k]>0
                for o in observations for k in ("theta_p_relative_L2_pct","cp_relative_L2_pct","cb_relative_L2_pct")))
        write_json(out/"manifest.json",manifest)
        training_summary(out,observations,results,legacy,manifest,oracle_report)
        print(f"\nCompleted -> {out/'SUMMARY.md'}",flush=True)
        print("nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python train_missing_physics_standalone.py --epochs 200 > train_hidden_state.log 2>&1 &",flush=True)
    except Exception as exc:
        manifest.update(status="failed",error=str(exc))
        write_json(out/"manifest.json",manifest)
        write_json(out/"RUN_FAILED.json",dict(error=str(exc),traceback=traceback.format_exc()))
        (out/"SUMMARY.md").write_text("# Run failed\n\nNo recovery success is claimed.\n\n"+str(exc)+"\n\nSee RUN_FAILED.json.\n")
        raise


def main():
    args=parse_args(); gt_root=Path(args.gt).resolve(); out=Path(args.output).resolve()
    if args.epochs<1 or args.learning_rate<=0 or args.smoothness<0 or args.patience<0:
        raise ValueError("Positive epochs/LR and nonnegative smoothness/patience required")
    if args.self_test:
        self_test(out)
        return
    # These checks happen before any output cleanup.
    ops,training,held=load_dataset(gt_root)
    if args.oracle_theta_b:
        oracle_test(training+held,ops,gt_root,out,max(o.cfg.c0 for o in training))
        return
    run_training(args,ops,training,held,gt_root,out)


def legacy_results(path,observations):
    data=json.loads(path.read_text()) if path and path.is_file() else {}
    result={}
    for obs in observations:
        item=data.get(obs.name,{})
        result[obs.name]=dict(
            reported_closure_relative_L2_pct=REPORTED_MEMORYLESS_G.get(condition_key(obs)),
            provenance="User-provided failure summary; not recomputed in this run",
            fields=item.get("learned"),raw_saved_closure_relative_L2_pct=item.get("closure_posthoc_relative_L2_pct"),
            checkpoint_available=False)
    return result


def fmt(value):
    return "N/A" if value is None else f"{value:.4g}"


def make_plots(folder,observations,ideal,learned,hidden,results,legacy,history,ops):
    folder.mkdir(parents=True,exist_ok=True)
    colors={"GT":"#132e51","Ideal":"#ca9045","Hidden-state ML":"#5d59b7"}
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    def finish(fig,name):
        fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(folder/name,dpi=180,bbox_inches="tight"); plt.close(fig)
    def panels(cases,title):
        cols=2 if len(cases)>1 else 1; rows=(len(cases)+cols-1)//cols
        fig,axes=plt.subplots(rows,cols,figsize=(6*cols,3.7*rows),squeeze=False)
        for ax in axes.ravel()[len(cases):]: ax.axis("off")
        fig.suptitle(title,fontsize=13)
        return fig,axes.ravel()
    training=[o for o in observations if o.role=="train"]
    held=[o for o in observations if o.role=="held_out"]
    def final_profile(ax,obs):
        x=ops.coordinates[obs.active,0]; ix=np.argsort(x)
        for label,item in [("GT",obs),("Ideal",ideal[obs.name].observation),("Hidden-state ML",learned[obs.name].observation)]:
            ax.plot(x[ix],item.theta[-1,ix],label=label,color=colors[label],lw=2)
        ax.set(title=f"{case_label(obs)} [{obs.role}]",xlabel="Wafer x [m]",ylabel="Final precursor coverage",ylim=(-.02,1.02)); ax.legend(fontsize=9)
    fig,ax=plt.subplots(figsize=(7,4))
    for key,label in [("loss","Total"),("theta_loss","theta_p"),("cp_loss","cp"),("cb_loss","cb")]:
        ax.semilogy([r["epoch"] for r in history],[max(r[key],1e-16) for r in history],label=label)
    ax.set(xlabel="Epoch",ylabel="Normalized squared field loss",title="Observable-field inverse training (no theta_b labels)"); ax.legend()
    finish(fig,"01_training_loss.png")
    for cases,filename in [(training,"02_theta_p_final_train.png"),(held,"03_theta_p_final_heldout.png")]:
        fig,axes=panels(cases,"Final precursor coverage")
        for ax,obs in zip(axes,cases): final_profile(ax,obs)
        finish(fig,filename)
    for key,filename,ylabel in [("theta","04_theta_p_transient.png","Mean precursor coverage"),
            ("cp","05_precursor_outlet.png","Precursor outlet [mol/m³]"),
            ("cb","06_byproduct_outlet.png","Byproduct outlet [mol/m³]")]:
        fig,axes=panels(observations,ylabel)
        for ax,obs in zip(axes,observations):
            for label,item in [("GT",obs),("Ideal",ideal[obs.name].observation),("Hidden-state ML",learned[obs.name].observation)]:
                weight=ops.wafer[obs.active] if key=="theta" else ops.outlet
                values=getattr(item,key)@weight/weight.sum()
                ax.plot(obs.times,values,label=label,color=colors[label])
            ax.axvline(obs.cfg.pulse_duration+obs.cfg.rise_time,color=".7",ls=":")
            ax.set(title=f"{case_label(obs)} [{obs.role}]",xlabel="Time [s]",ylabel=ylabel); ax.legend(fontsize=8)
        finish(fig,filename)
    for final,filename in [(True,"07_hidden_theta_b_final.png"),(False,"08_hidden_theta_b_transient.png")]:
        fig,axes=panels(observations,"Hidden byproduct coverage — GT theta_b was not a training target")
        for ax,obs in zip(axes,observations):
            z0=hidden[obs.name]; zh=learned[obs.name].theta_b_hat
            if final:
                x=ops.coordinates[obs.active,0]; ix=np.argsort(x); x=x[ix]
                y0=z0[-1,ix]; yh=zh[-1,ix]; xlabel="Wafer x [m]"
            else:
                w=ops.wafer[obs.active]; x=obs.times; y0=z0@w/w.sum(); yh=zh@w/w.sum(); xlabel="Time [s]"
            ax.plot(x,y0,label="Hidden GT (post-hoc)",color=colors["GT"])
            ax.plot(x,yh,label="Predicted theta_b_hat",color=colors["Hidden-state ML"])
            ax.set(title=f"{case_label(obs)} [{obs.role}]",xlabel=xlabel,ylabel="Byproduct coverage",ylim=(-.01,1.01)); ax.legend(fontsize=8)
        finish(fig,filename)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,cases,title in zip(axes,[training,held],["Training conditions","Frozen held-out condition"]):
        maxvalue=.05
        for obs in cases:
            zg=hidden[obs.name][1:].ravel(); zp=learned[obs.name].theta_b_hat[1:].ravel(); stride=max(1,len(zg)//2500)
            ax.scatter(zg[::stride],zp[::stride],s=6,alpha=.35,label=case_label(obs))
            maxvalue=max(maxvalue,float(zg.max()),float(zp.max()))
        limit=min(1.,maxvalue*1.05)
        ax.plot([0,limit],[0,limit],"k--",lw=1); ax.set(xlabel="GT theta_b (revealed after training)",ylabel="Predicted theta_b_hat",title=title,xlim=(0,limit),ylim=(0,limit)); ax.legend(fontsize=8)
    fig.suptitle("Hidden-state parity — no GT theta_b input or loss")
    finish(fig,"09_theta_b_parity.png")
    fig,axes=panels(observations,"Observable field errors and hidden-state recovery")
    keys=["theta_p_relative_L2_pct","cp_relative_L2_pct","cb_relative_L2_pct"]
    for ax,obs in zip(axes,observations):
        result=results[obs.name]; x=np.arange(3)
        ax.bar(x-.18,[result["ideal"][k] for k in keys],.36,label="Ideal",color=colors["Ideal"])
        ax.bar(x+.18,[result["hidden_state_model"][k] for k in keys],.36,label="Hidden-state ML",color=colors["Hidden-state ML"])
        ax.set(xticks=x,xticklabels=["theta_p","cp","cb"],ylabel="Relative L2 [%]",title=case_label(obs)+f" | hidden z: {fmt(result['posthoc']['theta_b_hidden_relative_L2_pct'])}%"); ax.legend(fontsize=8)
    finish(fig,"10_error_comparison.png")
    obs=held[0]; fig,axes=plt.subplots(2,2,figsize=(13,9))
    final_profile(axes[0,0],obs); axes[0,0].set_title("A. Frozen held-out precursor coverage")
    x=ops.coordinates[obs.active,0]; ix=np.argsort(x)
    axes[0,1].plot(x[ix],hidden[obs.name][-1,ix],label="Hidden GT",color=colors["GT"],lw=2)
    axes[0,1].plot(x[ix],learned[obs.name].theta_b_hat[-1,ix],label="Predicted hidden state",color=colors["Hidden-state ML"],lw=2)
    axes[0,1].set(title="B. Held-out hidden-state reconstruction",xlabel="Wafer x [m]",ylabel="Final theta_b"); axes[0,1].legend()
    labels=[case_label(o) for o in observations]; pos=np.arange(len(observations))
    axes[1,0].bar(pos-.18,[results[o.name]["hidden_state_model"]["theta_p_relative_L2_pct"] for o in observations],.36,label="theta_p field")
    axes[1,0].bar(pos+.18,[results[o.name]["posthoc"]["theta_b_hidden_relative_L2_pct"] for o in observations],.36,label="Hidden theta_b")
    axes[1,0].set(xticks=pos,xticklabels=labels,ylabel="Relative L2 [%]",title="C. Field and hidden-state errors (last: held out)"); axes[1,0].tick_params(axis="x",rotation=25); axes[1,0].legend(fontsize=9)
    axes[1,1].bar(pos-.18,[legacy[o.name]["reported_closure_relative_L2_pct"] for o in observations],.36,label="Old g error (reported)",color=colors["Ideal"])
    axes[1,1].bar(pos+.18,[results[o.name]["posthoc"]["reconstructed_g_relative_L2_pct"] for o in observations],.36,label="New z-based g diagnostic",color=colors["Hidden-state ML"])
    axes[1,1].set(xticks=pos,xticklabels=labels,ylabel="g diagnostic relative L2 [%]",title="D. Historical comparison; diagnostic protocols differ")
    axes[1,1].tick_params(axis="x",rotation=25); axes[1,1].legend(fontsize=8)
    fig.suptitle("Explicit hidden-state FEML | training never used GT theta_b",fontsize=14)
    finish(fig,"00_main_result.png")


def self_test(out):
    """Manufactured algebra checks inside this file; never reported as ALD recovery."""
    class AnalyticFixture:
        def __init__(self,p):
            self.p=np.asarray(p,dtype=float).copy(); self.grad=np.zeros_like(self.p)
        def value(self,x):
            a=np.column_stack([x,np.ones(len(x))])@self.p.T
            return np.logaddexp(0,a)
        def value_tangent(self,x):
            a=np.column_stack([x,np.ones(len(x))])@self.p.T
            sigmoid=np.exp(-np.logaddexp(0,-a))
            return np.logaddexp(0,a),sigmoid[:,:,None]*self.p[None,:,:4]
        def vjp(self,x,cot):
            features=np.column_stack([x,np.ones(len(x))])
            raw=features@self.p.T
            self.grad+=(cot*np.exp(-np.logaddexp(0,-raw))).T@features
    B=sparse.csr_matrix([[.012,-.002,0],[-.012,.014,-.002],[0,-.012,.012]])
    D=sparse.csr_matrix([[.006,-.006,0],[-.006,.012,-.006],[0,-.006,.006]])
    w=np.array([.05,0.,.05])
    ops=Operators(B,B+D,np.array([.002,.004,.002]),w,w.copy(),np.array([.01,0,0]),
        np.array([0,0,.01]),np.array([[0,0,0],[.25,.02,0],[.5,0,0]]))
    cfg=replace(stage5_config(),beta0=2e-4,pulse_duration=.13,rise_time=.02,purge_duration=.17,dt=.017)
    events=[0.,cfg.rise_time,cfg.pulse_duration,cfg.pulse_duration+cfg.rise_time,cfg.end_time]
    times=np.r_[0.,np.concatenate([np.linspace(a,b,int(np.ceil((b-a)/cfg.dt))+1)[1:] for a,b in zip(events[:-1],events[1:])])]
    teacher_adapter=AnalyticFixture([[.2,-.3,.4,-.2,-.5],[-.1,.2,-.1,.3,-1.5]])
    adapter=AnalyticFixture([[-.4,.5,-.1,.2,.2],[.2,-.1,.3,-.2,-.5]])
    teacher=TransientFEM(ops,cfg,cfg.c0,teacher_adapter,tolerance=1e-12)
    truth=teacher.forward(times)
    solver=TransientFEM(ops,cfg,cfg.c0,adapter,tolerance=1e-12)
    checks=[]
    def check(name,condition,**details):
        checks.append(dict(name=name,passed=bool(condition),**details))
        if not condition: raise AssertionError(f"{name}: {details}")
    u=np.array([.3,.4,.5,.07,.04,.02,.2,.25,.1,.15])
    direction=np.random.default_rng(11).normal(size=len(u)); eps=1e-6
    _,K=solver.residual_jacobian(u,.97*u,.03,.048)
    fd=(solver.residual_jacobian(u+eps*direction,.97*u,.03,.048,False)
        -solver.residual_jacobian(u-eps*direction,.97*u,.03,.048,False))/(2*eps)
    err=float(np.linalg.norm(K@direction-fd)/np.linalg.norm(fd))
    check("coupled_state_Jacobian_all_four_inputs",err<1e-7,relative_error=err)
    obs=Observation("manufactured_not_ALD_GT",cfg,times,truth.states[:,:3]*cfg.c0,
                    truth.states[:,3:6]*cfg.c0,truth.states[:,teacher.ps],teacher.active,"train")
    tr=solver.forward(times,True)
    loss=FieldLoss(obs,solver)
    value,dLdu,_=loss(tr.states)
    check("no_direct_hidden_state_loss",np.count_nonzero(dLdu[:,solver.zs])==0)
    adjres=solver.adjoint(tr,dLdu); analytic=adapter.grad.copy()
    finite=np.zeros_like(analytic); eps=1e-5
    for i,j in np.ndindex(adapter.p.shape):
        base=adapter.p[i,j]
        adapter.p[i,j]=base+eps; plus=loss(solver.forward(times).states,False)[0]
        adapter.p[i,j]=base-eps; minus=loss(solver.forward(times).states,False)[0]
        adapter.p[i,j]=base; finite[i,j]=(plus-minus)/(2*eps)
    rel=float(np.linalg.norm(analytic-finite)/max(np.linalg.norm(finite),1e-30))
    check("transient_parameter_adjoint_both_rate_heads",rel<2e-6,
          relative_L2_error=rel,maximum_absolute_error=float(np.max(abs(analytic-finite))),linear_residual=adjres)
    # Late observation forces sensitivity propagation through every earlier step.
    adapter.grad[:]=0
    derivative=np.zeros_like(tr.states); derivative[-1,solver.ps]=2*(tr.states[-1,solver.ps]-truth.states[-1,teacher.ps])
    solver.adjoint(tr,derivative)
    pdirection=np.random.default_rng(12).normal(size=adapter.p.shape); pdirection/=np.linalg.norm(pdirection)
    exact=float(np.sum(adapter.grad*pdirection)); original=adapter.p.copy()
    adapter.p=original+eps*pdirection
    plus=float(np.sum((solver.forward(times).states[-1,solver.ps]-truth.states[-1,teacher.ps])**2))
    adapter.p=original-eps*pdirection
    minus=float(np.sum((solver.forward(times).states[-1,solver.ps]-truth.states[-1,teacher.ps])**2))
    adapter.p=original
    late_fd=(plus-minus)/(2*eps)
    check("terminal_observation_time_dependencies",abs(exact-late_fd)<1e-9,
          adjoint=exact,central_difference=late_fd)
    check("gas_surface_mass_and_simplex",max(tr.diagnostics["precursor_relative"],tr.diagnostics["byproduct_relative"])<1e-10
          and tr.states.min()>=-1e-10 and tr.diagnostics["maximum_total_occupancy"]<=1+1e-10,**tr.diagnostics)
    # Explicit prescribed-hidden oracle on the manufactured truth only.
    oracle=TransientFEM(ops,cfg,cfg.c0,mode="oracle",tolerance=1e-12)
    reconstructed=oracle.forward(times,oracle_hidden=truth.states[:,teacher.zs])
    oracle_difference=float(np.max(abs(reconstructed.states-truth.states)))
    check("manufactured_oracle_coupling",oracle_difference<1e-9,
          max_state_difference=oracle_difference,not_a_supplied_ALD_GT_test=True)
    previous=np.zeros(solver.size); previous[solver.ps]=.1; previous[solver.zs]=.3
    new,_,_,_=solver.step(previous,.5,.53,False)
    db=(new[3:6]-previous[3:6])*cfg.c0
    dz=new[solver.zs]-previous[solver.zs]
    balance=float(db@ops.mass+cfg.gamma*(dz@ops.surface[solver.active])+.03*(new[3:6]*cfg.c0@ops.outlet))
    check("removal_returns_byproduct_to_gas",np.min(new[solver.zs]-previous[solver.zs])<0 and new[3:6].max()>0
          and abs(balance)<1e-15,absolute_inventory_error_mol=abs(balance))
    zero,_,_,_=solver.step(np.zeros(solver.size),.5,.53,False)
    check("no_adsorption_from_empty_gas",np.max(abs(zero))==0)
    fields_target=np.c_[obs.cp/cfg.c0,obs.cb/cfg.c0,obs.theta,np.zeros_like(obs.theta)]
    fields_target[~obs.training_mask(),:2*solver.n+solver.s]+=10
    check("withheld_snapshots_excluded",loss(fields_target,False)[0]<1e-20)
    torch_checks={"status":"not run","reason":TORCH_IMPORT_ERROR}
    if torch is not None:
        torch.set_num_threads(1); torch.manual_seed(41)
        net=HiddenRateNet(cfg.c0,width=8,depth=2)
        anchors=regularization_anchors([obs],cfg.c0,count=32)
        rows=directional_gradcheck(net,[obs],ops,smoothness=1e-6,anchors=anchors)
        check("Torch_parameter_VJP_and_smoothness",all(r["passed"] for r in rows))
        torch_checks={"status":"passed","checks":rows,"data":"manufactured; not 2D ALD GT"}
    report=dict(status="passed",kind="dependency-light manufactured conservative algebra verification",
        real_ALD_oracle_executed=False,real_ALD_training_executed=False,
        torch_check=torch_checks,checks=checks,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        created_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/"verification"/"self_test.json",report)
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=="__main__":
    main()
