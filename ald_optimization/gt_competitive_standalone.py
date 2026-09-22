#!/usr/bin/env python3
"""Standalone 2D competitive-adsorption GT generator for ALD FEML.

Everything required for the 2D GT is inside this file:
- SI configuration and pulse definition
- 2D reactor geometry and boundary tags
- prescribed carrier velocity
- conservative DOLFINx transport discretization
- ideal/soft/competitive surface kinetics (GT uses competitive)
- mass-balance diagnostics
- operator/data export for the standalone ML learner

No local project modules are imported.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import ufl
from scipy import sparse
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.fem import petsc as fp


# ========================= CONFIG =========================
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

# ========================= 2D GEOMETRY =========================
"""Planar surrogate. One metre of out-of-plane depth is understood."""

INLET, OUTLET, WAFER, TOP, SIDE = 11, 12, 21, 22, 23


@dataclass
class Reactor:
    domain: object
    tags: object
    reactive_tags: tuple
    description: str

    @property
    def ds(self):
        return ufl.Measure("ds", domain=self.domain, subdomain_data=self.tags)


def build_reactor_2d(cfg, benchmark=False, comm=MPI.COMM_WORLD):
    domain = mesh.create_rectangle(comm, [np.array([0.,0.]),np.array([cfg.length,cfg.height])],
                                   [cfg.nx,cfg.ny],cell_type=mesh.CellType.triangle)
    fdim = 1
    domain.topology.create_connectivity(fdim,2)
    boundary = mesh.exterior_facet_indices(domain.topology)
    mid = mesh.compute_midpoints(domain,fdim,boundary)
    values = np.full(len(boundary),SIDE,dtype=np.int32)
    values[np.isclose(mid[:,0],0)] = INLET
    values[np.isclose(mid[:,0],cfg.length)] = OUTLET
    values[np.isclose(mid[:,1],cfg.height)] = TOP
    bottom = np.isclose(mid[:,1],0)
    if benchmark:
        values[bottom] = WAFER
    else:
        mask = bottom & (mid[:,0]>=cfg.wafer_start-1e-12) & (mid[:,0]<=cfg.wafer_end+1e-12)
        values[mask] = WAFER
    tags = mesh.meshtags(domain,fdim,boundary,values)
    return Reactor(domain,tags,(WAFER,TOP) if benchmark else (WAFER,),
                   "parallel-plate plug-flow benchmark" if benchmark else "planar wafer surrogate")

def prescribed_velocity(reactor, cfg):
    """Uniform plug velocity used by the verified Stage-5 nonideal comparison."""
    d = reactor.domain.geometry.dim
    return fem.Constant(reactor.domain, np.array([cfg.mean_velocity] + [0.0]*(d-1), dtype=PETSc.ScalarType))


# ========================= SURFACE KINETICS =========================
"""Local constitutive laws independent of geometry, FEM and PETSc.

State rows: precursor/fast, slow-site, blocked-byproduct coverage.
All reaction fluxes are mol m^-2 s^-1; gas concentrations mol m^-3.
"""


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


def coefficients(c, state, T, parameters, byproduct=None, closure=None):
    """Return molar Robin velocities (precursor, byproduct), both m/s."""
    kp, ks, kb = rate_constants(T, parameters)
    if closure is not None and parameters.kinetics_model not in ("ideal", "learned_closure"):
        raise ValueError("The learned multiplier is defined on ideal kinetics only")
    if parameters.kinetics_model == "learned_closure" and closure is None:
        raise ValueError("learned_closure requires a supplied frozen closure callable")
    g = (closure(c, np.zeros_like(c) if byproduct is None else byproduct, coverage(state,parameters))
         if closure is not None else closure_multiplier(c, coverage(state,parameters), T, parameters))
    if np.any(~np.isfinite(g)) or np.any(g < 0):
        raise ValueError("Surface closure must be finite and nonnegative")
    if closure is not None and np.any(g > 1):
        raise ValueError("The competitive-recovery multiplier must be at most one")
    model = parameters.kinetics_model
    if model in ("ideal", "learned_closure"):
        return kp*g*(1-state[0]), np.zeros_like(c)
    if model == "soft_saturation":
        f = parameters.slow_fraction
        return g*((1-f)*kp*(1-state[0])+f*ks*(1-state[1])), np.zeros_like(c)
    empty = 1-state[0]-state[2]
    return kp*g*empty, kb*empty


def reaction_rate(c, theta, T, parameters, byproduct=None, closure=None):
    """Closure interface. theta is the complete three-row local state."""
    cb = np.zeros_like(c) if byproduct is None else byproduct
    if closure is not None:
        if parameters.kinetics_model not in ("ideal", "learned_closure"):
            raise ValueError("A supplied learned closure is defined on the ideal baseline")
    ap, ab = coefficients(c, theta, T, parameters, cb, closure)
    return ap*c, ab*cb


def implicit_update(old, c, byproduct, dt, parameters, guess=None, closure=None):
    """Backward Euler, analytically eliminated surface unknowns.

    For nonconstant learned g, evaluate at the nonlinear outer iterate.
    No concentration or coverage clipping is performed here.
    """
    p = parameters
    kp, ks, kb = rate_constants(p.temperature, p)
    theta = coverage(old if guess is None else guess,p)
    if p.kinetics_model == "learned_closure" and closure is None:
        raise ValueError("learned_closure requires a supplied closure callable")
    g = (closure(c, byproduct, theta) if closure is not None
         else closure_multiplier(c, theta, p.temperature, p))
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

# ========================= CONSERVATIVE TRANSPORT =========================
"""Conservative P1 FEM with lumped storage and boundary reaction quadrature.

Algebraic graph viscosity removes positive spatial off-diagonal entries. It
preserves column sums, hence global conservation, unlike clipping. No SUPG is
needed with this monotone low-order alternative. Numerical diffusion is
explicitly reported and assessed in the mesh/time refinement benchmark.
"""


def assembled_vector(form):
    v=fp.assemble_vector(fem.form(form))
    v.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE)
    data=v.array.real.copy(); v.destroy()
    return data


def monotone_operator(matrix):
    """B=A+G; G_ij=-max(A_ij,A_ji,0), G_ii=-sum_{j!=i}G_ij."""
    # petsc4py Mat.transpose() is in-place unless an output matrix is supplied.
    trans=matrix.copy()
    trans.transpose()
    result=matrix.copy()
    total=0.
    for i in range(*matrix.getOwnershipRange()):
        cols,vals=matrix.getRow(i)
        tcols,tvals=trans.getRow(i)
        reverse=dict(zip(tcols,tvals))
        new=vals.copy(); diagonal=np.flatnonzero(cols==i)
        added=0.
        for k,j in enumerate(cols):
            if j!=i:
                diffusion=max(float(vals[k]),float(reverse.get(j,0.)),0.)
                new[k]-=diffusion; added+=diffusion
        if not len(diagonal):
            raise RuntimeError("Missing FEM diagonal")
        new[diagonal[0]]+=added
        result.setValues(i,cols,new)
        total+=added/2
    result.assemble(); trans.destroy()
    return result,matrix.comm.tompi4py().allreduce(total)


class Transport:
    def __init__(self,reactor,cfg,velocity,diffusion_tensor=None,closure=None):
        self.reactor,self.cfg,self.velocity=reactor,cfg,velocity
        self.closure = closure
        self.mesh=reactor.domain; self.comm=self.mesh.comm
        self.V=fem.functionspace(self.mesh,("Lagrange",1))
        self.imap=self.V.dofmap.index_map
        self.n=self.imap.size_local
        self.coords=self.V.tabulate_dof_coordinates()[:self.n].copy()
        self.c=fem.Function(self.V); self.b=fem.Function(self.V)
        self.state=np.zeros((3,self.n))
        self.c_values=np.zeros(self.n); self.b_values=np.zeros(self.n)
        v=ufl.TestFunction(self.V); c=ufl.TrialFunction(self.V)
        dx=ufl.Measure("dx",domain=self.mesh); ds=reactor.ds
        n=ufl.FacetNormal(self.mesh); un=ufl.dot(velocity,n)
        # Outlet-only outflow. Inlet has prescribed total incoming flux.
        adv=-c*ufl.dot(velocity,ufl.grad(v))*dx+ufl.max_value(un,0)*c*v*ds(OUTLET)
        self.mass=assembled_vector(v*dx)
        self.surface=np.zeros(self.n)
        for tag in reactor.reactive_tags:
            self.surface+=assembled_vector(v*ds(tag))
        self.wafer=assembled_vector(v*ds(WAFER))
        self.inlet=assembled_vector(ufl.max_value(-un,0)*v*ds(INLET))
        self.outlet=assembled_vector(ufl.max_value(un,0)*v*ds(OUTLET))
        self.inlet_area=assembled_vector(v*ds(INLET))
        self.outlet_area=assembled_vector(v*ds(OUTLET))
        self.active=self.surface>0
        self.wafer_nodes=self.wafer>0
        self.inlet_local=np.flatnonzero(self.inlet_area>0).astype(np.int32)
        self.inlet_global=self.imap.local_to_global(self.inlet_local).astype(PETSc.IntType)
        diffusion=(cfg.diffusivity*ufl.inner(ufl.grad(c),ufl.grad(v)) if diffusion_tensor is None
                   else ufl.inner(ufl.dot(ufl.as_matrix(diffusion_tensor),ufl.grad(c)),ufl.grad(v)))
        matrices=[]; viscosity=[]
        for diff in (diffusion,cfg.byproduct_diffusivity*ufl.inner(ufl.grad(c),ufl.grad(v))):
            raw=fp.assemble_matrix(fem.form(adv+diff*dx)); raw.assemble()
            operator,added=monotone_operator(raw); raw.destroy()
            matrices.append(operator); viscosity.append(added)
        self.B,self.Bb=matrices
        self.graph_viscosity=viscosity
        self.rhs=self.B.createVecRight(); self.sol=self.rhs.duplicate()
        self.diagonal=self.rhs.duplicate(); self.temp=self.rhs.duplicate()
        self.ksp=PETSc.KSP().create(self.comm)
        self.ksp.setType("gmres"); self.ksp.setGMRESRestart(100)
        self.ksp.getPC().setType("ilu" if self.comm.size==1 else "bjacobi")
        self.ksp.setTolerances(rtol=cfg.linear_rtol,atol=1e-20,max_it=1000)
        self.ksp.setErrorIfNotConverged(True)
        self.qin=self.sum(self.inlet); self.qout=self.sum(self.outlet)
        if self.qin<=0 or self.qout<=0:
            raise RuntimeError("Inlet and outlet fluxes must both be positive")
        self.steps=[]; self.snapshots=[]; self.t=0.; self.rejected=0
        self.cumulative_in=0.; self.cumulative_out=0.
        self.cumulative_bp_out=0.; self.cumulative_bp_generated=0.

    def sum(self,x):
        return self.comm.allreduce(float(np.sum(x)),op=MPI.SUM)

    def max(self,x):
        return self.comm.allreduce(float(np.max(x)) if np.size(x) else -np.inf,op=MPI.MAX)

    def min(self,x):
        return self.comm.allreduce(float(np.min(x)) if np.size(x) else np.inf,op=MPI.MIN)

    def linear(self,B,old,robin,source,dt,cin=None):
        A=B.copy()
        self.diagonal.array[:]=self.mass/dt+self.surface*robin
        A.setDiagonal(self.diagonal,addv=PETSc.InsertMode.ADD_VALUES); A.assemble()
        self.rhs.array[:]=self.mass*old/dt+source
        if cin is not None:
            A.zeroRows(self.inlet_global,diag=1.)
            self.rhs.array[self.inlet_local]=cin
        self.ksp.setOperators(A); self.ksp.solve(self.rhs,self.sol)
        value=self.sol.array.real.copy(); its=self.ksp.getIterationNumber()
        A.destroy()
        return value,its

    def operator_total(self,B,values):
        self.sol.array[:]=values; B.mult(self.sol,self.temp)
        return self.sum(self.temp.array)

    def attempt(self,dt):
        p=self.cfg; oldc=self.c_values.copy(); oldb=self.b_values.copy(); old=self.state.copy()
        guess=old.copy(); cg=oldc.copy(); bg=oldb.copy()
        cin=pulse_average(self.t,self.t+dt,p)
        source=self.inlet*cin if p.inlet_condition=="flux" else np.zeros(self.n)
        boundary=cin if p.inlet_condition=="dirichlet" else None
        linear_iterations=0
        for iteration in range(1,p.max_coupling_iterations+1):
            ap,ab=coefficients(cg,guess,p.temperature,p,bg,self.closure)
            c,its=self.linear(self.B,oldc,ap,source,dt,boundary)
            linear_iterations+=its
            b,its=self.linear(self.Bb,oldb,ab,p.byproduct_yield*self.surface*ap*c,dt)
            linear_iterations+=its
            if min(self.min(c),self.min(b)) < -p.negative_tolerance*p.c0:
                raise ArithmeticError("Negative gas concentration beyond numerical tolerance")
            candidate=implicit_update(old,c,b,dt,p,guess,self.closure)
            candidate[:,~self.active]=0.
            error=max(self.max(abs(candidate-guess)),self.max(abs(c-cg))/p.c0,
                      self.max(abs(b-bg))/p.c0)
            if error<p.coupling_tolerance:
                # c,b solved with guess. Surface accepted from the same flux:
                # candidate differs only by controlled fixed-point residual.
                break
            w=p.relaxation
            guess=w*candidate+(1-w)*guess
            cg=w*c+(1-w)*cg; bg=w*b+(1-w)*bg
        else:
            raise ArithmeticError("Gas/surface nonlinear coupling did not converge")
        total_occupancy=candidate[0]+candidate[2]
        if self.min(candidate)<-1e-9 or self.max(candidate)>1+1e-9 or self.max(total_occupancy)>1+1e-9:
            raise ArithmeticError("Surface bounds violated")
        theta=coverage(candidate,p); oldtheta=coverage(old,p)
        uptake=self.sum(self.surface*p.gamma*(theta-oldtheta))
        bpads=self.sum(self.surface*p.gamma*(candidate[2]-old[2]))
        out=self.operator_total(self.B,c)
        bpout=self.operator_total(self.Bb,b)
        supplied=self.qin*cin
        if boundary is not None:
            self.sol.array[:]=c; self.B.mult(self.sol,self.temp)
            residual=self.mass*(c-oldc)/dt+self.temp.array+self.surface*ap*c
            supplied=self.sum(residual[self.inlet_local])
        balance=self.sum(self.mass*(c-oldc))+uptake+dt*(out-supplied)
        generated=dt*self.sum(p.byproduct_yield*self.surface*ap*c)
        bp_balance=self.sum(self.mass*(b-oldb))+bpads+dt*bpout-generated
        scale=max(dt*supplied,uptake,self.qin*p.c0*dt*1e-3,1e-30)
        if abs(balance)/scale>2e-5 or abs(bp_balance)/scale>2e-5:
            raise ArithmeticError("Discrete gas/surface mass balance failed")
        self.c_values,self.b_values,self.state=c,b,candidate
        self.c.x.array[:self.n]=c; self.c.x.scatter_forward()
        self.b.x.array[:self.n]=b; self.b.x.scatter_forward()
        self.t+=dt
        self.cumulative_in+=dt*supplied; self.cumulative_out+=dt*out
        self.cumulative_bp_out+=dt*bpout; self.cumulative_bp_generated+=generated
        total_ads=self.sum(self.surface*p.gamma*theta)
        gas=self.sum(self.mass*c)
        inventory_error=gas+total_ads+self.cumulative_out-self.cumulative_in
        bp_inventory_error=(self.sum(self.mass*b)+self.sum(self.surface*p.gamma*candidate[2])
                            +self.cumulative_bp_out-self.cumulative_bp_generated)
        sw=self.sum(self.wafer)
        row={"time_s":self.t,"dt_s":dt,"inlet_c_mol_m3":cin,
             "theta_min":self.min(theta[self.wafer_nodes]),
             "theta_mean":self.sum(self.wafer*theta)/sw,
             "theta_max":self.max(theta[self.wafer_nodes]),
             "blocked_mean":self.sum(self.wafer*candidate[2])/sw,
             "c_min_mol_m3":self.min(c),"c_max_mol_m3":self.max(c),
             "byproduct_min_mol_m3":self.min(b),
             "outlet_precursor_c_mol_m3":self.sum(self.outlet*c)/self.qout,
             "outlet_byproduct_c_mol_m3":self.sum(self.outlet*b)/self.qout,
             "outlet_precursor_area_c_mol_m3":self.sum(self.outlet_area*c)/self.sum(self.outlet_area),
             "gas_inventory_mol":gas,"surface_uptake_mol":total_ads,
             "cumulative_in_mol":self.cumulative_in,"cumulative_out_mol":self.cumulative_out,
             "balance_residual_mol":inventory_error,
             "balance_relative":inventory_error/max(self.cumulative_in,1e-30),
             "byproduct_balance_residual_mol":bp_inventory_error,
             "byproduct_balance_relative":bp_inventory_error/max(self.cumulative_bp_generated,1e-30),
             "step_balance_mol":balance,"coupling_iterations":iteration,
             "coupling_error":error,"linear_iterations":linear_iterations,
             "rejected_steps_total":self.rejected,
             "qcm_normalized_mass":self.sum(self.wafer*theta)/sw}
        # Spatial QCM-like finite patches; report normalized mass, no invented film density.
        for j,fraction in enumerate((0.1,0.3,0.5,0.7,0.9)):
            if self.mesh.geometry.dim==2:
                center=p.wafer_start+fraction*(p.wafer_end-p.wafer_start)
                mask=abs(self.coords[:,0]-center)<max(p.length/p.nx,0.012)
            else:
                center=(2*fraction-1)*p.wafer_radius*0.8
                mask=(self.coords[:,0]-center)**2+self.coords[:,1]**2<0.035**2
            weight=self.wafer*mask; area=self.sum(weight)
            row[f"qcm_{j+1}_coverage"]=self.sum(weight*theta)/area if area>0 else None
        self.steps.append(row)
        return row

    def collect(self,fields):
        """Gather owned CG1 DOFs in global order; root-only plotting."""
        chunks=self.comm.gather((self.imap.local_range[0],self.coords,fields),root=0)
        if self.comm.rank:
            return None
        chunks.sort(key=lambda item:item[0])
        return {"coordinates":np.concatenate([a[1] for a in chunks]),
                **{key:np.concatenate([a[2][key] for a in chunks]) for key in fields}}

    def snapshot(self):
        fields={"concentration":self.c_values,"byproduct":self.b_values,
                "theta":coverage(self.state,self.cfg),"blocked":self.state[2],
                "wafer_weight":self.wafer}
        result=self.collect(fields)
        if result is not None:
            result["time"]=self.t
            self.snapshots.append(result)

    def run(self,times=None):
        p=self.cfg
        requested=[0.1,0.2,0.3,0.5,0.7,1.0,p.end_time] if times is None else list(times)
        events=sorted(set([round(t,12) for t in requested if 0<t<=p.end_time]+[p.end_time,
                     p.rise_time,p.pulse_duration,p.pulse_duration+p.rise_time]))
        events=[t for t in events if t>0]
        snapshot_times=np.array(sorted(set([t for t in requested if 0<t<=p.end_time]+[p.end_time])))
        self.snapshot()
        dt=p.dt
        for target in events:
            while self.t<target-1e-11:
                trial=min(dt,target-self.t)
                try:
                    self.attempt(trial)
                except (ArithmeticError,PETSc.Error) as exc:
                    self.rejected+=1; dt=trial/2
                    if dt<p.min_dt:
                        raise RuntimeError(f"dt fell below {p.min_dt:g} at t={self.t:g}: {exc}") from exc
                    continue
                dt=min(p.dt,dt*1.25)
            if np.any(abs(snapshot_times-self.t)<1e-9):
                self.snapshot()
        return self.steps,self.snapshots

    def close(self):
        for item in (self.B,self.Bb,self.rhs,self.sol,self.diagonal,self.temp,self.ksp):
            item.destroy()

# ========================= GT DATA/EXPORT =========================
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

    @classmethod
    def from_transport(cls, transport):
        if transport.comm.size != 1:
            raise RuntimeError("GT generation currently requires one MPI rank")

        def csr(A):
            indptr, indices, data = A.getValuesCSR()
            return sparse.csr_matrix(
                (data.copy(), indices.copy(), indptr.copy()), shape=A.getSize()
            )

        return cls(
            B=csr(transport.B),
            Bb=csr(transport.Bb),
            mass=transport.mass.copy(),
            surface=transport.surface.copy(),
            wafer=transport.wafer.copy(),
            inlet=transport.inlet.copy(),
            outlet=transport.outlet.copy(),
            coordinates=transport.coords.copy(),
        )


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


def stage5_like_config() -> Config:
    """Stage-5 nonideal benchmark without requiring config.stage5_config()."""
    return replace(
        Config(),
        precursor_pressure=50e-3 * TORR,
        pulse_duration=0.8,
        purge_duration=1.0,
        kinetics_model="competitive_adsorption",
    )


def time_grid(cfg: Config) -> np.ndarray:
    events = sorted(
        set(
            [
                0.0,
                cfg.rise_time,
                cfg.pulse_duration,
                cfg.pulse_duration + cfg.rise_time,
                cfg.end_time,
            ]
        )
    )
    times = [0.0]
    for left, right in zip(events[:-1], events[1:]):
        if right <= left:
            continue
        count = max(1, int(np.ceil((right - left) / cfg.dt - 1e-10)))
        times.extend(np.linspace(left, right, count + 1)[1:])
    return np.asarray(times, dtype=float)


def phase(t: float, cfg: Config) -> str:
    if t <= cfg.rise_time:
        return "rise"
    if t <= cfg.pulse_duration:
        return "dose"
    if t <= cfg.pulse_duration + cfg.rise_time:
        return "fall"
    return "purge"


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


def save_operators(path: Path, ops: Operators) -> None:
    np.savez_compressed(
        path,
        B_data=ops.B.data,
        B_indices=ops.B.indices,
        B_indptr=ops.B.indptr,
        B_shape=np.asarray(ops.B.shape, dtype=np.int64),
        Bb_data=ops.Bb.data,
        Bb_indices=ops.Bb.indices,
        Bb_indptr=ops.Bb.indptr,
        Bb_shape=np.asarray(ops.Bb.shape, dtype=np.int64),
        mass=ops.mass,
        surface=ops.surface,
        wafer=ops.wafer,
        inlet=ops.inlet,
        outlet=ops.outlet,
        coordinates=ops.coordinates,
    )


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


def generate_case(name: str, cfg: Config, role: str, root: Path):
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Run GT generation with one MPI rank")

    case = replace(cfg, kinetics_model="competitive_adsorption")
    case.validate()
    reactor = build_reactor_2d(case)
    transport = Transport(reactor, case, prescribed_velocity(reactor, case))
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)

    times = time_grid(case)
    ops = Operators.from_transport(transport)
    active = np.flatnonzero(transport.active)

    cp = [transport.c_values.copy()]
    cb = [transport.b_values.copy()]
    theta = [transport.state[0, active].copy()]
    theta_b = [transport.state[2, active].copy()]

    try:
        for t0, t1 in zip(times[:-1], times[1:]):
            transport.attempt(float(t1 - t0))
            cp.append(transport.c_values.copy())
            cb.append(transport.b_values.copy())
            theta.append(transport.state[0, active].copy())
            theta_b.append(transport.state[2, active].copy())

        cp = np.asarray(cp)
        cb = np.asarray(cb)
        theta = np.asarray(theta)
        theta_b = np.asarray(theta_b)

        np.savez_compressed(
            folder / "fields.npz",
            times=times,
            cp_mol_m3=cp,
            cb_mol_m3=cb,
            theta_p=theta,
            surface_dofs=active,
            role=np.asarray(role),
        )
        # Kept separate so the ML script cannot accidentally use it as a target.
        np.savez_compressed(
            folder / "hidden_surface_state.npz",
            times=times,
            theta_b=theta_b,
        )

        rows = []
        for row in transport.steps:
            clean = dict(row)
            clean["phase"] = phase(float(row["time_s"]), case)
            rows.append(clean)
        write_rows(folder / "observables.csv", rows)
        write_json(
            folder / "config.json",
            {"name": name, "role": role, "config": asdict(case)},
        )
        write_json(
            folder / "diagnostics.json",
            {
                "theta_mean_final": float(rows[-1]["theta_mean"]),
                "theta_min_final": float(rows[-1]["theta_min"]),
                "theta_max_final": float(rows[-1]["theta_max"]),
                "max_abs_precursor_balance_relative": float(
                    max(abs(r["balance_relative"]) for r in rows)
                ),
                "max_abs_byproduct_balance_relative": float(
                    max(abs(r["byproduct_balance_relative"]) for r in rows)
                ),
                "maximum_total_site_occupancy": float(np.max(theta + theta_b)),
                "steps": int(len(times) - 1),
            },
        )
        return ops
    finally:
        transport.close()


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




def parse_args():
    p = argparse.ArgumentParser(description="Standalone competitive-adsorption GT generator")
    p.add_argument("--quick", action="store_true", help="Coarse smoke run")
    p.add_argument("--preset", choices=["single", "multi"], default="single")
    p.add_argument("--nx", type=int)
    p.add_argument("--ny", type=int)
    p.add_argument("--dt", type=float)
    p.add_argument("--output", default="outputs/08_gt_competitive")
    return p.parse_args()


def main():
    args = parse_args()
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("This standalone GT generator currently requires one MPI rank")

    root = Path(args.output).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    cfg = stage5_like_config()
    if args.quick:
        cfg = replace(cfg, nx=min(cfg.nx, 60), ny=min(cfg.ny, 6), dt=max(cfg.dt, 0.02))
    if args.nx is not None:
        cfg = replace(cfg, nx=args.nx)
    if args.ny is not None:
        cfg = replace(cfg, ny=args.ny)
    if args.dt is not None:
        cfg = replace(cfg, dt=args.dt)
    cfg.validate()

    conditions = [
        ("train_50mTorr_0p8s", cfg, "train"),
        ("heldout_35mTorr_0p6s", replace(cfg, precursor_pressure=35e-3*TORR, pulse_duration=0.6), "held_out"),
    ]
    if args.preset == "multi":
        conditions.extend([
            ("train_20mTorr_0p2s", replace(cfg, precursor_pressure=20e-3*TORR, pulse_duration=0.2), "train"),
            ("train_50mTorr_0p4s", replace(cfg, precursor_pressure=50e-3*TORR, pulse_duration=0.4), "train"),
            ("train_75mTorr_0p8s", replace(cfg, precursor_pressure=75e-3*TORR, pulse_duration=0.8), "train"),
        ])

    reference_ops = None
    summary = []
    for name, case, role in conditions:
        print(f"[GT] {name} | role={role} | P={case.precursor_pressure/TORR*1000:.1f} mTorr | dose={case.pulse_duration:g} s", flush=True)
        ops = generate_case(name, case, role, root)
        if reference_ops is None:
            reference_ops = ops
            save_operators(root/"operators.npz", ops)
        else:
            if ops.B.shape != reference_ops.B.shape or not np.allclose(ops.coordinates, reference_ops.coordinates):
                raise RuntimeError("All GT conditions must use the same mesh/operator layout")
        obs = load_case(root/name)
        with np.load(root/name/"hidden_surface_state.npz") as d:
            theta_b = d["theta_b"]
        summary.append({
            "name": name,
            "role": role,
            "pressure_mTorr": case.precursor_pressure/TORR*1000,
            "pulse_s": case.pulse_duration,
            "purge_s": case.purge_duration,
            "steps": len(obs.times)-1,
            "theta_mean_final": float(np.mean(obs.theta[-1])),
            "max_hidden_blocked": float(np.max(theta_b)),
        })

    write_rows(root/"summary.csv", summary)
    write_json(root/"manifest.json", {
        "generator": "gt_competitive_standalone.py",
        "physics": "2D conservative DOLFINx transport + competitive adsorption GT",
        "hidden_state": "theta_b stored separately and not intended as an ML training target",
        "conditions": summary,
        "base_config": asdict(cfg),
    })
    print(f"[DONE] GT written to {root}")


if __name__ == "__main__":
    main()
