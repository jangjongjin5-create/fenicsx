#!/usr/bin/env python3
"""2D-trained hidden-state dynamics -> frozen 3D wafer transfer.

ONE standalone file. Requires the existing best_model.pt, not the 2D modules.
Production: serial DOLFINx 0.11 + Gmsh + PETSc/MUMPS + PyTorch + SciPy.

  python run_3d_zeroshot_standalone.py --checkpoint best_model.pt --quick
  python run_3d_zeroshot_standalone.py --checkpoint best_model.pt
  python run_3d_zeroshot_standalone.py --training-output outputs/09_hidden_state_feml
  python run_3d_zeroshot_standalone.py --strong
  python run_3d_zeroshot_standalone.py --checkpoint best_model.pt --self-test

--self-test needs only NumPy/SciPy; it does NOT execute the 3D reactor.
Default: short coarse preflight, then 50 mTorr / 0.8 s full calculation.
--quick: ONLY the first 0.10 s on a coarse mesh; NOT research accuracy.
--strong: additionally 35 mTorr / 0.6 s, only if the first full test qualifies.
Every new run replaces ONLY outputs/10_3d_zeroshot (under --output-root).
No optimizer, training, autograd, adaptive fit, GT oracle, or ideal simulation.

Checkpoint inference uses exactly the saved dense/tanh/softplus operations.
Input tangents are propagated analytically in NumPy; no reverse propagation.
Production loads the identical PyTorch module, calls eval(), disables parameter
gradients, and checks NumPy/PyTorch output parity before any reactor run.
Every committed ML surface state is also checked against frozen PyTorch.

Full 3D gas fields are never dumped. Only time traces, final wafer fields,
bounded local-state samples and the six requested PNG figures are saved.
Missing 2D predicted fields => state-support diagnostic unavailable, never 0%.
Missing training manifest => explicit, SHA-bound source-code fallback for the
supplied checkpoint only. Other checkpoints require their training manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import pickle
import shutil
import sys
import time
import traceback
import warnings
import zipfile
from collections import OrderedDict
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

R, NA, TORR = 8.31446261815324, 6.02214076e23, 133.32236842105263
INLET, OUTLET, WAFER, TOP, SIDE = 11, 12, 21, 22, 23
SUPPLIED_CHECKPOINT_SHA256 = "4a0b1efbe865d0ec6df92c1db853abd5fef4a5ab9e5dbd543d514ba2e4880de7"
SOURCE_PROVENANCE = {
    "training_code_sha256": "94d8dad07a85137173dff8ff179f8030b3e8a8c98de699d602cfacabbc8a3f33",
    "gt_code_sha256": "efa11fee3418230e03bc1c9327f6eed80762f8be9c5f1545326faae524edc296",
    "stage7_geometry_sha256": "ed30964d4671bfa3fe9b4998487fae95bdc5de47b32ebe2eb53b8a5ffb0fb663",
    "stage7_flow_sha256": "bc735248b5c31d9cd60f82a9dd6b7538e59cb56885f9eb4d8e2654069a2bab3d",
    "provided_2d_metrics_sha256": "009833b07df9f859c4a14d56342b2375af4579141020ec43555ebf4d76a34972",
}
TRAINING_CONDITIONS = ["train_20mTorr_0p2s", "train_50mTorr_0p4s",
                       "train_50mTorr_0p8s", "train_75mTorr_0p8s"]
INPUTS = ["cp/c_ref", "cb/c_ref", "theta_p", "predicted_theta_b_hat"]
RATE_DEFINITION = "f=(kp*c_ref/Gamma)*(cb/c_ref)*free*q_ads - theta_b_hat*q_rem/time_ref"
MASS_TOL = 2e-5
FLOW_DIV_WARNING = 0.05  # Diagnostic, predeclared; never tuned to GT error.
PURGE_THRESHOLD = 1e-3  # Both volume-mean and outlet cp, divided by case inlet c0.
SATURATION_TARGET = 0.99


@dataclass(frozen=True)
class Config:
    temperature: float = 473.0
    pressure: float = TORR
    precursor_molar_mass: float = .150
    byproduct_molar_mass: float = .016
    carrier_molar_mass: float = .0280134
    diffusivity: float = .01
    byproduct_diffusivity: float = .05
    kinematic_viscosity: float = .05
    grad_div_factor: float = 10.0
    beta0: float = .01
    beta_byproduct: float = .001  # Read ONLY by competitive GT rate construction.
    byproduct_yield: float = 1.0
    site_area: float = 24e-20  # Inherited unverified 24 A^2 interpretation.
    precursor_pressure: float = 50e-3 * TORR
    height: float = .020
    chamber_radius: float = .250
    wafer_radius: float = .150
    port_radius: float = .015
    port_offset: float = .220
    flow_sccm: float = 300.0
    standard_temperature: float = 273.15
    standard_pressure: float = 101325.0
    pulse_duration: float = .8
    rise_time: float = .02
    purge_duration: float = 1.0
    dt: float = .005
    mesh_size_3d: float = .012
    port_mesh_size: float = .004
    flow_model: str = "navier_stokes"

    @property
    def c0(self): return self.precursor_pressure / (R * self.temperature)
    @property
    def gamma(self): return 1 / (NA * self.site_area)
    @property
    def end_time(self): return self.pulse_duration + self.rise_time + self.purge_duration
    @property
    def actual_volume_flow(self):
        return self.flow_sccm * 1e-6 / 60 * self.temperature / self.standard_temperature * self.standard_pressure / self.pressure
    @property
    def density(self): return self.pressure * self.carrier_molar_mass / (R * self.temperature)


def thermal_speed(molar_mass, temperature):
    return math.sqrt(8 * R * temperature / (math.pi * molar_mass))


def pulse_integral(t, cfg):
    t = max(0., float(t)); tr, td = cfg.rise_time, cfg.pulse_duration
    if tr == 0: return min(t, td)
    if t < tr: return t*t/(2*tr)
    if t < td: return t-tr/2
    if t < td+tr:
        a = t-td
        return td-tr/2+a-a*a/(2*tr)
    return td


def pulse_average(t0, t1, cfg):
    return cfg.c0 * (pulse_integral(t1, cfg)-pulse_integral(t0, cfg))/(t1-t0)


def time_grid(cfg, stop=None):
    stop = cfg.end_time if stop is None else min(stop, cfg.end_time)
    events = sorted(set([0., stop]+[v for v in (cfg.rise_time, cfg.pulse_duration,
                       cfg.pulse_duration+cfg.rise_time) if 0 < v < stop]))
    values = [0.]
    for a, b in zip(events[:-1], events[1:]):
        n = max(1, int(np.ceil((b-a)/cfg.dt-1e-10)))
        values.extend(np.linspace(a, b, n+1)[1:])
    return np.array(values)


def validate_config(cfg):
    positive = (cfg.temperature, cfg.pressure, cfg.precursor_molar_mass, cfg.byproduct_molar_mass,
        cfg.carrier_molar_mass, cfg.diffusivity, cfg.byproduct_diffusivity, cfg.kinematic_viscosity,
        cfg.site_area, cfg.height, cfg.chamber_radius, cfg.wafer_radius, cfg.port_radius,
        cfg.flow_sccm, cfg.standard_temperature, cfg.standard_pressure, cfg.dt,
        cfg.mesh_size_3d, cfg.port_mesh_size, cfg.precursor_pressure)
    if not all(np.isfinite(v) and v > 0 for v in positive): raise ValueError("Nonpositive or nonfinite physical configuration")
    if not (0 <= cfg.rise_time <= cfg.pulse_duration and cfg.purge_duration >= 0):
        raise ValueError("Invalid pulse/purge duration")
    if not (0 <= cfg.beta0 <= 1 and 0 <= cfg.beta_byproduct <= 1 and cfg.byproduct_yield >= 0):
        raise ValueError("Invalid sticking coefficient or byproduct yield")
    if cfg.port_offset+cfg.port_radius >= cfg.chamber_radius or cfg.port_offset-cfg.port_radius <= cfg.wafer_radius:
        raise ValueError("Ports overlap wafer or chamber wall")
    if cfg.flow_model != "navier_stokes": raise ValueError("This transfer preserves the Stage 7 Navier-Stokes flow model")


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays_hash(values, parameters_only=False):
    # Identical order/name/byte convention to the supplied training state_hash.
    h = hashlib.sha256()
    for name, value in values.items():
        if parameters_only and name in ("concentration_ref", "time_ref"): continue
        h.update(name.encode()); h.update(np.asarray(value).tobytes())
    return h.hexdigest()


def restricted_checkpoint(path):
    """Offline audit only: allowlisted tensor archive reader, never arbitrary pickle.

    This is not a production substitute for torch.load(weights_only=True).
    Optimizer tensors in the archive are deserialized as inert arrays; no
    optimizer object is ever instantiated or used.
    """
    def rebuild(storage, offset, size, stride, requires_grad, hooks, metadata=None):
        a = storage
        offset, size, stride = int(offset), tuple(size), tuple(stride)
        if offset < 0 or any(n < 0 for n in size) or any(s < 0 for s in stride):
            raise ValueError("Invalid tensor layout")
        last = offset + sum((n-1)*s for n, s in zip(size, stride))
        if last >= len(a): raise ValueError("Tensor exceeds storage")
        return np.ndarray(size, dtype=a.dtype, buffer=a, offset=offset*a.itemsize,
                          strides=tuple(s*a.itemsize for s in stride)).copy()
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        if len(names) != 1: raise ValueError("Expected one PyTorch data.pkl")
        prefix = names[0][:-len("data.pkl")]
        if archive.read(prefix+"byteorder") != b"little":
            raise ValueError("Offline reader supports little endian checkpoints only")
        class Restricted(pickle.Unpickler):
            def find_class(self, module, name):
                allowed = {("collections", "OrderedDict"): OrderedDict,
                    ("torch", "DoubleStorage"): np.dtype("<f8"),
                    ("torch", "FloatStorage"): np.dtype("<f4"),
                    ("torch._utils", "_rebuild_tensor_v2"): rebuild}
                if (module, name) not in allowed:
                    raise pickle.UnpicklingError(f"Forbidden checkpoint global: {module}.{name}")
                return allowed[module, name]
            def persistent_load(self, pid):
                if not isinstance(pid, tuple) or len(pid) != 5 or pid[0] != "storage":
                    raise pickle.UnpicklingError("Unexpected persistent object")
                _, dtype, key, location, length = pid
                if not str(key).isdigit() or not isinstance(dtype, np.dtype):
                    raise pickle.UnpicklingError("Unexpected storage")
                a = np.frombuffer(archive.read(prefix+"data/"+key), dtype=dtype)
                if len(a) != length: raise ValueError("Storage length mismatch")
                return a
        return Restricted(io.BytesIO(archive.read(names[0]))).load()


class FrozenRates:
    """Exact saved network and forward-mode input tangent, immutable weights."""
    def __init__(self, state, spec):
        self.spec = spec
        self.state = OrderedDict((k, np.array(v, copy=True)) for k, v in state.items())
        required = {"concentration_ref", "time_ref"}
        for head in ("ads", "rem"):
            for i in range(spec["depth"]+1):
                required.update([f"{head}.{2*i}.weight", f"{head}.{2*i}.bias"])
        if set(self.state) != required: raise ValueError("Checkpoint tensor keys differ from training architecture")
        for head in ("ads", "rem"):
            for i in range(spec["depth"]+1):
                n_in = 4 if i == 0 else spec["width"]
                n_out = 1 if i == spec["depth"] else spec["width"]
                if self.state[f"{head}.{2*i}.weight"].shape != (n_out, n_in) or self.state[f"{head}.{2*i}.bias"].shape != (n_out,):
                    raise ValueError("Checkpoint layer shape mismatch")
        for a in self.state.values():
            if a.dtype != np.float64 or not np.all(np.isfinite(a)):
                raise ValueError("Expected finite float64 weights and normalization buffers")
            a.setflags(write=False)
        self.ref = float(self.state["concentration_ref"])
        self.time_ref = float(self.state["time_ref"])
        if min(self.ref, self.time_ref) <= 0: raise ValueError("Nonpositive normalization")
        if self.ref != spec["concentration_ref_mol_m3"] or self.time_ref != spec["time_ref_s"]:
            raise ValueError("Specification/buffer normalization mismatch")

    def calculate(self, x, tangent=False):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != 4: raise ValueError("Only four local physical states are allowed")
        outputs, tangents = [], []
        for head in ("ads", "rem"):
            h = x
            if tangent: d = np.broadcast_to(np.eye(4), (len(x), 4, 4)).copy()
            for i in range(self.spec["depth"]+1):
                w = self.state[f"{head}.{2*i}.weight"]
                h = h @ w.T + self.state[f"{head}.{2*i}.bias"]
                if tangent: d = np.einsum("oi,nij->noj", w, d, optimize=True)
                if i < self.spec["depth"]:
                    h = np.tanh(h)
                    if tangent: d *= (1-h*h)[:, :, None]
            # torch functional.softplus defaults beta=1, threshold=20.
            q = np.where(h > 20., h, np.logaddexp(0., h))
            outputs.append(q[:, 0])
            if tangent:
                sigmoid = np.where(h > 20., 1., np.exp(-np.logaddexp(0., -h)))
                tangents.append((d*sigmoid[:, :, None])[:, 0, :])
        q = np.stack(outputs, axis=1)
        return (q, np.stack(tangents, axis=1)) if tangent else q

    def value(self, x): return self.calculate(x, False)
    def value_tangent(self, x): return self.calculate(x, True)


def validate_spec(checkpoint):
    s = checkpoint["specification"]
    expected = {"inputs": INPUTS, "activation": "tanh", "output_activation": "softplus",
                "rate_definition": RATE_DEFINITION, "dtype": "float64", "width": 32, "depth": 3,
                "outputs": ["positive_adsorption_multiplier", "positive_removal_multiplier"]}
    for k, v in expected.items():
        if s.get(k) != v: raise ValueError(f"Unsupported/mismatched training specification: {k}")
    if checkpoint.get("hidden_gt_used") is not False:
        raise ValueError("Checkpoint does not declare hidden_gt_used=False")
    if checkpoint.get("training_conditions") != TRAINING_CONDITIONS:
        raise ValueError("Training conditions differ from this transfer protocol")
    return s


def load_checkpoint(path, offline=False):
    if offline:
        checkpoint = restricted_checkpoint(path); model = None
        state = checkpoint["state_dict"]
    else:
        import torch
        from torch import nn
        torch.set_num_threads(1)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        spec = validate_spec(checkpoint)
        class HiddenRateNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("concentration_ref", torch.tensor(spec["concentration_ref_mol_m3"], dtype=torch.float64))
                self.register_buffer("time_ref", torch.tensor(spec["time_ref_s"], dtype=torch.float64))
                def network():
                    layers = []
                    for i in range(spec["depth"]):
                        layers += [nn.Linear(4 if i == 0 else spec["width"], spec["width"]), nn.Tanh()]
                    return nn.Sequential(*layers, nn.Linear(spec["width"], 1)).double()
                self.ads, self.rem = network(), network()
            def forward(self, x):
                return torch.cat([torch.nn.functional.softplus(self.ads(x)),
                                  torch.nn.functional.softplus(self.rem(x))], dim=-1)
        model = HiddenRateNet()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError("Model is not frozen")
        state = OrderedDict((k, v.detach().cpu().numpy()) for k, v in model.state_dict().items())
    spec = validate_spec(checkpoint)
    adapter = FrozenRates(state, spec)
    metadata = {k: checkpoint[k] for k in ("specification", "best_epoch", "best_loss",
                "training_conditions", "hidden_gt_used", "selection", "seed")}
    audit = dict(checkpoint_path=str(path.resolve()), checkpoint_sha256=file_hash(path),
        parameter_sha256_before=arrays_hash(adapter.state, True),
        state_sha256_before=arrays_hash(adapter.state),
        metadata=metadata, normalization_source="checkpoint buffers, cross-checked with checkpoint specification",
        model_eval=None if offline else True, all_parameters_require_grad_false=None if offline else True,
        torch_inference_executed=False, optimizer_created=False, backpropagation_used=False)
    if model is not None:
        x = np.random.default_rng(713).uniform(0, 1, (128, 4)); x[:, 2:] *= .45
        audit["numpy_torch_max_absolute_difference"] = parity(model, adapter, x)
        audit["torch_inference_executed"] = True
    return model, adapter, audit


def parity(model, adapter, x):
    import torch
    with torch.inference_mode():
        truth = model(torch.as_tensor(x, dtype=torch.float64)).cpu().numpy()
    predicted = adapter.value(x)
    if not np.allclose(predicted, truth, rtol=2e-12, atol=2e-14):
        raise ArithmeticError("NumPy frozen inference differs from the saved PyTorch architecture")
    return float(np.max(abs(predicted-truth)))


def verify_freeze(path, model, adapter, audit):
    current = adapter.state
    if model is not None:
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError("Freeze flags changed")
        current = OrderedDict((k, v.detach().cpu().numpy()) for k, v in model.state_dict().items())
    param = arrays_hash(current, True); state = arrays_hash(current)
    changed = (param != audit["parameter_sha256_before"] or state != audit["state_sha256_before"]
               or arrays_hash(adapter.state) != audit["state_sha256_before"]
               or file_hash(path) != audit["checkpoint_sha256"])
    result = dict(parameter_sha256_after=param, state_sha256_after=state, weights_changed=bool(changed))
    if changed: raise RuntimeError("Frozen checkpoint, network, or normalization changed")
    return result


def load_training_manifest(root, explicit, audit):
    path = Path(explicit).resolve() if explicit else root/"manifest.json"
    if explicit and not path.is_file(): raise FileNotFoundError(path)
    if path.is_file():
        m = read_json(path)
        if m.get("learner_id") != "explicit_hidden_state_feml" or not m.get("status", "").startswith("completed"):
            raise ValueError("Training manifest is not a completed hidden-state training run")
        if m.get("frozen_state_sha256") != audit["state_sha256_before"]:
            raise ValueError("Training manifest belongs to a different checkpoint")
        if m.get("network") != audit["metadata"]["specification"]:
            raise ValueError("Training manifest/checkpoint architecture or normalization mismatch")
        cases = {c["name"]: c["config"] for c in m["training_cases"]}
        if set(cases) != set(TRAINING_CONDITIONS): raise ValueError("Training manifest cases differ")
        base = cases["train_50mTorr_0p8s"]
        keys = {f.name for f in fields(Config)}
        if any(k not in base for k in keys): raise ValueError("Incomplete training physical configuration")
        cfg = Config(**{k: base[k] for k in keys})
        for key in keys-{"precursor_pressure", "pulse_duration"}:
            if any(c[key] != base[key] for c in cases.values()):
                raise ValueError(f"Training configurations disagree on {key}")
        if any(c.get("inlet_condition") != "flux" for c in cases.values()):
            raise ValueError("Only the trained conservative flux inlet is supported")
        return cfg, m, dict(status="verified", path=str(path), sha256=file_hash(path))
    if audit["checkpoint_sha256"] != SUPPLIED_CHECKPOINT_SHA256:
        raise ValueError("Unknown checkpoint requires its original --training-manifest")
    return Config(), None, dict(status="missing", path=str(path),
        fallback="Physical constants and empty initial state transcribed from supplied training/GT code; checkpoint SHA-bound",
        provenance=SOURCE_PROVENANCE,
        limitation="Original run manifest unavailable; physical runtime overrides cannot be independently excluded")


# ======================= STAGE 7 GEOMETRY / FLOW =======================
@dataclass
class Reactor:
    domain: object
    tags: object
    reactive_tags: tuple
    description: str
    @property
    def ds(self): return ufl.Measure("ds", domain=self.domain, subdomain_data=self.tags)


def require_fem():
    global MPI, ufl, fem, PETSc, gmshio, element, mixed_element, LinearProblem
    from mpi4py import MPI
    import ufl
    from dolfinx import fem
    from dolfinx.io import gmsh as gmshio
    from petsc4py import PETSc
    from basix.ufl import element, mixed_element
    from dolfinx.fem.petsc import LinearProblem
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Use one MPI rank: this standalone transport solve uses serial SciPy sparse matrices")
    if np.dtype(PETSc.ScalarType).kind != "f":
        raise RuntimeError("A real-scalar PETSc build is required")


def integral(expr,comm):
    return comm.allreduce(fem.assemble_scalar(fem.form(expr)),op=MPI.SUM).real


def build_geometry(cfg):
    comm=MPI.COMM_WORLD

    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal",0)
    if comm.rank == 0:
        gmsh.model.add("ald_300mm")
        o = gmsh.model.occ
        volume = o.addCylinder(0,0,0,0,0,cfg.height,cfg.chamber_radius)
        wafer = o.addDisk(0,0,0,cfg.wafer_radius,cfg.wafer_radius)
        inlet = o.addDisk(-cfg.port_offset,0,0,cfg.port_radius,cfg.port_radius)
        outlet = o.addDisk(cfg.port_offset,0,0,cfg.port_radius,cfg.port_radius)
        o.fragment([(3,volume)],[(2,wafer),(2,inlet),(2,outlet)])
        o.synchronize()
        volumes = [t for d,t in gmsh.model.getEntities(3)]
        gmsh.model.addPhysicalGroup(3,volumes,1)
        boundary = gmsh.model.getBoundary([(3,t) for t in volumes],combined=True,oriented=False)
        groups = {i:[] for i in (INLET,OUTLET,WAFER,TOP,SIDE)}
        for d,t in boundary:
            x,y,z = o.getCenterOfMass(d,t)
            area = o.getMass(d,t)
            if abs(z-cfg.height)<1e-7:
                label = TOP
            elif abs(z)<1e-7 and abs(area-np.pi*cfg.port_radius**2)<1e-7:
                label = INLET if x<0 else OUTLET
            elif abs(z)<1e-7 and abs(area-np.pi*cfg.wafer_radius**2)<1e-7:
                label = WAFER
            else:
                label = SIDE
            groups[label].append(t)
        for label,faces in groups.items():
            if not faces:
                raise RuntimeError(f"Gmsh missing boundary tag {label}")
            gmsh.model.addPhysicalGroup(2,faces,label)
        gmsh.option.setNumber("Mesh.MeshSizeMin",cfg.port_mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax",cfg.mesh_size_3d)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",12)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0),cfg.mesh_size_3d)
        points = gmsh.model.getBoundary([(2,t) for t in groups[INLET]+groups[OUTLET]],
                                       combined=False,oriented=False,recursive=True)
        gmsh.model.mesh.setSize(points,cfg.port_mesh_size)
        gmsh.model.mesh.generate(3)
    data = gmshio.model_to_mesh(gmsh.model,comm,0,gdim=3)
    gmsh.finalize()
    return Reactor(data.mesh,data.facet_tags,(WAFER,),"300 mm wafer, bottom-port disk reactor")


def solve_flow(reactor,cfg,diagnostic_path=None):
    m=reactor.domain; comm=m.comm; d=m.geometry.dim
    cell=m.basix_cell()
    W=fem.functionspace(m,mixed_element([element("Lagrange",cell,2,shape=(d,)),
                                         element("Lagrange",cell,1)]))
    V,_=W.sub(0).collapse()
    inlet=fem.Function(V)
    if d==2:
        def profile(x):
            a=x[1]/cfg.height
            return np.vstack((6*cfg.mean_velocity*a*(1-a),np.zeros_like(a)))
    else:
        def profile(x):
            r2=((x[0]+cfg.port_offset)**2+x[1]**2)/cfg.port_radius**2
            return np.vstack((np.zeros_like(r2),np.zeros_like(r2),
                              2*cfg.actual_volume_flow/(np.pi*cfg.port_radius**2)*(1-r2)))
    inlet.interpolate(profile)
    fdim=m.topology.dim-1
    wall_facets=np.unique(np.concatenate([reactor.tags.find(t) for t in (WAFER,TOP,SIDE)]))
    # The polygonal circular rim has P2 edge nodes inside the true circle.
    # Enforce shared wall DOFs exactly zero before normalizing inlet flux.
    wall_blocks=fem.locate_dofs_topological(V,fdim,wall_facets)
    inlet.x.array.reshape(-1,d)[wall_blocks,:]=0.
    inlet.x.scatter_forward()
    n=ufl.FacetNormal(m); dx=ufl.Measure("dx",domain=m); ds=reactor.ds
    requested=cfg.mean_velocity*cfg.height if d==2 else cfg.actual_volume_flow
    measured=-integral(ufl.dot(inlet,n)*ds(INLET),comm)
    if not np.isfinite(measured) or measured<=0:
        raise ArithmeticError("Inlet profile has nonpositive integrated flux")
    inlet.x.array[:]*=requested/measured
    inlet.x.scatter_forward()
    zero=fem.Function(V)
    wall_dofs=fem.locate_dofs_topological((W.sub(0),V),fdim,wall_facets)
    in_dofs=fem.locate_dofs_topological((W.sub(0),V),fdim,reactor.tags.find(INLET))
    bcs=[fem.dirichletbc(inlet,in_dofs,W.sub(0)),fem.dirichletbc(zero,wall_dofs,W.sub(0))]
    u,p=ufl.TrialFunctions(W); v,q=ufl.TestFunctions(W)
    previous=fem.Function(V)
    inertia=fem.Constant(m,PETSc.ScalarType(0.))
    a=(cfg.kinematic_viscosity*ufl.inner(ufl.grad(u),ufl.grad(v))
       +cfg.grad_div_factor*cfg.kinematic_viscosity*ufl.div(u)*ufl.div(v)
       +inertia*ufl.inner(ufl.dot(ufl.grad(u),previous),v)
       -p*ufl.div(v)+q*ufl.div(u))*dx
    rhs=ufl.inner(fem.Constant(m,np.zeros(d,dtype=PETSc.ScalarType)),v)*dx
    solution=fem.Function(W)
    problem=LinearProblem(a,rhs,bcs=bcs,u=solution,petsc_options_prefix="ald_flow_",
        petsc_options={"ksp_type":"preonly","pc_type":"lu",
                       "pc_factor_mat_solver_type":"mumps","ksp_error_if_not_converged":True})
    history=[]
    for iteration in range(1,81):
        problem.solve(); solution.x.scatter_forward()
        velocity=solution.sub(0).collapse()
        local=np.sum(abs(velocity.x.array-previous.x.array)**2)
        denom=np.sum(abs(velocity.x.array)**2)
        relative=np.sqrt(comm.allreduce(local)/max(comm.allreduce(denom),1e-30))
        history.append({"iteration":iteration,"relative_velocity_change":float(relative),
                        "inertia_included":bool(float(inertia.value))})
        if cfg.flow_model=="stokes" or (iteration>1 and relative<1e-8):
            break
        previous.x.array[:]=0.7*velocity.x.array+0.3*previous.x.array
        previous.x.scatter_forward()
        inertia.value=PETSc.ScalarType(1.)
    else:
        raise RuntimeError("Steady Navier-Stokes Picard iteration failed")
    pressure=solution.sub(1).collapse()  # kinematic pressure p/rho
    pressure.x.array[:]*=cfg.density
    pressure.name="pressure_Pa_relative_to_outlet_traction"
    velocity.name="carrier_velocity"
    qin=-integral(ufl.dot(velocity,n)*ds(INLET),comm)
    qout=integral(ufl.dot(velocity,n)*ds(OUTLET),comm)
    div2=integral(ufl.div(velocity)**2*dx,comm)
    volume=integral(fem.Constant(m,PETSc.ScalarType(1.))*dx,comm)
    speed_rms=np.sqrt(integral(ufl.inner(velocity,velocity)*dx,comm)/volume)
    coord=ufl.SpatialCoordinate(m)
    if d==2:
        region=ufl.conditional(ufl.And(ufl.ge(coord[0],cfg.wafer_start),
                                      ufl.le(coord[0],cfg.wafer_end)),1.,0.)
        speed=velocity[0]
    else:
        region=ufl.conditional(ufl.le(coord[0]**2+coord[1]**2,cfg.wafer_radius**2),1.,0.)
        speed=ufl.sqrt(velocity[0]**2+velocity[1]**2)
    avg=integral(region*speed*dx,comm)/integral(region*dx,comm)
    mean_speed=integral(ufl.sqrt(ufl.inner(velocity,velocity))*dx,comm)/volume
    nodal_speed=np.linalg.norm(velocity.x.array.reshape(-1,d),axis=1)
    backflow=integral(ufl.max_value(-ufl.dot(velocity,n),0)*ds(OUTLET),comm)
    data={"mean_velocity_m_s":float(mean_speed),
          "max_velocity_m_s":float(nodal_speed.max()),
          "max_velocity_definition":"maximum sampled at P2 velocity nodes; not certified cell-interior maximum",
          "speed_rms_m_s":float(speed_rms),"chamber_volume_m3":float(volume),
          "requested_flow_m3_s":float(requested),
          "outlet_backflow_m3_s":float(backflow),
          "outlet_backflow_fraction":float(backflow/qin),
          "model":cfg.flow_model,"flow_in":qin,"flow_out":qout,
          "flow_balance_relative":abs(qout-qin)/qin,"divergence_rms_s_inv":float(np.sqrt(div2/volume)),
          "divergence_scaled_by_H_over_u_rms":float(np.sqrt(div2/volume)*cfg.height/speed_rms),
          "grad_div_stabilization_m2_s":cfg.grad_div_factor*cfg.kinematic_viscosity,
          "wafer_region_mean_horizontal_velocity_m_s":avg,"paper_wafer_velocity_m_s":0.5,
          "difference_from_paper_velocity_percent":100*(avg/0.5-1),
          "pressure_approximation":"incompressible; p is mechanical gauge pressure",
          "iterations":history}
    if d==2:
        exact=ufl.as_vector([6*cfg.mean_velocity*(coord[1]/cfg.height)*(1-coord[1]/cfg.height),0.])
        data["poiseuille_velocity_relative_L2"]=float(np.sqrt(
            integral(ufl.inner(velocity-exact,velocity-exact)*dx,comm)/
            integral(ufl.inner(exact,exact)*dx,comm)))
        data["poiseuille_pressure_drop_Pa"]=12*cfg.density*cfg.kinematic_viscosity*cfg.mean_velocity*cfg.length/cfg.height**2
    data["warnings"]=[]
    if not all(np.isfinite(x) for x in (qin,qout,div2,volume,speed_rms,avg,mean_speed)) or qin<=0 or qout<=0:
        raise ArithmeticError("Nonfinite or nonphysical steady flow")
    if data["divergence_scaled_by_H_over_u_rms"]>FLOW_DIV_WARNING:
        data["warnings"].append("Scaled divergence exceeds 0.05; transport agreement alone cannot validate the continuum flow")
    data["interpretation_qualified_by_flow"]=bool(data["warnings"])
    if diagnostic_path is not None: write_json(diagnostic_path,data)
    if data["flow_balance_relative"]>0.01:
        raise RuntimeError(f"Flow balance failed: {data['flow_balance_relative']:.3e}")
    if data["outlet_backflow_fraction"]>0.01:
        raise ArithmeticError("Outlet backflow exceeds the outflow-only transport assumption")
    for message in data["warnings"]: warnings.warn(message)
    return velocity,pressure,data


# ======================= CONSERVATIVE DISCRETE PHYSICS =======================
@dataclass(frozen=True)
class Operators:
    B: sparse.csr_matrix
    Bb: sparse.csr_matrix
    mass: np.ndarray
    surface: np.ndarray
    wafer: np.ndarray
    inlet: np.ndarray
    outlet: np.ndarray
    coordinates: np.ndarray


def graph_viscosity(A):
    """Same symmetric graph diffusion as the supplied 2D/Stage 7 transport."""
    d = A.maximum(A.T).maximum(0).tocsr()
    d.setdiag(0); d.eliminate_zeros()
    B = (A-d+sparse.diags(np.asarray(d.sum(axis=1)).ravel())).tocsr()
    return B, float(d.sum()/2)


def assemble_operators(reactor, cfg, velocity):
    from dolfinx.fem import petsc as fp
    m = reactor.domain; V = fem.functionspace(m, ("Lagrange", 1))
    v, c = ufl.TestFunction(V), ufl.TrialFunction(V)
    dx, ds = ufl.Measure("dx", domain=m), reactor.ds
    un = ufl.dot(velocity, ufl.FacetNormal(m))
    def vector(form):
        a = fp.assemble_vector(fem.form(form))
        a.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        values = a.array.real.copy(); a.destroy()
        return values
    mass = vector(v*dx); surface = vector(v*ds(WAFER))
    inlet = vector(ufl.max_value(-un, 0)*v*ds(INLET))
    outlet = vector(ufl.max_value(un, 0)*v*ds(OUTLET))
    adv = -c*ufl.dot(velocity, ufl.grad(v))*dx+ufl.max_value(un, 0)*c*v*ds(OUTLET)
    matrices, reports = [], []
    for diffusion in (cfg.diffusivity, cfg.byproduct_diffusivity):
        raw = fp.assemble_matrix(fem.form(adv+diffusion*ufl.inner(ufl.grad(c), ufl.grad(v))*dx))
        raw.assemble(); indptr, indices, values = raw.getValuesCSR()
        A = sparse.csr_matrix((values.copy(), indices.copy(), indptr.copy()), shape=raw.getSize())
        raw.destroy()
        B, added = graph_viscosity(A)
        error = float(np.max(abs(np.asarray(B.sum(axis=0)).ravel()-outlet)))
        relative = error/max(float(outlet.sum()), 1e-30)
        if relative > 1e-9: raise ArithmeticError("Transport column sums do not match outlet flux")
        matrices.append(B)
        reports.append(dict(graph_viscosity_sum_m3_s=added, column_balance_relative=relative))
    if np.any(mass <= 0) or surface.sum() <= 0 or min(inlet.sum(), outlet.sum()) <= 0:
        raise ArithmeticError("Invalid FEM mass, surface, or port weights")
    active = np.flatnonzero(surface > 0)
    index = {int(d): i for i, d in enumerate(active)}
    triangles = []
    for facet in reactor.tags.find(WAFER):
        dofs = fem.locate_dofs_topological(V, 2, np.array([facet], dtype=np.int32))
        if len(dofs) != 3: raise RuntimeError("Expected triangular P1 wafer facets")
        triangles.append([index[int(d)] for d in dofs])
    coords = V.tabulate_dof_coordinates().copy()
    op = Operators(*matrices, mass, surface, surface.copy(), inlet, outlet, coords)
    stats = dict(cell_count=int(m.topology.index_map(3).size_local),
        vertex_count=int(m.topology.index_map(0).size_local), gas_dof_count=len(mass),
        wafer_dof_count=len(active), wafer_triangle_count=len(triangles),
        chamber_volume_m3=float(mass.sum()), wafer_area_m2=float(surface.sum()),
        nominal_wafer_area_m2=math.pi*cfg.wafer_radius**2,
        nominal_mesh_size_m=cfg.mesh_size_3d, port_mesh_size_m=cfg.port_mesh_size,
        transport_assembly=reports,
        boundary_areas_m2={str(tag): float(integral(fem.Constant(m, PETSc.ScalarType(1.))*ds(tag), m.comm))
                           for tag in (INLET, OUTLET, WAFER, TOP, SIDE)})
    return op, np.asarray(triangles, dtype=np.int32), stats


class SurfaceLaw:
    """ML law has no GT state and no byproduct sticking coefficient."""
    def __init__(self, kp, ref, gamma, adapter=None, kb=None, time_ref=1.):
        if (adapter is None) == (kb is None): raise ValueError("Select exactly one independent surface law")
        self.kp, self.kappa = kp, kp*ref/gamma
        self.adapter, self.kb_scaled = adapter, None if kb is None else kb*ref/gamma
        self.rem_scale = 1/time_ref

    def rates(self, x, jacobian=True):
        a, b, p, z = x.T; free = 1-p-z
        rp = self.kp*a*free
        dr = np.column_stack([self.kp*free, np.zeros(len(x)), -self.kp*a, -self.kp*a])
        if self.adapter is None:
            f = self.kb_scaled*b*free
            df = np.column_stack([np.zeros(len(x)), self.kb_scaled*free,
                                  -self.kb_scaled*b, -self.kb_scaled*b])
            return rp, f, dr, df
        if jacobian: q, dq = self.adapter.value_tangent(x)
        else: q = self.adapter.value(x)
        if not np.all(np.isfinite(q)) or np.any(q < 0): raise ArithmeticError("Invalid rate multiplier")
        ads, rem = self.kappa*b*free, self.rem_scale*z
        f = ads*q[:, 0]-rem*q[:, 1]
        if not jacobian: return rp, f, None, None
        df = ads[:, None]*dq[:, 0, :]-rem[:, None]*dq[:, 1, :]
        df[:, 1] += self.kappa*free*q[:, 0]
        df[:, 2] -= self.kappa*b*q[:, 0]
        df[:, 3] -= self.kappa*b*q[:, 0]+self.rem_scale*q[:, 1]
        return rp, f, dr, df


class TransientFEM:
    """Monolithic backward Euler; no trajectory/factor history kept in memory.

    u=(cp/c_ref, cb/c_ref, theta_p, z). Exactly the trained weak-discrete
    coupling: precursor consumes Gamma*dtheta_p; gas byproduct gets the same
    Gamma*dz as the surface equation with the opposite sign, including removal.
    """
    def __init__(self, ops, cfg, ref, law, tolerance=2e-11):
        self.ops, self.cfg, self.ref, self.law = ops, cfg, float(ref), law
        self.tolerance = tolerance; self.n = len(ops.mass)
        self.active = np.flatnonzero(ops.surface > 0); self.s = len(self.active)
        self.size = 2*self.n+2*self.s
        self.ps = slice(2*self.n, 2*self.n+self.s)
        self.zs = slice(2*self.n+self.s, self.size)
        a, j = self.active, np.arange(self.s); w = ops.surface[a]/ops.mass[a]
        self.Hp = sparse.csc_matrix((np.r_[w, -cfg.byproduct_yield*w,
            np.full(self.s, -ref/cfg.gamma)], (np.r_[a, self.n+a, 2*self.n+j], np.tile(j, 3))),
            shape=(self.size, self.s))
        self.Hf = sparse.csc_matrix((np.r_[w*cfg.gamma/ref, -np.ones(self.s)],
            (np.r_[self.n+a, 2*self.n+self.s+j], np.tile(j, 2))), shape=(self.size, self.s))
        self.G = sparse.block_diag([sparse.diags(1/ops.mass)@ops.B,
            sparse.diags(1/ops.mass)@ops.Bb, sparse.csr_matrix((2*self.s, 2*self.s))], format="csc")
        self.I = sparse.eye(self.size, format="csc")
        self.cached_dt, self.cached_A = None, None

    def local_state(self, u):
        return np.stack([u[self.active], u[self.n+self.active], u[self.ps], u[self.zs]], axis=1)

    def derivative_matrix(self, derivative):
        columns = np.r_[self.active, self.n+self.active,
            np.arange(self.ps.start, self.ps.stop), np.arange(self.zs.start, self.zs.stop)]
        return sparse.csc_matrix((derivative.T.ravel(), (np.tile(np.arange(self.s), 4), columns)),
                                 shape=(self.s, self.size))

    def residual_jacobian(self, u, previous, t0, t1, jacobian=True):
        dt = t1-t0
        rp, f, dr, df = self.law.rates(self.local_state(u), jacobian)
        if dt != self.cached_dt:
            self.cached_dt, self.cached_A = dt, self.I+dt*self.G
        A = self.cached_A
        residual = A@u-previous+dt*(self.Hp@rp+self.Hf@f)
        residual[:self.n] -= dt*self.ops.inlet/self.ops.mass*pulse_average(t0, t1, self.cfg)/self.ref
        if not jacobian: return residual
        K = A+dt*(self.Hp@self.derivative_matrix(dr)+self.Hf@self.derivative_matrix(df))
        return residual, K.tocsc()

    def admissible(self, u):
        return np.all(np.isfinite(u)) and u.min() >= -1e-10 and np.max(u[self.ps]+u[self.zs]) <= 1+1e-10

    def step(self, previous, t0, t1):
        u = previous.copy()
        for iteration in range(50):
            residual, K = self.residual_jacobian(u, previous, t0, t1)
            norm = float(np.linalg.norm(residual, np.inf))
            if not np.isfinite(norm): raise ArithmeticError(f"Nonfinite Newton residual at {t1:g} s")
            if norm < self.tolerance:
                if not self.admissible(u): raise ArithmeticError("Concentration/simplex bound failure")
                return u, iteration, norm
            increment = splu(K).solve(-residual)
            alpha = 1.
            for _ in range(32):
                trial = u+alpha*increment
                if self.admissible(trial):
                    value = float(np.linalg.norm(self.residual_jacobian(trial, previous, t0, t1, False), np.inf))
                    if value < self.tolerance or value <= (1-1e-4*alpha)*norm:
                        u = trial; break
                alpha *= .5
            else: raise ArithmeticError(f"Bounded Newton line search failed at t={t1:g}; residual={norm:.3e}")
        raise ArithmeticError(f"Newton failed at t={t1:g}; no clipping, retuning, or automatic retries")


class StateSampler:
    """Exact extrema/counts on committed wafer states; bounded uniform reservoir."""
    def __init__(self, reference=None, capacity=12000):
        self.reference = reference; self.capacity = capacity
        self.minimum = np.full(4, np.inf); self.maximum = np.full(4, -np.inf)
        self.count = 0; self.outside = 0; self.outside_by_axis = np.zeros(4, dtype=np.int64)
        self.weight = 0.; self.outside_weight = 0.
        self.samples = np.empty((0, 4)); self.priorities = np.empty(0)
        self.rng = np.random.default_rng(510)

    def update(self, x, weights=None):
        self.minimum = np.minimum(self.minimum, x.min(axis=0))
        self.maximum = np.maximum(self.maximum, x.max(axis=0)); self.count += len(x)
        if self.reference is not None:
            lo, hi = self.reference["minimum"], self.reference["maximum"]
            mask = (x < lo-1e-10) | (x > hi+1e-10); outside = np.any(mask, axis=1)
            self.outside += int(outside.sum()); self.outside_by_axis += mask.sum(axis=0)
            if weights is not None:
                self.weight += float(weights.sum()); self.outside_weight += float(weights@outside)
        keys = np.r_[self.priorities, self.rng.random(len(x))]
        values = np.vstack([self.samples, x])
        keep = np.argpartition(keys, self.capacity-1)[:self.capacity] if len(keys) > self.capacity else np.arange(len(keys))
        self.priorities, self.samples = keys[keep], values[keep]

    def report(self):
        available = self.reference is not None
        result = dict(status="available" if available else "unavailable_missing_verified_2D_predictions",
            input_order=INPUTS, three_d_minimum=self.minimum.tolist(), three_d_maximum=self.maximum.tolist(),
            committed_node_time_samples=self.count, reservoir_size=len(self.samples),
            training_minimum=self.reference["minimum"].tolist() if available else None,
            training_maximum=self.reference["maximum"].tolist() if available else None,
            fraction_outside_training_envelope=self.outside/self.count if available else None,
            fraction_outside_by_variable=(self.outside_by_axis/self.count).tolist() if available else None,
            area_time_weighted_fraction_outside=self.outside_weight/self.weight if available and self.weight else None,
            envelope_absolute_tolerance=1e-10,
            definition="All accepted wafer nodes at all saved integration times, including t=0; no Newton trial iterates. Area-time fraction excludes t=0.",
            reference_definition="Frozen best-checkpoint predictions at the four 2D training conditions; not a log of all optimizer iterates; held-out excluded")
        if available:
            from scipy.spatial import cKDTree
            scale = np.maximum(self.reference["maximum"]-self.reference["minimum"], 1e-12)
            tree = cKDTree(self.reference["samples"]/scale)
            d, _ = tree.query(self.samples/scale)
            result["sampled_4D_nearest_neighbor_distance"] = dict(mean=float(d.mean()),
                p95=float(np.percentile(d, 95)), maximum=float(d.max()),
                normalization="2D training min/max range per input; neither model nor physics normalization changes",
                approximate=True, two_d_reference_sample_count=len(self.reference["samples"]))
        return result


def load_training_support(root, manifest, adapter):
    if manifest is None:
        return None, dict(status="unavailable", reason="No checkpoint-matched completed training manifest")
    sampler = StateSampler(capacity=24000); sources = []
    for name in TRAINING_CONDITIONS:
        path = root/"predictions"/(name+"_hidden_state.npz")
        if not path.is_file():
            return None, dict(status="unavailable", reason=f"Missing frozen 2D prediction: {path}")
        with np.load(path, allow_pickle=False) as a:
            dofs = a["surface_dofs"]
            cp, cb, p, z = a["cp_mol_m3"][:, dofs]/adapter.ref, a["cb_mol_m3"][:, dofs]/adapter.ref, a["theta_p"], a["theta_b_hat"]
            if cp.shape != cb.shape or cp.shape != p.shape or p.shape != z.shape:
                raise ValueError("Misaligned 2D state-support fields")
            for i in range(len(a["times"])):
                x = np.column_stack([cp[i], cb[i], p[i], z[i]])
                if not np.all(np.isfinite(x)): raise ValueError("Nonfinite 2D predicted states")
                sampler.update(x)
        sources.append(dict(path=str(path), sha256=file_hash(path)))
    reference = dict(minimum=sampler.minimum, maximum=sampler.maximum, samples=sampler.samples)
    return reference, dict(status="available", sources=sources, samples_examined=sampler.count,
        minimum=sampler.minimum.tolist(), maximum=sampler.maximum.tolist(),
        checkpoint_binding="completed training manifest frozen_state_sha256 equals loaded state hash",
        limitation="Original training format does not put checkpoint hashes in each NPZ; assumes the completed output tree was not mixed or manually edited")


def stream_simulation(solver, times, triangles, folder, sampler=None, model=None):
    """No input result paths: this function cannot load a GT trajectory."""
    start = time.monotonic(); folder.mkdir(parents=True, exist_ok=True)
    op, cfg, ref = solver.ops, solver.cfg, solver.ref
    active = solver.active; w = op.wafer[active]; cap = cfg.gamma*op.surface[active]
    u = np.zeros(solver.size)  # Exact empty-gas/empty-surface training initial state.
    incoming = outp = outb = 0.; rows = []; parity_error = 0.
    for i, t in enumerate(times):
        dt = 0. if i == 0 else float(t-times[i-1]); nit = 0; residual = 0.
        if i:
            u, nit, residual = solver.step(u, times[i-1], t)
            incoming += dt*op.inlet.sum()*pulse_average(times[i-1], t, cfg)
        cp, cb, p, z = u[:solver.n]*ref, u[solver.n:2*solver.n]*ref, u[solver.ps], u[solver.zs]
        outp += dt*float(cp@op.outlet); outb += dt*float(cb@op.outlet)
        gas_p, gas_b, surf_p, surf_b = float(cp@op.mass), float(cb@op.mass), float(p@cap), float(z@cap)
        rows.append(dict(time_s=float(t), dt_s=dt, mean_theta_p=float(p@w/w.sum()), min_theta_p=float(p.min()),
            max_theta_p=float(p.max()), mean_theta_b=float(z@w/w.sum()), max_theta_b=float(z.max()),
            min_theta_b=float(z.min()), maximum_total_occupancy=float((p+z).max()),
            outlet_cp_mol_m3=float(cp@op.outlet/op.outlet.sum()), outlet_cb_mol_m3=float(cb@op.outlet/op.outlet.sum()),
            chamber_mean_cp_mol_m3=gas_p/float(op.mass.sum()), chamber_mean_cb_mol_m3=gas_b/float(op.mass.sum()),
            gas_cp_inventory_mol=gas_p, gas_cb_inventory_mol=gas_b,
            precursor_min_mol_m3=float(cp.min()), precursor_max_mol_m3=float(cp.max()),
            byproduct_min_mol_m3=float(cb.min()), byproduct_max_mol_m3=float(cb.max()),
            incoming_precursor_mol=float(incoming), outgoing_precursor_mol=float(outp), outgoing_byproduct_mol=float(outb),
            adsorbed_precursor_mol=surf_p, adsorbed_byproduct_mol=surf_b,
            precursor_balance_mol=float(gas_p+surf_p+outp-incoming),
            byproduct_balance_mol=float(gas_b+surf_b+outb-cfg.byproduct_yield*surf_p),
            newton_iterations=nit, residual_inf=residual))
        if sampler is not None:
            x = solver.local_state(u); sampler.update(x, dt*w)
            if model is not None:
                parity_error = max(parity_error, parity(model, solver.law.adapter, x))
        if i and (i % 25 == 0 or i == len(times)-1):
            print(f"[{folder.parent.name}/{folder.name}] {i}/{len(times)-1} t={t:.4f} s, Newton={nit}, residual={residual:.2e}", flush=True)
    with (folder/"transients.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(folder/"wafer_final.npz", coordinates=op.coordinates[active], triangles=triangles,
                        area_weights=w, theta_p=p, theta_b=z, final_time_s=times[-1])
    maximum = lambda key: float(max(r[key] for r in rows))
    minimum = lambda key: float(min(r[key] for r in rows))
    pa = max(abs(r["precursor_balance_mol"]) for r in rows)
    ba = max(abs(r["byproduct_balance_mol"]) for r in rows)
    generated = cfg.byproduct_yield*rows[-1]["adsorbed_precursor_mol"]
    diagnostics = dict(runtime_s=time.monotonic()-start, step_count=len(times)-1,
        precursor_absolute_mol=pa, byproduct_absolute_mol=ba,
        precursor_relative=pa/max(incoming, 1e-30), byproduct_relative=ba/max(incoming, 1e-30),
        byproduct_relative_to_generated=ba/max(generated, 1e-30),
        mass_normalizer_mol=float(incoming), generated_byproduct_mol=float(generated),
        mass_normalizer="total injected precursor in the actual 3D chamber [mol], no out-of-plane factor",
        precursor_min_mol_m3=minimum("precursor_min_mol_m3"), precursor_max_mol_m3=maximum("precursor_max_mol_m3"),
        byproduct_min_mol_m3=minimum("byproduct_min_mol_m3"), byproduct_max_mol_m3=maximum("byproduct_max_mol_m3"),
        theta_p_min=minimum("min_theta_p"), theta_p_max=maximum("max_theta_p"),
        theta_b_min=minimum("min_theta_b"), theta_b_max=maximum("max_theta_b"),
        maximum_total_occupancy=maximum("maximum_total_occupancy"),
        max_newton_iterations=int(maximum("newton_iterations")),
        total_newton_iterations=sum(r["newton_iterations"] for r in rows), maximum_residual=maximum("residual_inf"),
        min_dt=float(np.diff(times).min()), max_dt=float(np.diff(times).max()),
        clipping=False, numpy_torch_max_absolute_difference=parity_error if model is not None else None)
    diagnostics["passed"] = max(diagnostics[k] for k in ("precursor_relative", "byproduct_relative", "byproduct_relative_to_generated")) <= MASS_TOL
    write_json(folder/"diagnostics.json", diagnostics)
    if sampler is not None:
        write_json(folder/"state_coverage.json", sampler.report())
        np.savez_compressed(folder/"local_state_samples.npz", local_states=sampler.samples)
    if not diagnostics["passed"]: raise ArithmeticError(f"Mass-balance failure: {folder}")


def run_competitive_gt(ops, cfg, ref, times, triangles, folder):
    kp = cfg.beta0*thermal_speed(cfg.precursor_molar_mass, cfg.temperature)/4
    kb = cfg.beta_byproduct*thermal_speed(cfg.byproduct_molar_mass, cfg.temperature)/4
    law = SurfaceLaw(kp, ref, cfg.gamma, kb=kb)
    stream_simulation(TransientFEM(ops, cfg, ref, law), times, triangles, folder)


def run_frozen_ml(ops, cfg, adapter, model, times, triangles, folder, reference):
    # Deliberately has no GT result, trajectory, metrics, or path argument.
    kp = cfg.beta0*thermal_speed(cfg.precursor_molar_mass, cfg.temperature)/4
    law = SurfaceLaw(kp, adapter.ref, cfg.gamma, adapter=adapter, time_ref=adapter.time_ref)
    stream_simulation(TransientFEM(ops, cfg, adapter.ref, law), times, triangles, folder,
                      StateSampler(reference), model)


# ======================= POST-HOC EVALUATION ONLY =======================
def load_trace(folder):
    with (folder/"transients.csv").open(encoding="utf-8") as f: rows = list(csv.DictReader(f))
    return {k: np.asarray([float(r[k]) for r in rows]) for k in rows[0]}


def load_wafer(folder):
    with np.load(folder/"wafer_final.npz", allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def relative_l2(pred, truth, weight):
    den = float(np.sum(weight*truth**2)); num = float(np.sum(weight*(pred-truth)**2))
    return 100*math.sqrt(num/den) if den > 1e-30 else (0. if num <= 1e-30 else None)


def field_metrics(pred, truth, weight):
    delta = abs(pred-truth)
    return dict(relative_L2_pct=relative_l2(pred, truth, weight),
        MAE=float(delta@weight/weight.sum()), maximum_absolute_error=float(delta.max()))


def uniformity(x, weight):
    mean = float(x@weight/weight.sum())
    std = float(np.sqrt((x-mean)**2@weight/weight.sum()))
    return dict(mean=mean, minimum=float(x.min()), maximum=float(x.max()), std=std,
                CV=std/mean if mean > 1e-14 else None, quadrature="lumped P1 wafer area weights")


def crossing(times, mask, start=0., sustained=False):
    mask = np.asarray(mask, dtype=bool) & (times >= start-1e-12)
    if sustained: mask &= np.logical_and.accumulate(mask[::-1])[::-1]
    indexes = np.flatnonzero(mask)
    return dict(status="reached" if len(indexes) else "not reached",
                time_s=float(times[indexes[0]]) if len(indexes) else None,
                definition="first accepted timestep satisfying criterion"+(" and staying below through end of simulation" if sustained else ""))


def timing_comparison(gt, ml):
    a, b = gt["time_s"], ml["time_s"]
    return dict(competitive_gt=gt, frozen_ml=ml,
                relative_error_pct=100*abs(b-a)/a if a is not None and b is not None and a > 0 else None)


def evaluate(gt_folder, ml_folder, cfg, freeze, flow, quick=False):
    # The ONLY comparison entry point. Called after BOTH simulations completed.
    gt, ml = load_wafer(gt_folder), load_wafer(ml_folder)
    tg, tm = load_trace(gt_folder), load_trace(ml_folder)
    for key in ("coordinates", "triangles", "area_weights", "final_time_s"):
        if not np.array_equal(gt[key], ml[key]): raise ValueError(f"GT/ML mesh or sampling mismatch: {key}")
    if not np.array_equal(tg["time_s"], tm["time_s"]): raise ValueError("GT/ML time grid mismatch")
    w, times, dt = gt["area_weights"], tg["time_s"], tg["dt_s"]
    ug, um = uniformity(gt["theta_p"], w), uniformity(ml["theta_p"], w)
    cv_error = None if ug["CV"] is None or um["CV"] is None else abs(um["CV"]-ug["CV"])
    cv_rel = 100*cv_error/ug["CV"] if cv_error is not None and ug["CV"] > 1e-14 else None
    saturation = timing_comparison(crossing(times, tg["min_theta_p"] >= SATURATION_TARGET),
                                   crossing(times, tm["min_theta_p"] >= SATURATION_TARGET))
    saturation.update(target=SATURATION_TARGET, time_resolution_bound_s=float(dt.max()))
    purge = dict(reference="case inlet precursor concentration c0", reference_mol_m3=cfg.c0,
                 threshold=PURGE_THRESHOLD, purge_start_s=cfg.pulse_duration+cfg.rise_time,
                 status="diagnostic only, no optimization")
    for key in ("outlet_cp_mol_m3", "chamber_mean_cp_mol_m3"):
        a, b = [crossing(times, tr[key]/cfg.c0 < PURGE_THRESHOLD,
                        cfg.pulse_duration+cfg.rise_time, True) for tr in (tg, tm)]
        entry = timing_comparison(a, b)
        for result in (a, b):
            result["time_since_purge_start_s"] = max(0., result["time_s"]-purge["purge_start_s"]) if result["time_s"] is not None else None
        entry["GT_final_normalized_residual"] = float(tg[key][-1]/cfg.c0)
        entry["ML_final_normalized_residual"] = float(tm[key][-1]/cfg.c0)
        purge[key] = entry
    coverage = read_json(ml_folder/"state_coverage.json")
    fraction = coverage["fraction_outside_training_envelope"]
    if fraction is None:
        support_interpretation = "Unavailable: cannot classify local-state interpolation/extrapolation without verified 2D predicted fields."
    else:
        support_interpretation = f"Unseen geometry; {100*fraction:.4g}% of accepted local states outside the frozen 2D training-condition min/max envelope. An axis-aligned box alone does not establish joint 4D interpolation."
    result = dict(condition=dict(pressure_mTorr=cfg.precursor_pressure/TORR*1000, pulse_equivalent_s=cfg.pulse_duration,
        purge_s=cfg.purge_duration, final_time_s=float(times[-1])), quick=quick,
        primary_metric_definition="final-time wafer-area-weighted relative L2 (percent); lumped P1 boundary quadrature",
        theta_p=field_metrics(ml["theta_p"], gt["theta_p"], w),
        theta_b=field_metrics(ml["theta_b"], gt["theta_b"], w),
        uniformity=dict(competitive_gt=ug, frozen_ml=um, CV_absolute_error=cv_error, CV_relative_error_pct=cv_rel),
        surface_transient_relative_L2_pct={k: relative_l2(tm[k], tg[k], dt) for k in
            ("mean_theta_p", "min_theta_p", "mean_theta_b", "max_theta_b")},
        outlet_relative_L2_pct={k: relative_l2(tm[k], tg[k], dt) for k in ("outlet_cp_mol_m3", "outlet_cb_mol_m3")},
        saturation=saturation, purge=purge, state_coverage=coverage, state_support_interpretation=support_interpretation,
        competitive_gt=read_json(gt_folder/"diagnostics.json"), frozen_ml=read_json(ml_folder/"diagnostics.json"),
        freeze=freeze, weights_changed=freeze["weights_changed"],
        flow_interpretation_qualified=flow["interpretation_qualified_by_flow"],
        figures=[])
    error = result["theta_p"]["relative_L2_pct"]
    if quick: judgement = "SMOKE TEST ONLY: shortened dose and coarse mesh; no transfer accuracy conclusion."
    elif error is None: judgement = "Undefined relative error: GT norm vanishes; inspect absolute errors."
    elif error < 1: judgement = "Strong agreement for the tested discrete geometry (<1% theta_p final relative L2)."
    elif error <= 5: judgement = "Meaningful agreement for the tested discrete geometry (1-5% theta_p final relative L2)."
    else: judgement = "Transfer degradation (>5% theta_p final relative L2); no fine-tuning performed."
    if not quick:
        judgement += " Mesh/time convergence is not established by this run."
        if flow["interpretation_qualified_by_flow"]: judgement += " Flow divergence warning limits physical interpretation."
        if fraction is None: judgement += " Local-state support remains unclassified."
    result["technical_judgement"] = judgement
    result["diagnostic_causes"] = dict(
        A_flow="warning" if flow["interpretation_qualified_by_flow"] else "reported sanity checks passed",
        B_mesh="coarse smoke mesh" if quick else "single-resolution calculation; convergence unverified",
        C_local_extrapolation=support_interpretation,
        D_geometry_dependence="Cannot isolate from a single geometry/resolution; do not infer microscopic chemistry",
        E_coupling="identical conservative gas/surface structure; algebra self-test and numerical balance required",
        F_checkpoint="strict architecture, normalization, output parity and freeze hashes checked")
    return result


def show_number(x, digits=4): return "N/A" if x is None else f"{x:.{digits}g}"


def make_figures(folder, gt_folder, ml_folder, metrics, reference):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from matplotlib.colors import Normalize
    folder.mkdir(parents=True, exist_ok=True)
    gt, ml = load_wafer(gt_folder), load_wafer(ml_folder)
    tg, tm = load_trace(gt_folder), load_trace(ml_folder)
    tri = Triangulation(gt["coordinates"][:, 0]*1000, gt["coordinates"][:, 1]*1000, gt["triangles"])
    title = "2D-trained hidden-state dynamics → frozen 3D wafer transfer"
    if metrics["quick"]: title += "\nQUICK: short coarse smoke test, not research accuracy"
    def row(fig, axes, variable, letters):
        a, b = gt[variable], ml[variable]
        low, high = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
        norm = Normalize(low, max(high, low+1e-12))
        error = abs(b-a); errnorm = Normalize(0, max(float(error.max()), 1e-15))
        names = ["Competitive GT", "Frozen ML" if variable == "theta_p" else "Predicted hidden state", "Absolute error"]
        for ax, val, label, letter, color in zip(axes, (a, b, error), names, letters, (norm, norm, errnorm)):
            artist = ax.tripcolor(tri, val, shading="gouraud", norm=color,
                                  cmap="viridis" if label != "Absolute error" else "magma")
            ax.set_title(f"{letter}. {label}\n{variable}", fontsize=10)
            ax.set(xlabel="x [mm]", ylabel="y [mm]", aspect="equal")
            fig.colorbar(artist, ax=ax, shrink=.8)
    def save(fig, name):
        fig.savefig(folder/name, dpi=180, bbox_inches="tight"); plt.close(fig)
        return str((folder/name).resolve())
    outputs = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.8), layout="constrained")
    row(fig, axes[0], "theta_p", "ABC"); row(fig, axes[1], "theta_b", "DEF")
    fraction = metrics["state_coverage"]["fraction_outside_training_envelope"]
    outside_text = "N/A" if fraction is None else f"{100*fraction:.4g}%"
    annotation = (f"Final wafer L2: θp {show_number(metrics['theta_p']['relative_L2_pct'])}%  |  "
        f"θb {show_number(metrics['theta_b']['relative_L2_pct'])}%  |  "
        f"Uniformity |ΔCV|: {show_number(metrics['uniformity']['CV_absolute_error'])}\n"
        f"weights_changed = {str(metrics['weights_changed']).lower()}  |  "
        f"3D states outside 2D envelope: {outside_text}  |  "
        f"t = {metrics['condition']['final_time_s']:.3f} s")
    fig.suptitle(title+"\n"+annotation, fontsize=12)
    outputs.append(save(fig, "00_main_result.png"))
    for variable, name in (("theta_p", "01_theta_p_maps.png"), ("theta_b", "02_theta_b_maps.png")):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
        row(fig, axes, variable, "ABC"); fig.suptitle(title, fontsize=12); outputs.append(save(fig, name))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, key in zip(axes.flat, ("mean_theta_p", "min_theta_p", "mean_theta_b", "max_theta_b")):
        ax.plot(tg["time_s"], tg[key], label="Competitive GT", color="black")
        ax.plot(tm["time_s"], tm[key], "--", label="Frozen ML", color="tab:orange")
        ax.set(xlabel="Time [s]", ylabel=key, title=key.replace("_", " ")); ax.grid(alpha=.2); ax.legend()
    if not metrics["quick"]: axes[0, 1].axhline(SATURATION_TARGET, color="gray", lw=.7, ls=":")
    fig.suptitle(title, fontsize=12); outputs.append(save(fig, "03_surface_transients.png"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for ax, key in zip(axes, ("outlet_cp_mol_m3", "outlet_cb_mol_m3")):
        ax.plot(tg["time_s"], tg[key], label="Competitive GT", color="black")
        ax.plot(tm["time_s"], tm[key], "--", label="Frozen ML", color="tab:orange")
        ax.set(xlabel="Time [s]", ylabel="Flux-weighted concentration [mol/m³]", title=key)
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle(title, fontsize=12); outputs.append(save(fig, "04_outlet_transients.png"))
    with np.load(ml_folder/"local_state_samples.npz", allow_pickle=False) as data: sample = data["local_states"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for i, ax in enumerate(axes.flat):
        if reference is not None:
            ax.hist(reference["samples"][:, i], bins=45, density=True, histtype="step", lw=1.6, label="2D frozen training-condition states")
            ax.axvline(reference["minimum"][i], color="black", lw=.7, ls=":")
            ax.axvline(reference["maximum"][i], color="black", lw=.7, ls=":")
        ax.hist(sample[:, i], bins=45, density=True, histtype="step", lw=1.4, label="3D frozen ML")
        ax.set(xlabel=INPUTS[i], ylabel="Sample density"); ax.legend(fontsize=8)
    message = "2D reference unavailable; outside fraction is N/A" if reference is None else f"Outside 2D min/max envelope: {100*fraction:.4g}% (not a joint-support proof)"
    fig.suptitle(title+"\n"+message, fontsize=11)
    outputs.append(save(fig, "05_state_space_coverage.png"))
    return outputs


def write_summary(out, manifest, all_metrics):
    audit = manifest["checkpoint_audit"]
    c = manifest["physical_config"]
    lines = ["# 2D-trained hidden-state dynamics → frozen 3D wafer transfer", "",
        f"Status: {manifest['status']}", "",
        f"Checkpoint: `{audit['checkpoint_path']}`",
        f"Checkpoint SHA256: `{audit['checkpoint_sha256']}`",
        f"Network parameters before: `{audit['parameter_sha256_before']}`",
        f"Network parameters after: `{audit.get('parameter_sha256_after', 'not executed')}`",
        f"Normalization: {audit['normalization_source']}; c_ref={audit['metadata']['specification']['concentration_ref_mol_m3']:.16g} mol/m³; time_ref={audit['metadata']['specification']['time_ref_s']} s.",
        f"Training manifest: {manifest['training_manifest']['status']}. {manifest['training_manifest'].get('limitation', '')}", "",
        "No retraining, optimizer, backpropagation, coordinate inputs, normalization refit, checkpoint selection, or GT-hidden-state input.",
        "ML uses predicted z only. GT and ML compute independently; evaluation reads both trees only after both complete.",
        "Initial cp=cb=theta_p=z=0, identical to supplied 2D training code.", "",
        f"Geometry: chamber radius {c['chamber_radius']:.5g} m, height {c['height']:.5g} m; wafer radius {c['wafer_radius']:.5g} m; bottom ports radius {c['port_radius']:.5g} m at x=±{c['port_offset']:.5g} m (Stage 7 construction).",
        "Geometry-only denotes the same pressure/pulse condition; the flow changes from the 2D prescribed plug field to the Stage 7 3D 300 sccm flow. It is not an identical-velocity-field experiment.",
        "Inherited assumptions include site area 24 Å² (unverified interpretation), methane-like byproduct and nitrogen carrier. They were not adjusted using 3D GT.", ""]
    for name, result in all_metrics.items():
        if not isinstance(result, dict) or "theta_p" not in result: continue
        mesh, flow = result["mesh"], result["flow"]
        condition = result["condition"]
        lines += [f"## {name}", "", result["technical_judgement"], "",
            f"Condition: {condition['pressure_mTorr']:.5g} mTorr / {condition['pulse_equivalent_s']:.5g} s; simulated through {condition['final_time_s']:.5g} s.",
            f"Mesh: {mesh['cell_count']} tetrahedra; {mesh['gas_dof_count']} gas DOFs; {mesh['wafer_dof_count']} wafer DOFs.",
            f"Flow: volume mean speed {flow['mean_velocity_m_s']:.6g} m/s; maximum sampled speed {flow['max_velocity_m_s']:.6g} m/s; wafer mean horizontal speed {flow['wafer_region_mean_horizontal_velocity_m_s']:.6g} m/s.",
            f"Flow: scaled divergence {flow['divergence_scaled_by_H_over_u_rms']:.6g}; Qin={flow['flow_in']:.8g}, Qout={flow['flow_out']:.8g} m³/s; relative mismatch={flow['flow_balance_relative']:.4g}.",
            f"Runtime: GT {result['competitive_gt']['runtime_s']:.2f} s; ML {result['frozen_ml']['runtime_s']:.2f} s; mesh/flow setup {result['setup_runtime_s']:.2f} s.", "",
            "| Final wafer quantity | Relative L2 (%) | Area-weighted MAE | Max abs error |",
            "|---|---:|---:|---:|"]
        for field in ("theta_p", "theta_b"):
            a = result[field]; lines.append(f"| {field} | {show_number(a['relative_L2_pct'])} | {a['MAE']:.6g} | {a['maximum_absolute_error']:.6g} |")
        lines += ["", "| Uniformity | GT | Frozen ML |", "|---|---:|---:|"]
        for key in ("mean", "minimum", "maximum", "std", "CV"):
            lines.append(f"| {key} | {show_number(result['uniformity']['competitive_gt'][key], 7)} | {show_number(result['uniformity']['frozen_ml'][key], 7)} |")
        sat = result["saturation"]
        lines += ["", f"Uniformity |ΔCV|={show_number(result['uniformity']['CV_absolute_error'])}; relative CV error={show_number(result['uniformity']['CV_relative_error_pct'])}%.",
            f"Saturation (wafer minimum >=0.99): GT {sat['competitive_gt']['status']} ({show_number(sat['competitive_gt']['time_s'])} s); ML {sat['frozen_ml']['status']} ({show_number(sat['frozen_ml']['time_s'])} s); timing error={show_number(sat['relative_error_pct'])}%.",
            f"Outlet relative L2: precursor {show_number(result['outlet_relative_L2_pct']['outlet_cp_mol_m3'])}%; byproduct {show_number(result['outlet_relative_L2_pct']['outlet_cb_mol_m3'])}%.",
            result["state_support_interpretation"], f"weights_changed = {str(result['weights_changed']).lower()}", ""]
        for model_name in ("competitive_gt", "frozen_ml"):
            d = result[model_name]
            lines.append(f"{model_name}: precursor balance / total injection={d['precursor_relative']:.4e}; byproduct balance / generated={d['byproduct_relative_to_generated']:.4e}; max(theta_p+theta_b)={d['maximum_total_occupancy']:.12g}; max Newton iterations={d['max_newton_iterations']}; max residual={d['maximum_residual']:.4e}.")
        lines += ["", f"Purge diagnostic: cp/c0 < {PURGE_THRESHOLD:g}, sustained through end; see metrics.json for outlet and chamber mean crossing times. Threshold is illustrative and predeclared; no process optimization.",
                  "", "Figures:", ""]+[f"- `{Path(f).relative_to(out)}`" for f in result["figures"]]+[""]
    lines += ["## Interpretation limits", "",
        "If observed in a completed reactor run, agreement would support transfer of effective local hidden-state dynamics for the tested discrete problem. It would not establish the true microscopic chemistry, universal geometry independence, or mesh-converged physical accuracy.",
        "A shared flow/discretization error may affect GT and ML similarly. Report flow and local-state coverage together with prediction errors.",
        "The original training prediction exports describe the final frozen checkpoint, not every optimizer iterate. The support test uses only the four training conditions.", ""]
    if manifest.get("strong_status"): lines += ["Strong condition: "+manifest["strong_status"], ""]
    if "3d_not_executed" in manifest["status"]:
        lines[4:4] = ["**No 3D reactor simulation has been executed. Only checkpoint and manufactured-algebra verification is available.**", ""]
    (out/"SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


# ======================= VERIFICATION / ENTRY POINT =======================
def manufactured_tetrahedron():
    """One real P1 tetrahedral element, NOT the wafer reactor or ALD GT data."""
    x = np.array([[0., 0., 0.], [.02, 0., 0.], [0., .02, 0.], [0., 0., .02]])
    volume = abs(np.linalg.det((x[1:]-x[0]).T))/6
    gradients = np.linalg.inv(np.column_stack([np.ones(4), x]))[1:, :].T
    velocity = np.array([.05, 0., 0.]); flux = .05*.02**2/2
    out = np.array([1, 2, 3]); inc = np.array([0, 2, 3])
    advection = -volume/4*np.outer(gradients@velocity, np.ones(4))
    advection[np.ix_(out, out)] += flux/12*(np.ones((3, 3))+np.eye(3))
    stiffness = volume*gradients@gradients.T
    B, _ = graph_viscosity(sparse.csr_matrix(advection+.01*stiffness))
    Bb, _ = graph_viscosity(sparse.csr_matrix(advection+.05*stiffness))
    inlet, outlet, surface = np.zeros(4), np.zeros(4), np.zeros(4)
    inlet[inc] = flux/3; outlet[out] = flux/3; surface[:3] = .02**2/6
    return Operators(B, Bb, np.full(4, volume/4), surface, surface.copy(), inlet, outlet, x)


def self_test(adapter, cfg, model=None):
    checks = []
    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))
        if not passed: raise AssertionError(f"{name}: {details}")
    before = arrays_hash(adapter.state)
    rng = np.random.default_rng(43)
    x = rng.uniform(.01, .6, (48, 4)); x[:, 2:] *= .5
    q, tangent = adapter.value_tangent(x); eps = 1e-6
    finite = np.empty_like(tangent)
    for j in range(4):
        dx = np.zeros_like(x); dx[:, j] = eps
        finite[:, :, j] = (adapter.value(x+dx)-adapter.value(x-dx))/(2*eps)
    error = float(np.linalg.norm(tangent-finite)/max(np.linalg.norm(finite), 1e-30))
    check("actual_checkpoint_four_input_analytic_tangent", error < 2e-7, relative_L2=error)
    check("actual_checkpoint_positive_outputs", np.all(q > 0) and np.all(np.isfinite(q)))
    if model is not None:
        check("torch_numpy_inference_parity", True, maximum_absolute_difference=parity(model, adapter, x))
    ops = manufactured_tetrahedron()
    check("tetrahedral_conservative_column_sums", all(np.max(abs(np.asarray(B.sum(axis=0)).ravel()-ops.outlet)) < 1e-15 for B in (ops.B, ops.Bb)))
    cfg = replace(cfg, pulse_duration=.08, rise_time=.02, purge_duration=.10, dt=.01)
    kp = cfg.beta0*thermal_speed(cfg.precursor_molar_mass, cfg.temperature)/4
    kb = cfg.beta_byproduct*thermal_speed(cfg.byproduct_molar_mass, cfg.temperature)/4
    balance_reports = {}
    for name, law in (("frozen_ml", SurfaceLaw(kp, adapter.ref, cfg.gamma, adapter=adapter, time_ref=adapter.time_ref)),
                      ("competitive_gt", SurfaceLaw(kp, adapter.ref, cfg.gamma, kb=kb))):
        solver = TransientFEM(ops, cfg, adapter.ref, law)
        u = np.r_[np.full(4, .3), np.full(4, .05), np.full(3, .2), np.full(3, .1)]
        direction = rng.normal(size=len(u)); previous = .97*u
        _, K = solver.residual_jacobian(u, previous, .03, .04)
        fd = (solver.residual_jacobian(u+eps*direction, previous, .03, .04, False)
             -solver.residual_jacobian(u-eps*direction, previous, .03, .04, False))/(2*eps)
        error = float(np.linalg.norm(K@direction-fd)/np.linalg.norm(fd))
        check(name+"_coupled_residual_jacobian", error < 1e-7, relative_L2=error)
        current = np.zeros(solver.size); incoming = outp = outb = 0.
        max_rp = max_rb = maximum_residual = 0.; maximum_occupancy = 0.
        cap = cfg.gamma*ops.surface[solver.active]; times = time_grid(cfg)
        for t0, t1 in zip(times[:-1], times[1:]):
            current, _, residual = solver.step(current, t0, t1); dt = t1-t0
            cp, cb = current[:4]*adapter.ref, current[4:8]*adapter.ref
            p, z = current[solver.ps], current[solver.zs]
            incoming += dt*ops.inlet.sum()*pulse_average(t0, t1, cfg)
            outp += dt*(cp@ops.outlet); outb += dt*(cb@ops.outlet)
            max_rp = max(max_rp, abs(cp@ops.mass+p@cap+outp-incoming))
            max_rb = max(max_rb, abs(cb@ops.mass+z@cap+outb-cfg.byproduct_yield*(p@cap)))
            maximum_residual = max(maximum_residual, residual)
            maximum_occupancy = max(maximum_occupancy, float((p+z).max()))
            check_bounds = solver.admissible(current)
            if not check_bounds: raise AssertionError("Manufactured bounds violated")
        generated = cfg.byproduct_yield*(p@cap)
        balance_reports[name] = dict(precursor_relative=float(max_rp/incoming),
            byproduct_relative_to_generated=float(max_rb/max(generated, 1e-30)),
            maximum_total_occupancy=maximum_occupancy, maximum_residual=maximum_residual)
        check(name+"_tetrahedral_mass_and_bounds", max(max_rp/incoming, max_rb/max(generated, 1e-30)) < 2e-8
              and maximum_occupancy <= 1+1e-10, **balance_reports[name])
        empty, _, _ = solver.step(np.zeros(solver.size), .4, .41)
        check(name+"_empty_purge_no_spurious_source", np.max(abs(empty)) == 0.)
        if name == "frozen_ml":
            initial = np.zeros(solver.size); initial[solver.ps] = .1; initial[solver.zs] = .3
            after, _, _ = solver.step(initial, .4, .41)
            bp = after[4:8]*adapter.ref; dz = after[solver.zs]-initial[solver.zs]
            balance = float(bp@ops.mass+dz@cap+.01*(bp@ops.outlet))
            check("removal_returns_byproduct_to_gas", np.min(dz) < 0 and bp.max() > 0 and abs(balance) < 1e-15,
                  inventory_residual_mol=balance)
    ref = dict(minimum=np.array([0., 0., 0., 0.]), maximum=np.ones(4), samples=np.eye(4))
    sampler = StateSampler(ref, capacity=3)
    sampler.update(np.array([[.5, .5, .5, .5], [1.2, .5, .5, .5], [.5, -.2, .5, .5]]), np.ones(3))
    check("coverage_known_outside_fraction", abs(sampler.report()["fraction_outside_training_envelope"]-2/3) < 1e-14)
    check("saturation_not_reached_is_not_invented", crossing(np.array([0., 1.]), np.array([False, False]))["time_s"] is None)
    check("area_weighted_error_definition", abs(field_metrics(np.array([0., 2.]), np.ones(2), np.array([1., 3.]))["relative_L2_pct"]-100) < 1e-12)
    check("actual_weights_and_normalization_unchanged", arrays_hash(adapter.state) == before)
    return dict(status="passed", real_3d_reactor_executed=False,
        scope="actual saved network + manufactured single tetrahedron P1 transport; no 3D reactor accuracy claim",
        torch_execution="passed" if model is not None else "not available; restricted archive audit and NumPy evaluation only",
        checks=checks, balance_reports=balance_reports)


def resolve_checkpoint(args):
    if args.checkpoint:
        path = Path(args.checkpoint).expanduser().resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        return path
    if args.training_output:
        path = Path(args.training_output).expanduser().resolve()/"best_model.pt"
        if not path.is_file(): raise FileNotFoundError(path)
        return path
    bases = [Path.cwd(), Path(__file__).resolve().parent]
    candidates = list(dict.fromkeys((b/rel).resolve() for b in bases
        for rel in ("outputs/09_hidden_state_feml/best_model.pt", "best_model.pt") if (b/rel).is_file()))
    if not candidates: raise FileNotFoundError("No checkpoint found. Supply --checkpoint /path/to/best_model.pt")
    if len(candidates) != 1:
        raise ValueError("Several checkpoints found; explicitly select --checkpoint. No automatic checkpoint selection.")
    return candidates[0]


def prepare_output(root, protected):
    candidate = Path(root).expanduser()/"10_3d_zeroshot"
    if candidate.is_symlink(): raise ValueError("Refusing to clean a symlink output directory")
    out = candidate.resolve()
    if out.name != "10_3d_zeroshot": raise ValueError("Unsafe output directory")
    for value in protected:
        p = Path(value).resolve()
        if p == out or out in p.parents: raise ValueError(f"Output cleanup would delete an input: {p}")
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("competitive_gt", "frozen_ml", "figures"): (out/name).mkdir()
    return out


def setup_case(cfg, out, name):
    start = time.monotonic()
    reactor = build_geometry(cfg)
    print("[FLOW] Stage 7 steady Navier-Stokes field (no velocity rescaling to 0.5 m/s)", flush=True)
    velocity, pressure, flow = solve_flow(reactor, cfg, out/f"flow_{name}.json")
    ops, triangles, mesh = assemble_operators(reactor, cfg, velocity)
    write_json(out/f"mesh_{name}.json", mesh)
    return ops, triangles, mesh, flow, time.monotonic()-start


def run_case(name, cfg, out, checkpoint, model, adapter, audit, reference, quick=False, draw=True, setup=None):
    reused_setup = setup is not None
    if setup is None: setup = setup_case(cfg, out, name)
    ops, triangles, mesh, flow, setup_runtime = setup
    times = time_grid(cfg, .10 if quick else None)
    gt_folder, ml_folder = out/"competitive_gt"/name, out/"frozen_ml"/name
    # Required order: GT, then frozen ML, then post-hoc access to either result.
    run_competitive_gt(ops, cfg, adapter.ref, times, triangles, gt_folder)
    run_frozen_ml(ops, cfg, adapter, model, times, triangles, ml_folder, reference)
    frozen = verify_freeze(checkpoint, model, adapter, audit)
    result = evaluate(gt_folder, ml_folder, cfg, frozen, flow, quick)
    result.update(mesh=mesh, flow=flow, setup_runtime_s=0. if reused_setup else setup_runtime)
    if draw:
        figdir = out/"figures" if name in ("geometry_only", "quick") else out/"figures"/name
        result["figures"] = make_figures(figdir, gt_folder, ml_folder, result, reference)
    return result, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="Exact frozen checkpoint; never automatically select among multiple files")
    parser.add_argument("--training-output", help="Completed 2D hidden-state output tree (manifest, predictions, checkpoint)")
    parser.add_argument("--training-manifest", help="Explicit original 2D manifest path")
    parser.add_argument("--output-root", default="outputs", help="Parent; writes/cleans only its 10_3d_zeroshot child")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Only 0.10 s coarse smoke test")
    mode.add_argument("--self-test", action="store_true", help="Actual checkpoint + manufactured algebra; no 3D reactor")
    parser.add_argument("--strong", action="store_true", help="Also 35 mTorr/0.6 s after qualified default test")
    args = parser.parse_args()
    if args.strong and (args.quick or args.self_test): parser.error("--strong requires the full default run")
    if importlib.util.find_spec("mpi4py") is not None:
        from mpi4py import MPI
        if MPI.COMM_WORLD.size != 1: parser.error("Use one MPI rank; aborting before output cleanup")
    checkpoint = resolve_checkpoint(args)
    root = Path(args.training_output).resolve() if args.training_output else checkpoint.parent
    # Audit BEFORE cleanup; a bad checkpoint/manifest must not erase a prior run.
    _, adapter, audit = load_checkpoint(checkpoint, offline=True)
    cfg, training_manifest, manifest_source = load_training_manifest(root, args.training_manifest, audit)
    validate_config(cfg)
    if abs(cfg.precursor_pressure/TORR*1000-50) > 1e-8 or abs(cfg.pulse_duration-.8) > 1e-12:
        raise ValueError("Default transfer condition must be trained 50 mTorr / 0.8 s")
    protected = [checkpoint, root, Path(__file__)]+([args.training_manifest] if args.training_manifest else [])
    out = prepare_output(args.output_root, protected)
    manifest = dict(status="initializing", created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=file_hash(__file__), arguments=vars(args), checkpoint_audit=audit,
        training_manifest=manifest_source, source_provenance=SOURCE_PROVENANCE,
        physical_config=asdict(cfg), input_order=INPUTS, initial_state="cp=cb=theta_p=z=0",
        original_training_metrics=dict(status="not present in selected training output tree"),
        output_directory=str(out), geometry_source="provided Stage 7 geometry and flow implementation",
        method="conservative lumped P1, symmetric graph viscosity, backward Euler, bounded Newton/SuperLU",
        mass_tolerance=MASS_TOL, flow_scaled_divergence_warning=FLOW_DIV_WARNING,
        saturation_target=SATURATION_TARGET, purge_threshold=PURGE_THRESHOLD,
        purge_reference="case inlet concentration c0, not network concentration_ref",
        no_optimizer=True, no_training=True, no_backpropagation=True, no_coordinate_inputs=True,
        no_normalization_refit=True, no_ideal_simulation=True, no_3d_gt_data_in_ml=True,
        process_transfer_gate="default full theta_p and theta_b error <5%, numerical gates pass, no flow warning, original manifest verified, 2D support available",
        api_reference="https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.fem.petsc.html",
        mesh_time_convergence_established=False)
    all_metrics = {}; model = None
    if (root/"metrics.json").is_file():
        original_metrics = read_json(root/"metrics.json")
        if all(name in original_metrics for name in TRAINING_CONDITIONS):
            manifest["original_training_metrics"] = dict(status="read for provenance only; never used in 3D inference or tuning",
                path=str(root/"metrics.json"), sha256=file_hash(root/"metrics.json"),
                heldout_report=original_metrics.get("heldout_35mTorr_0p6s", {}).get("hidden_state_model"))
    write_json(out/"manifest.json", manifest)
    try:
        missing = [n for n in ("torch", "dolfinx", "ufl", "petsc4py", "mpi4py", "gmsh") if importlib.util.find_spec(n) is None]
        manifest["missing_runtime_dependencies"] = missing
        if importlib.util.find_spec("torch") is not None:
            model, adapter, production_audit = load_checkpoint(checkpoint)
            if production_audit["state_sha256_before"] != audit["state_sha256_before"]:
                raise ValueError("Restricted audit and torch.load differ")
            audit = production_audit; manifest["checkpoint_audit"] = audit
        reference, support = load_training_support(root, training_manifest, adapter)
        manifest["training_state_support"] = support
        write_json(out/"manifest.json", manifest)
        verification = self_test(adapter, cfg, model)
        write_json(out/"verification"/"self_test.json", verification)
        audit.update(verify_freeze(checkpoint, model, adapter, audit))
        if args.self_test:
            manifest["status"] = "offline_self_test_passed_3d_not_executed" if model is None else "self_test_passed_3d_not_executed"
            all_metrics = dict(status=manifest["status"], actual_3d_run=False,
                geometry_zero_shot_success=None, theta_p_3d_relative_L2_pct=None,
                theta_b_3d_relative_L2_pct=None, freeze=audit, missing_runtime_dependencies=missing)
            write_json(out/"metrics.json", all_metrics); write_json(out/"manifest.json", manifest)
            write_summary(out, manifest, all_metrics)
            with (out/"SUMMARY.md").open("a", encoding="utf-8") as f:
                f.write("\n## Execution limitation\n\nNo 3D mesh, flow, GT or ML reactor simulation was executed. No 3D accuracy, runtime, mass-balance, uniformity or transfer success is reported.\n\n")
                f.write(f"Verified {len(verification['checks'])} checkpoint/manufactured-algebra checks. These are not reactor results.\n\n")
                f.write("Missing runtime packages: "+", ".join(missing)+".\n\n")
                f.write("Run in the existing FEniCSx/PyTorch training environment:\n\n```bash\n")
                f.write("python run_3d_zeroshot_standalone.py --training-output outputs/09_hidden_state_feml --quick\n")
                f.write("python run_3d_zeroshot_standalone.py --training-output outputs/09_hidden_state_feml\n```\n")
            print(f"{manifest['status']} -> {out/'SUMMARY.md'}", flush=True)
            return
        if missing: raise RuntimeError("3D run not executed; activate the existing training environment. Missing: "+", ".join(missing))
        require_fem()
        import dolfinx, torch
        manifest["versions"] = dict(python=sys.version.split()[0], numpy=np.__version__, torch=str(torch.__version__), dolfinx=dolfinx.__version__)
        manifest["status"] = "quick_preflight_running"; write_json(out/"manifest.json", manifest)
        quick_cfg = replace(cfg, mesh_size_3d=max(.055, cfg.mesh_size_3d), port_mesh_size=.012, dt=max(.02, cfg.dt))
        key = "quick" if args.quick else "quick_preflight"
        result, quick_setup = run_case(key, quick_cfg, out, checkpoint, model, adapter, audit, reference,
                                      quick=True, draw=args.quick)
        all_metrics[key] = result
        write_json(out/"metrics.json", all_metrics)
        # No tuning and no accuracy gate for the quick comparison. Only numerical failure blocks full simulation.
        del quick_setup
        if not args.quick:
            manifest["status"] = "full_geometry_only_running"; write_json(out/"manifest.json", manifest)
            result, setup = run_case("geometry_only", cfg, out, checkpoint, model, adapter, audit, reference)
            all_metrics["geometry_only"] = result
            write_json(out/"metrics.json", all_metrics)
            if args.strong:
                errors = [result[k]["relative_L2_pct"] for k in ("theta_p", "theta_b")]
                eligible = (all(e is not None and e < 5 for e in errors)
                    and not result["flow_interpretation_qualified"]
                    and manifest_source["status"] == "verified" and reference is not None)
                if eligible:
                    strong_cfg = replace(cfg, precursor_pressure=35e-3*TORR, pulse_duration=.6)
                    strong, _ = run_case("geometry_plus_process", strong_cfg, out, checkpoint, model, adapter,
                                         audit, reference, setup=setup)
                    strong["setup_reused_from_geometry_only"] = True
                    all_metrics["geometry_plus_process"] = strong
                    manifest["strong_status"] = "completed without retraining; 35 mTorr/0.6 s"
                else:
                    manifest["strong_status"] = "skipped: default result did not meet the predeclared qualification gate; no tuning or retries"
        audit.update(verify_freeze(checkpoint, model, adapter, audit))
        manifest["status"] = "completed_quick_not_research_result" if args.quick else "completed_3d_evaluation"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out/"metrics.json", all_metrics); write_json(out/"manifest.json", manifest)
        write_summary(out, manifest, all_metrics)
        print(f"Completed -> {out/'SUMMARY.md'}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        if "checkpoint_audit" in manifest:
            try: manifest["checkpoint_audit"].update(verify_freeze(checkpoint, model, adapter, audit))
            except Exception as freeze_error: manifest["freeze_failure"] = str(freeze_error)
        write_json(out/"manifest.json", manifest)
        write_json(out/"metrics.json", dict(status="failed", error=str(exc), completed_cases=all_metrics))
        write_json(out/"RUN_FAILED.json", dict(error=str(exc), traceback=traceback.format_exc()))
        (out/"SUMMARY.md").write_text("# 3D zero-shot run not completed\n\n"+str(exc)+"\n\nNo overall transfer success is claimed. See manifest.json, metrics.json and RUN_FAILED.json.\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
