import csv
import json
from pathlib import Path
import numpy as np
from config import parameter_manifest, thermal_speed


def save_json(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def write_csv(path, rows):
    if not rows:
        return
    with open(path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def front_position(x,theta,level=0.5):
    """Most downstream downward crossing. None if absent, never fake zero."""
    order=np.argsort(x); x=np.asarray(x)[order]; y=np.asarray(theta)[order]-level
    indices=np.flatnonzero((y[:-1]>=0)&(y[1:]<0))
    if not len(indices):
        return None
    i=indices[-1]
    return float(x[i]+(x[i+1]-x[i])*y[i]/(y[i]-y[i+1]))


def dimensionless(cfg,h,velocity=None):
    u=cfg.mean_velocity if velocity is None else velocity
    ks=thermal_speed(cfg.precursor_molar_mass,cfg.temperature)*cfg.beta0/4
    return {"Re_H":u*cfg.height/cfg.kinematic_viscosity,
            "Pe_L":u*cfg.length/cfg.diffusivity,
            "Pe_H":u*cfg.height/cfg.diffusivity,
            "Pe_cell_estimate":u*h/(2*cfg.diffusivity),
            "Da_wall_diffusion":ks*cfg.height/cfg.diffusivity,
            "Da_wall_advection":ks*cfg.length/(u*cfg.height),
            "Kn_H_estimate":50e-6*133.322368/cfg.pressure/cfg.height,
            "residence_time_s":cfg.length/u,
            "local_adsorption_time_s":cfg.gamma/(ks*cfg.c0)}


def start_output(cfg,stage,comm):
    folder=Path(cfg.output_root)/stage
    if comm.rank==0:
        folder.mkdir(parents=True,exist_ok=True)
        save_json(folder/"parameters.json",parameter_manifest(cfg))
    comm.barrier()
    return folder
