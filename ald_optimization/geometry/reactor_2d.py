"""Planar surrogate. One metre of out-of-plane depth is understood."""
from dataclasses import dataclass
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh

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


def build(cfg, benchmark=False, comm=MPI.COMM_WORLD):
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
