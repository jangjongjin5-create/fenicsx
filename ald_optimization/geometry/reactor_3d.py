"""50 cm disk, central 300 mm wafer and two bottom circular ports.

Port center locations and orientation are an explicit reconstruction assumption.
All non-port/non-wafer surfaces are inert in the baseline configuration.
"""
import numpy as np
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
from .reactor_2d import Reactor, INLET, OUTLET, WAFER, TOP, SIDE


def build(cfg, comm=MPI.COMM_WORLD):
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
