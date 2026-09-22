from dataclasses import replace
from config import Config,arguments,TORR
from geometry.reactor_3d import build
from physics.flow import solve
from analysis.diagnostics import start_output,save_json
from workflow import run_transport,export_flow

def main():
    cfg,_=arguments("3D 300 mm wafer reactor",replace(Config(),precursor_pressure=75e-3*TORR,
                                                     pulse_duration=0.3,purge_duration=1.0))
    reactor=build(cfg); folder=start_output(cfg,"07_3d_wafer",reactor.domain.comm)
    from dolfinx import fem
    from physics.flow import integral
    from geometry.reactor_2d import INLET,OUTLET,WAFER,TOP,SIDE
    import ufl
    comm=reactor.domain.comm
    one=fem.Constant(reactor.domain,1.)
    areas={str(tag):integral(one*reactor.ds(tag),comm) for tag in (INLET,OUTLET,WAFER,TOP,SIDE)}
    cells=comm.allreduce(reactor.domain.topology.index_map(3).size_local)
    vertices=comm.allreduce(reactor.domain.topology.index_map(0).size_local)
    if comm.rank==0:
        save_json(folder/"mesh_statistics.json",{"cells":cells,"vertices":vertices,
            "boundary_areas_m2":areas,"tags":{"inlet":INLET,"outlet":OUTLET,"wafer":WAFER,"top":TOP,"side_and_inert_floor":SIDE},
            "nominal_h_m":cfg.mesh_size_3d,"port_h_m":cfg.port_mesh_size,
            "geometry":"full 50 cm diameter disk, height 2 cm; 300 mm wafer; bottom ports at x=+-0.220 m"})
    if reactor.domain.comm.rank==0:
        print("[FLOW] Solving 3D carrier field...",flush=True)
    u,p,flow=solve(reactor,cfg)
    export_flow(reactor,cfg,u,p,flow,folder)
    run_transport(reactor,cfg,u,folder)

if __name__=="__main__": main()
