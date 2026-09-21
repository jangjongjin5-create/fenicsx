"""Shared stage runners; physics remains in physics/ modules."""
from pathlib import Path
import numpy as np
from mpi4py import MPI
from config import TORR
from analysis.diagnostics import start_output,save_json,write_csv,dimensionless,front_position
from analysis.sampling import triangles,wafer_triangles,grid,sample
from physics.transport import Transport
from plotting.plots import transport_figures,sensor_figures,flow_figures,wafer_profile


def export_flow(reactor,cfg,velocity,pressure,data,folder):
    X,Y,points=grid(cfg,reactor.domain.geometry.dim)
    u=sample(velocity,points); p=sample(pressure,points)
    if reactor.domain.comm.rank==0:
        save_json(folder/"flow_summary.json",data)
        write_csv(folder/"flow_convergence.csv",data["iterations"])
        flow_figures(folder,(X,Y),(u.reshape(*X.shape,-1),p.reshape(X.shape)),reactor.domain.geometry.dim)
        rows=[]
        for xyz,uv,pv in zip(points,u,p[:,0]):
            if np.isfinite(pv):
                row={"x_m":xyz[0],"y_m":xyz[1],"z_m":xyz[2],"p_Pa":pv}
                row.update({f"u{j}_m_s":float(v) for j,v in enumerate(uv)})
                rows.append(row)
        write_csv(folder/"flow_slice.csv",rows)


def run_transport(reactor,cfg,velocity,folder,diffusion_tensor=None,times=None,make_plots=True):
    comm=reactor.domain.comm
    s=Transport(reactor,cfg,velocity,diffusion_tensor)
    cells=comm.allreduce(reactor.domain.topology.index_map(reactor.domain.topology.dim).size_local)
    if comm.rank==0:
        print(f"[MODEL]\n{cfg.kinetics_model} | {reactor.description}\n"
              f"[CONDITIONS]\nT={cfg.temperature:g} K | P={cfg.pressure:g} Pa | "
              f"precursor={cfg.precursor_pressure/TORR*1000:g} mTorr\n"
              f"D={cfg.diffusivity:g} m²/s | beta0={cfg.beta0:g} | "
              f"dose={cfg.pulse_duration:g} s | s0={cfg.site_area:g} m² ({cfg.site_area_mode})\n"
              f"[NUMERICS]\nCells={cells} | dt_max={cfg.dt:g} s | MPI={comm.size}",flush=True)
    steps,snapshots=s.run(times)
    tris=triangles(s)
    if reactor.domain.geometry.dim==3:
        surface_triangles=wafer_triangles(s)
        X,Y,points=grid(cfg,3)
        values=sample(s.c,points)
        if comm.rank==0:
            for snapshot in snapshots:
                snapshot["wafer_triangles"]=surface_triangles
            snapshots[-1].update(slice_x=X,slice_y=Y,slice_c=values.reshape(X.shape))
    meta={"cells":cells,"dofs":s.imap.size_global,"mpi_size":comm.size,
          "accepted_steps":len(steps),"rejected_steps":s.rejected,
          "graph_viscosity_edge_sum_precursor":s.graph_viscosity[0],
          "graph_viscosity_edge_sum_byproduct":s.graph_viscosity[1],
          "dimensionless_estimates":dimensionless(cfg,cfg.length/cfg.nx if reactor.domain.geometry.dim==2 else cfg.mesh_size_3d),
          "diffusion_tensor":diffusion_tensor,"inlet_condition":cfg.inlet_condition,
          "mass_units":"mol per metre depth" if reactor.domain.geometry.dim==2 else "mol",
          "final":steps[-1]}
    if comm.rank==0:
        write_csv(folder/"time_history.csv",steps)
        save_json(folder/"summary.json",meta)
        rows=[]; fronts=[]
        for sn in snapshots:
            mask=sn["wafer_weight"]>0
            for xyz,theta,blocked,weight in zip(sn["coordinates"][mask],sn["theta"][mask],
                                              sn["blocked"][mask],sn["wafer_weight"][mask]):
                rows.append({"time_s":sn["time"],"x_m":xyz[0],"y_m":xyz[1],"z_m":xyz[2],
                             "theta":theta,"theta_byproduct":blocked,"surface_weight":weight})
            if reactor.domain.geometry.dim==2:
                x,theta=wafer_profile(sn)
                fronts.append({"time_s":sn["time"],"x50_m":front_position(x,theta)})
        write_csv(folder/"wafer_profiles.csv",rows)
        if fronts:
            write_csv(folder/"saturation_front.csv",fronts)
            from plotting.plots import plt,finish
            fig,ax=plt.subplots(figsize=(7,4))
            ax.plot([r["time_s"] for r in fronts],[np.nan if r["x50_m"] is None else r["x50_m"] for r in fronts],"o-")
            ax.set(xlabel="Time [s]",ylabel="x50 [m]",title="Missing values: no 50% crossing within wafer")
            finish(fig,folder/"saturation_front.png")
        final=snapshots[-1]
        np.savez_compressed(folder/"final_state.npz",**final)
        if make_plots:
            transport_figures(folder,steps,snapshots,cfg,tris)
            sensor_figures(folder,steps,cfg)
        f=steps[-1]
        print(f"[RESULT]\ntheta_min/mean/max = {f['theta_min']:.6f} / {f['theta_mean']:.6f} / "
              f"{f['theta_max']:.6f}\nmass_balance_error = {f['balance_relative']:.3e}\n"
              f"accepted/rejected steps = {len(steps)}/{s.rejected}\noutput = {folder}\n",flush=True)
    s.close()
    return steps,snapshots,meta
