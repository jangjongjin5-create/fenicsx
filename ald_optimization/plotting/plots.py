"""Headless Matplotlib only. No fields are clipped in plotting."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":110,"savefig.dpi":180,"axes.grid":False})


def finish(fig,path):
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(path,bbox_inches="tight")
    plt.close(fig)


def wafer_profile(snapshot):
    mask=snapshot["wafer_weight"]>0
    x=snapshot["coordinates"][mask,0]
    y=snapshot["theta"][mask]
    order=np.argsort(x)
    return x[order],y[order]


def transport_figures(folder,steps,snapshots,cfg,triangles=None):
    t=np.array([r["time_s"] for r in steps])
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for name in ("min","mean","max"):
        ax[0].plot(t,[r[f"theta_{name}"] for r in steps],label=name)
    ax[0].set(xlabel="Time [s]",ylabel="Wafer coverage",ylim=(-0.01,1.02)); ax[0].legend()
    ax[1].plot(t,[r["balance_relative"] for r in steps],label="Precursor")
    ax[1].plot(t,[r["byproduct_balance_relative"] for r in steps],label="Byproduct")
    ax[1].set(xlabel="Time [s]",ylabel="Cumulative relative balance residual")
    ax[1].ticklabel_format(axis="y",style="sci",scilimits=(0,0)); ax[1].legend()
    finish(fig,folder/"coverage_and_mass_balance.png")
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].plot(t,[r["coupling_iterations"] for r in steps],label="Coupling iterations")
    ax[0].set(xlabel="Time [s]",ylabel="Iterations")
    ax[1].plot(t,[r["c_min_mol_m3"]/cfg.c0 for r in steps],label="Min c/c0")
    ax[1].plot(t,[r["byproduct_min_mol_m3"]/cfg.c0 for r in steps],label="Min byproduct/c0")
    ax[1].set(xlabel="Time [s]",ylabel="Raw gas minimum (no clipping)"); ax[1].legend()
    finish(fig,folder/"numerical_diagnostics.png")
    selected=[s for s in snapshots if s["time"]>0]
    if len(selected)>6:
        selected=[selected[i] for i in np.linspace(0,len(selected)-1,6,dtype=int)]
    if triangles is not None:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for s in selected:
            x,y=wafer_profile(s); ax.plot(x,y,label=f'{s["time"]:.2f} s')
        ax.set(xlabel="Wafer position x [m]",ylabel=r"Coverage $\Theta$",ylim=(-0.01,1.02))
        ax.legend(ncol=3); finish(fig,folder/"wafer_coverage_profiles.png")
        fig,axes=plt.subplots(len(selected),1,figsize=(10,2.1*len(selected)),squeeze=False)
        coords=selected[0]["coordinates"]
        tri=mtri.Triangulation(coords[:,0],coords[:,1],triangles)
        vmax=max(cfg.c0,max(float(s["concentration"].max()) for s in selected))
        vmin=min(0.,min(float(s["concentration"].min()) for s in selected))
        for ax,s in zip(axes[:,0],selected):
            artist=ax.tricontourf(tri,s["concentration"],levels=np.linspace(vmin,vmax,40),cmap="Blues")
            ax.plot([cfg.wafer_start,cfg.wafer_end],[0,0],lw=4,color="#ce8627")
            ax.set(xlabel="x [m]",ylabel="y [m]",title=f'Precursor, t = {s["time"]:.2f} s')
            fig.colorbar(artist,ax=ax,label="mol / m³")
        finish(fig,folder/"concentration_fields.png")
    else:
        fig,axes=plt.subplots(2,3,figsize=(11,7.6),squeeze=False,layout="constrained")
        for ax,s in zip(axes.flat,selected):
            xy=s["coordinates"][:,:2]
            tri=mtri.Triangulation(xy[:,0]*1000,xy[:,1]*1000,s["wafer_triangles"])
            artist=ax.tricontourf(tri,s["theta"],
                                  levels=np.linspace(0,1,31),cmap="Blues",extend="both")
            radius=cfg.wafer_radius*1000
            ax.set(aspect="equal",xlabel="x [mm]",ylabel="y [mm]",title=f't = {s["time"]:.2f} s',
                   xlim=(-radius,radius),ylim=(-radius,radius),xticks=[-radius,0,radius],yticks=[-radius,0,radius])
        for ax in axes.flat[len(selected):]:
            ax.set_visible(False)
        fig.colorbar(artist,ax=list(axes.flat),label=r"Surface coverage $\Theta$",ticks=[0,.25,.5,.75,1],fraction=.03,pad=.03)
        finish(fig,folder/"wafer_coverage_maps.png")
        final=selected[-1]
        if "slice_c" in final:
            fig,ax=plt.subplots(figsize=(7,6))
            field=np.ma.masked_invalid(final["slice_c"])
            im=ax.pcolormesh(final["slice_x"],final["slice_y"],field,shading="auto",cmap="Blues")
            ax.set(aspect="equal",xlabel="x [m]",ylabel="y [m]",title="Mid-height precursor slice")
            fig.colorbar(im,ax=ax,label="mol / m³")
            finish(fig,folder/"concentration_midplane.png")


def sensor_figures(folder,steps,cfg):
    t=[r["time_s"] for r in steps]
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for key,label in [("outlet_precursor_c_mol_m3","Precursor"),("outlet_byproduct_c_mol_m3","Byproduct")]:
        ax[0].plot(t,[r[key]/cfg.c0 for r in steps],label=label)
    ax[0].axvline(cfg.pulse_duration+cfg.rise_time,ls="--",color="0.5",label="Purge begins")
    ax[0].set(xlabel="Time [s]",ylabel="Outlet flow-weighted c / c0"); ax[0].legend()
    for i in range(1,6):
        ax[1].plot(t,[np.nan if r[f"qcm_{i}_coverage"] is None else r[f"qcm_{i}_coverage"] for r in steps],label=f"QCM {i}")
    ax[1].set(xlabel="Time [s]",ylabel="Local normalized QCM-like mass",ylim=(-0.01,1.02))
    ax[1].legend(ncol=2); finish(fig,folder/"virtual_sensor_signals.png")


def flow_figures(folder,grid,values,dim):
    X,Y=grid; u,p=values
    speed=np.sqrt(np.sum(u*u,axis=-1))
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    im=ax[0].pcolormesh(X,Y,np.ma.masked_invalid(speed),shading="auto",cmap="Blues")
    ax[0].streamplot(X[0],Y[:,0],np.ma.masked_invalid(u[:,:,0]),
                     np.ma.masked_invalid(u[:,:,1]),color="#53606b",density=1.0,linewidth=0.6)
    fig.colorbar(im,ax=ax[0],label="Velocity magnitude [m/s]")
    im=ax[1].pcolormesh(X,Y,np.ma.masked_invalid(p),shading="auto",cmap="coolwarm")
    fig.colorbar(im,ax=ax[1],label="Mechanical gauge pressure [Pa]")
    for a,title in zip(ax,["Carrier flow","Pressure"]):
        a.set(xlabel="x [m]",ylabel="y [m]",title=title)
        if dim==3:
            a.set_aspect("equal")
    finish(fig,folder/"carrier_flow_and_pressure.png")
