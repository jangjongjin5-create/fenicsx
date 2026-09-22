from dataclasses import replace
from config import Config,arguments,TORR
from geometry.reactor_2d import build
from physics.flow import prescribed
from analysis.diagnostics import start_output,write_csv
from plotting.plots import plt,finish,wafer_profile
from workflow import run_transport

def main():
    cfg,_=arguments("Ideal, two-pathway and site-blocking comparison",replace(Config(),
                     precursor_pressure=50e-3*TORR,pulse_duration=0.8,purge_duration=1.0))
    results={}; records=[]
    for model in ("ideal","soft_saturation","competitive_adsorption"):
        case=replace(cfg,kinetics_model=model)
        reactor=build(case); folder=start_output(case,f"05_nonideal/{model}",reactor.domain.comm)
        history,snaps,meta=run_transport(reactor,case,prescribed(reactor,case),folder)
        if reactor.domain.comm.rank==0:
            results[model]=(history,snaps)
            records.append({"model":model,**history[-1]})
    if reactor.domain.comm.rank==0:
        folder=folder.parent; write_csv(folder/"model_comparison.csv",records)
        fig,ax=plt.subplots(2,2,figsize=(12,8))
        for model,(history,snaps) in results.items():
            x,y=wafer_profile(snaps[-1]); ax[0,0].plot(x,y,label=model)
            t=[r["time_s"] for r in history]
            ax[0,1].plot(t,[r["theta_mean"] for r in history],label=model)
            ax[1,0].plot(t,[r["outlet_precursor_c_mol_m3"]/cfg.c0 for r in history],label=model)
            ax[1,1].plot(t,[r["outlet_byproduct_c_mol_m3"]/cfg.c0 for r in history],label=model)
        ax[0,0].set(xlabel="Wafer x [m]",ylabel="Final precursor coverage")
        ax[0,1].set(xlabel="Time [s]",ylabel="Mean coverage / apparent saturation")
        ax[1,0].set(xlabel="Time [s]",ylabel="Outlet precursor c/c0")
        ax[1,1].set(xlabel="Time [s]",ylabel="Outlet byproduct c/c0")
        for a in ax.flat: a.legend(fontsize=8)
        finish(fig,folder/"ideal_vs_nonideal.png")

if __name__=="__main__": main()
