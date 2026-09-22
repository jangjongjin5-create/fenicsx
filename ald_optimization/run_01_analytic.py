from dataclasses import replace
import numpy as np
from config import Config,arguments,parameter_manifest
from analysis.analytical_solution import final_coverage,transient_coverage,scales
from analysis.diagnostics import save_json,write_csv
from plotting.plots import plt,finish
from pathlib import Path


def main():
    cfg,_=arguments("Paper plug-flow analytical benchmark",replace(Config(),mean_velocity=1.0))
    folder=Path(cfg.output_root)/"01_analytic"; folder.mkdir(parents=True,exist_ok=True)
    x=np.linspace(0,0.4,801); vs=0.025/2  # paper tube R/2
    rows=[]
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for t in (0.05,0.1,0.2,0.3,0.5):
        final=final_coverage(x,t,cfg,vs)
        transient=transient_coverage(x,t,replace(cfg,pulse_duration=0.5,rise_time=0),vs)
        ax[0].plot(x,final,label=f"dose {t:g} s")
        ax[1].plot(x,transient,label=f"time {t:g} s")
        for z,a,b in zip(x,final,transient):
            rows.append({"exposure_or_time_s":t,"x_m":z,"final_coverage_eq24":a,"transient_coverage_eq28":b})
    for a,title in zip(ax,["Eq.24: final profile after dose arrival","Eq.28: transient, includes travel delay"]):
        a.set(xlabel="Axial position [m]",ylabel="Coverage",title=title,ylim=(-0.01,1.02)); a.legend()
    finish(fig,folder/"analytical_profiles.png")
    fig,ax=plt.subplots(figsize=(8,4))
    for area,label in [(24e-20,"Assumed 24 Å²"),(24e-18,"Literal table: 24 nm²")]:
        a=replace(cfg,site_area=area)
        ax.plot(x,final_coverage(x,0.05,a,vs),label=label)
    ax.set(xlabel="Axial position [m]",ylabel="Coverage",title="Site-area ambiguity: identical 0.05 s dose")
    ax.legend(); finish(fig,folder/"site_area_sensitivity.png")
    tau,length=scales(cfg,vs)
    write_csv(folder/"analytical_profiles.csv",rows)
    save_json(folder/"parameters.json",parameter_manifest(cfg))
    save_json(folder/"characteristic_scales.json",{"tau_s":tau,"zbar_m":length,"V_over_S_m":vs,
        "u_m_s":cfg.mean_velocity,"geometry":"tube radius 0.025 m; upstream taper omitted"})
    print(f"[MODEL]\nPaper analytical plug flow\n[CONDITIONS]\ns0={cfg.site_area:g} m² ({cfg.site_area_mode})\n"
          f"[RESULT]\ntau={tau:.6g} s | zbar={length:.6g} m\noutput={folder}")

if __name__=="__main__": main()
