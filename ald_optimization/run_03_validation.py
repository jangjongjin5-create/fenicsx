"""Refinement in the analytical limit plus an explicit model-discrepancy case."""
from dataclasses import replace
import numpy as np
from config import Config,arguments
from geometry.reactor_2d import build
from physics.flow import prescribed
from analysis.analytical_solution import transient_coverage
from analysis.validation import metrics
from analysis.diagnostics import start_output,write_csv,save_json
from plotting.plots import plt,finish,wafer_profile
from workflow import run_transport

def main():
    cfg,args=arguments("Quantitative validation",replace(Config(),length=0.4,height=0.025,
        mean_velocity=1.0,wafer_start=0.,wafer_end=0.4,rise_time=0.,pulse_duration=0.2,purge_duration=0.5))
    resolutions=[(60,4,0.01),(120,6,0.005)] if args.quick else [(100,6,0.004),(200,8,0.002),(400,12,0.001)]
    records=[]; results=[]; histories=[]
    for i,(nx,ny,dt) in enumerate(resolutions):
        case=replace(cfg,nx=nx,ny=ny,dt=dt)
        reactor=build(case,benchmark=True)
        folder=start_output(case,f"03_validation/refinement_{i+1}",reactor.domain.comm)
        history,snapshots,meta=run_transport(reactor,case,prescribed(reactor,case),folder,
                         diffusion_tensor=[[1e-8,0.],[0.,10.]],times=[0.1,0.2,0.3,0.5,case.end_time],make_plots=False)
        if reactor.domain.comm.rank==0:
            results.append((case,snapshots)); histories.append(meta)
            for sn in snapshots[1:]:
                x,y=wafer_profile(sn)
                analytic=transient_coverage(x,sn["time"],case,case.height/2)
                records.append({"case":f"limit_mesh_{nx}","nx":nx,"dt_s":dt,"time_s":sn["time"],**metrics(x,y,analytic)})
    case=replace(cfg,nx=resolutions[-1][0],ny=resolutions[-1][1],dt=resolutions[-1][2])
    reactor=build(case,benchmark=True)
    folder=start_output(case,"03_validation/physical_diffusion",reactor.domain.comm)
    _,physical,_=run_transport(reactor,case,prescribed(reactor,case),folder,
                    times=[0.1,0.2,0.3,0.5,case.end_time],make_plots=False)
    if reactor.domain.comm.rank==0:
        root=folder.parent
        for sn in physical[1:]:
            x,y=wafer_profile(sn); analytic=transient_coverage(x,sn["time"],case,case.height/2)
            records.append({"case":"physical_D_0.01","nx":case.nx,"dt_s":case.dt,
                            "time_s":sn["time"],**metrics(x,y,analytic)})
        write_csv(root/"validation_metrics.csv",records)
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for ax,target in zip(axes,(0.2,0.5,case.end_time)):
            x=np.linspace(0,case.length,1201)
            ax.plot(x,transient_coverage(x,target,case,case.height/2),"k--",lw=2,label="Analytical")
            for c,snaps in results:
                sn=min(snaps,key=lambda a:abs(a["time"]-target)); xx,yy=wafer_profile(sn)
                ax.plot(xx,yy,label=f"FEM limit nx={c.nx}")
            sn=min(physical,key=lambda a:abs(a["time"]-target)); xx,yy=wafer_profile(sn)
            ax.plot(xx,yy,color="#bd762a",lw=2,label="FEM physical diffusion")
            ax.set(xlabel="x [m]",ylabel="Coverage",title=f"t = {target:.2f} s",ylim=(-0.01,1.02))
        axes[-1].legend(fontsize=8); finish(fig,root/"analytical_vs_fem.png")
        final=[r for r in records if r["case"].startswith("limit") and abs(r["time_s"]-case.end_time)<1e-9]
        decreased=all(final[i+1]["rmse"]<final[i]["rmse"] for i in range(len(final)-1))
        passed=(decreased or max(r["rmse"] for r in final)<1e-10) and final[-1]["rmse"]<0.08
        save_json(root/"validation_assessment.json",{"analytical_limit_refinement_pass":passed,
            "final_profile_rmse_by_mesh":final,"criterion":"decreasing RMSE (or all below 1e-10) and finest final RMSE < 0.08",
            "disclaimer":"Code verification of an asymptotic limit, not reproduction of all paper figures.",
            "limit_changes":"Dx=1e-8, Dy=10 m²/s; plug velocity; both planar walls reactive; rectangular pulse; H/2=R_tube/2",
            "physical_discrepancy":"isotropic D=0.01 has axial spreading and transverse concentration gradients"})
        print(f"[VALIDATION]\nRefinement criterion passed: {passed}\nFinest final RMSE: {final[-1]['rmse']:.6g}")
        if not passed:
            raise RuntimeError("Analytical validation criterion failed; inspect refinement metrics")

if __name__=="__main__": main()
