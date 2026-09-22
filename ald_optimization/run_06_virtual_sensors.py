from dataclasses import replace
from config import Config,arguments,TORR
from geometry.reactor_2d import build
from physics.flow import solve
from analysis.diagnostics import start_output,write_csv,save_json
from workflow import run_transport,export_flow

def main():
    cfg,_=arguments("Virtual QMS/QCM signals with hidden-state targets",replace(Config(),
                    precursor_pressure=50e-3*TORR,pulse_duration=0.4,purge_duration=1.2))
    reactor=build(cfg); folder=start_output(cfg,"06_virtual_sensors",reactor.domain.comm)
    u,p,flow=solve(reactor,cfg); export_flow(reactor,cfg,u,p,flow,folder)
    steps,_,_=run_transport(reactor,cfg,u,folder)
    if reactor.domain.comm.rank==0:
        keys=["time_s","outlet_precursor_c_mol_m3","outlet_byproduct_c_mol_m3"]+[f"qcm_{i}_coverage" for i in range(1,6)]
        write_csv(folder/"sparse_measurements.csv",[{k:r[k] for k in keys} for r in steps])
        write_csv(folder/"hidden_state_targets.csv",[{k:r[k] for k in ("time_s","theta_min","theta_mean","theta_max","blocked_mean","gas_inventory_mol")} for r in steps])
        save_json(folder/"sensor_definition.json",{
            "QMS":"flow-weighted outlet molar concentration; no fragmentation/sensitivity model",
            "QCM":"finite wafer patches; normalized precursor uptake, not calibrated frequency",
            "blocking_limit":"byproduct occupied sites excluded from precursor QCM signal; retained ligand mass unknown",
            "noise":"none; add calibrated instrument response/noise separately",
            "identifiability":"signals alone do not establish uniqueness of hidden-state inference"})

if __name__=="__main__": main()
