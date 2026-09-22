from config import arguments
from geometry.reactor_2d import build
from physics.flow import solve
from analysis.diagnostics import start_output
from workflow import run_transport,export_flow

def main():
    cfg,_=arguments("Solved 2D carrier flow followed by frozen-flow ALD")
    reactor=build(cfg); folder=start_output(cfg,"04_flow",reactor.domain.comm)
    u,p,meta=solve(reactor,cfg)
    export_flow(reactor,cfg,u,p,meta,folder)
    run_transport(reactor,cfg,u,folder)

if __name__=="__main__": main()
