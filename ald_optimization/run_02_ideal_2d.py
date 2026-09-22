from config import arguments
from geometry.reactor_2d import build
from physics.flow import prescribed
from analysis.diagnostics import start_output
from workflow import run_transport

def main():
    cfg,_=arguments("2D pulsed ALD transport")
    reactor=build(cfg)
    folder=start_output(cfg,"02_ideal_2d",reactor.domain.comm)
    run_transport(reactor,cfg,prescribed(reactor,cfg),folder)

if __name__=="__main__": main()
