"""Steady incompressible carrier flow: Taylor-Hood P2/P1, frozen for transport."""
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc
from geometry.reactor_2d import INLET, OUTLET, WAFER, TOP, SIDE


def integral(expr,comm):
    return comm.allreduce(fem.assemble_scalar(fem.form(expr)),op=MPI.SUM).real


def prescribed(reactor,cfg):
    d=reactor.domain.geometry.dim
    return fem.Constant(reactor.domain,np.array([cfg.mean_velocity]+[0.]*(d-1),dtype=PETSc.ScalarType))


def solve(reactor,cfg):
    m=reactor.domain; comm=m.comm; d=m.geometry.dim
    cell=m.basix_cell()
    W=fem.functionspace(m,mixed_element([element("Lagrange",cell,2,shape=(d,)),
                                         element("Lagrange",cell,1)]))
    V,_=W.sub(0).collapse()
    inlet=fem.Function(V)
    if d==2:
        def profile(x):
            a=x[1]/cfg.height
            return np.vstack((6*cfg.mean_velocity*a*(1-a),np.zeros_like(a)))
    else:
        def profile(x):
            r2=((x[0]+cfg.port_offset)**2+x[1]**2)/cfg.port_radius**2
            return np.vstack((np.zeros_like(r2),np.zeros_like(r2),
                              2*cfg.actual_volume_flow/(np.pi*cfg.port_radius**2)*(1-r2)))
    inlet.interpolate(profile)
    fdim=m.topology.dim-1
    wall_facets=np.unique(np.concatenate([reactor.tags.find(t) for t in (WAFER,TOP,SIDE)]))
    # The polygonal circular rim has P2 edge nodes inside the true circle.
    # Enforce shared wall DOFs exactly zero before normalizing inlet flux.
    wall_blocks=fem.locate_dofs_topological(V,fdim,wall_facets)
    inlet.x.array.reshape(-1,d)[wall_blocks,:]=0.
    inlet.x.scatter_forward()
    n=ufl.FacetNormal(m); dx=ufl.Measure("dx",domain=m); ds=reactor.ds
    requested=cfg.mean_velocity*cfg.height if d==2 else cfg.actual_volume_flow
    measured=-integral(ufl.dot(inlet,n)*ds(INLET),comm)
    inlet.x.array[:]*=requested/measured
    inlet.x.scatter_forward()
    zero=fem.Function(V)
    wall_dofs=fem.locate_dofs_topological((W.sub(0),V),fdim,wall_facets)
    in_dofs=fem.locate_dofs_topological((W.sub(0),V),fdim,reactor.tags.find(INLET))
    bcs=[fem.dirichletbc(inlet,in_dofs,W.sub(0)),fem.dirichletbc(zero,wall_dofs,W.sub(0))]
    u,p=ufl.TrialFunctions(W); v,q=ufl.TestFunctions(W)
    previous=fem.Function(V)
    inertia=fem.Constant(m,PETSc.ScalarType(0.))
    a=(cfg.kinematic_viscosity*ufl.inner(ufl.grad(u),ufl.grad(v))
       +cfg.grad_div_factor*cfg.kinematic_viscosity*ufl.div(u)*ufl.div(v)
       +inertia*ufl.inner(ufl.dot(ufl.grad(u),previous),v)
       -p*ufl.div(v)+q*ufl.div(u))*dx
    rhs=ufl.inner(fem.Constant(m,np.zeros(d,dtype=PETSc.ScalarType)),v)*dx
    solution=fem.Function(W)
    problem=LinearProblem(a,rhs,bcs=bcs,u=solution,petsc_options_prefix="ald_flow_",
        petsc_options={"ksp_type":"preonly","pc_type":"lu",
                       "pc_factor_mat_solver_type":"mumps","ksp_error_if_not_converged":True})
    history=[]
    for iteration in range(1,81):
        problem.solve(); solution.x.scatter_forward()
        velocity=solution.sub(0).collapse()
        local=np.sum(abs(velocity.x.array-previous.x.array)**2)
        denom=np.sum(abs(velocity.x.array)**2)
        relative=np.sqrt(comm.allreduce(local)/max(comm.allreduce(denom),1e-30))
        history.append({"iteration":iteration,"relative_velocity_change":float(relative),
                        "inertia_included":bool(float(inertia.value))})
        if cfg.flow_model=="stokes" or (iteration>1 and relative<1e-8):
            break
        previous.x.array[:]=0.7*velocity.x.array+0.3*previous.x.array
        previous.x.scatter_forward()
        inertia.value=PETSc.ScalarType(1.)
    else:
        raise RuntimeError("Steady Navier-Stokes Picard iteration failed")
    pressure=solution.sub(1).collapse()  # kinematic pressure p/rho
    pressure.x.array[:]*=cfg.density
    pressure.name="pressure_Pa_relative_to_outlet_traction"
    velocity.name="carrier_velocity"
    qin=-integral(ufl.dot(velocity,n)*ds(INLET),comm)
    qout=integral(ufl.dot(velocity,n)*ds(OUTLET),comm)
    div2=integral(ufl.div(velocity)**2*dx,comm)
    volume=integral(fem.Constant(m,PETSc.ScalarType(1.))*dx,comm)
    speed_rms=np.sqrt(integral(ufl.inner(velocity,velocity)*dx,comm)/volume)
    coord=ufl.SpatialCoordinate(m)
    if d==2:
        region=ufl.conditional(ufl.And(ufl.ge(coord[0],cfg.wafer_start),
                                      ufl.le(coord[0],cfg.wafer_end)),1.,0.)
        speed=velocity[0]
    else:
        region=ufl.conditional(ufl.le(coord[0]**2+coord[1]**2,cfg.wafer_radius**2),1.,0.)
        speed=ufl.sqrt(velocity[0]**2+velocity[1]**2)
    avg=integral(region*speed*dx,comm)/integral(region*dx,comm)
    data={"model":cfg.flow_model,"flow_in":qin,"flow_out":qout,
          "flow_balance_relative":abs(qout-qin)/qin,"divergence_rms_s_inv":float(np.sqrt(div2/volume)),
          "divergence_scaled_by_H_over_u_rms":float(np.sqrt(div2/volume)*cfg.height/speed_rms),
          "grad_div_stabilization_m2_s":cfg.grad_div_factor*cfg.kinematic_viscosity,
          "wafer_region_mean_horizontal_velocity_m_s":avg,"paper_wafer_velocity_m_s":0.5,
          "difference_from_paper_velocity_percent":100*(avg/0.5-1),
          "pressure_approximation":"incompressible; p is mechanical gauge pressure",
          "iterations":history}
    if d==2:
        exact=ufl.as_vector([6*cfg.mean_velocity*(coord[1]/cfg.height)*(1-coord[1]/cfg.height),0.])
        data["poiseuille_velocity_relative_L2"]=float(np.sqrt(
            integral(ufl.inner(velocity-exact,velocity-exact)*dx,comm)/
            integral(ufl.inner(exact,exact)*dx,comm)))
        data["poiseuille_pressure_drop_Pa"]=12*cfg.density*cfg.kinematic_viscosity*cfg.mean_velocity*cfg.length/cfg.height**2
    if data["flow_balance_relative"]>0.01:
        raise RuntimeError(f"Flow balance failed: {data['flow_balance_relative']:.3e}")
    return velocity,pressure,data
