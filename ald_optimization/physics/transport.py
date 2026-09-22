"""Conservative P1 FEM with lumped storage and boundary reaction quadrature.

Algebraic graph viscosity removes positive spatial off-diagonal entries. It
preserves column sums, hence global conservation, unlike clipping. No SUPG is
needed with this monotone low-order alternative. Numerical diffusion is
explicitly reported and assessed in the mesh/time refinement benchmark.
"""
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fp
from mpi4py import MPI
from petsc4py import PETSc
from geometry.reactor_2d import INLET, OUTLET, WAFER
from config import pulse_average
from .kinetics import coefficients, coverage, implicit_update


def assembled_vector(form):
    v=fp.assemble_vector(fem.form(form))
    v.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,mode=PETSc.ScatterMode.REVERSE)
    data=v.array.real.copy(); v.destroy()
    return data


def monotone_operator(matrix):
    """B=A+G; G_ij=-max(A_ij,A_ji,0), G_ii=-sum_{j!=i}G_ij."""
    # petsc4py Mat.transpose() is in-place unless an output matrix is supplied.
    trans=matrix.copy()
    trans.transpose()
    result=matrix.copy()
    total=0.
    for i in range(*matrix.getOwnershipRange()):
        cols,vals=matrix.getRow(i)
        tcols,tvals=trans.getRow(i)
        reverse=dict(zip(tcols,tvals))
        new=vals.copy(); diagonal=np.flatnonzero(cols==i)
        added=0.
        for k,j in enumerate(cols):
            if j!=i:
                diffusion=max(float(vals[k]),float(reverse.get(j,0.)),0.)
                new[k]-=diffusion; added+=diffusion
        if not len(diagonal):
            raise RuntimeError("Missing FEM diagonal")
        new[diagonal[0]]+=added
        result.setValues(i,cols,new)
        total+=added/2
    result.assemble(); trans.destroy()
    return result,matrix.comm.tompi4py().allreduce(total)


class Transport:
    def __init__(self,reactor,cfg,velocity,diffusion_tensor=None):
        self.reactor,self.cfg,self.velocity=reactor,cfg,velocity
        self.mesh=reactor.domain; self.comm=self.mesh.comm
        self.V=fem.functionspace(self.mesh,("Lagrange",1))
        self.imap=self.V.dofmap.index_map
        self.n=self.imap.size_local
        self.coords=self.V.tabulate_dof_coordinates()[:self.n].copy()
        self.c=fem.Function(self.V); self.b=fem.Function(self.V)
        self.state=np.zeros((3,self.n))
        self.c_values=np.zeros(self.n); self.b_values=np.zeros(self.n)
        v=ufl.TestFunction(self.V); c=ufl.TrialFunction(self.V)
        dx=ufl.Measure("dx",domain=self.mesh); ds=reactor.ds
        n=ufl.FacetNormal(self.mesh); un=ufl.dot(velocity,n)
        # Outlet-only outflow. Inlet has prescribed total incoming flux.
        adv=-c*ufl.dot(velocity,ufl.grad(v))*dx+ufl.max_value(un,0)*c*v*ds(OUTLET)
        self.mass=assembled_vector(v*dx)
        self.surface=np.zeros(self.n)
        for tag in reactor.reactive_tags:
            self.surface+=assembled_vector(v*ds(tag))
        self.wafer=assembled_vector(v*ds(WAFER))
        self.inlet=assembled_vector(ufl.max_value(-un,0)*v*ds(INLET))
        self.outlet=assembled_vector(ufl.max_value(un,0)*v*ds(OUTLET))
        self.inlet_area=assembled_vector(v*ds(INLET))
        self.outlet_area=assembled_vector(v*ds(OUTLET))
        self.active=self.surface>0
        self.wafer_nodes=self.wafer>0
        self.inlet_local=np.flatnonzero(self.inlet_area>0).astype(np.int32)
        self.inlet_global=self.imap.local_to_global(self.inlet_local).astype(PETSc.IntType)
        diffusion=(cfg.diffusivity*ufl.inner(ufl.grad(c),ufl.grad(v)) if diffusion_tensor is None
                   else ufl.inner(ufl.dot(ufl.as_matrix(diffusion_tensor),ufl.grad(c)),ufl.grad(v)))
        matrices=[]; viscosity=[]
        for diff in (diffusion,cfg.byproduct_diffusivity*ufl.inner(ufl.grad(c),ufl.grad(v))):
            raw=fp.assemble_matrix(fem.form(adv+diff*dx)); raw.assemble()
            operator,added=monotone_operator(raw); raw.destroy()
            matrices.append(operator); viscosity.append(added)
        self.B,self.Bb=matrices
        self.graph_viscosity=viscosity
        self.rhs=self.B.createVecRight(); self.sol=self.rhs.duplicate()
        self.diagonal=self.rhs.duplicate(); self.temp=self.rhs.duplicate()
        self.ksp=PETSc.KSP().create(self.comm)
        self.ksp.setType("gmres"); self.ksp.setGMRESRestart(100)
        self.ksp.getPC().setType("ilu" if self.comm.size==1 else "bjacobi")
        self.ksp.setTolerances(rtol=cfg.linear_rtol,atol=1e-20,max_it=1000)
        self.ksp.setErrorIfNotConverged(True)
        self.qin=self.sum(self.inlet); self.qout=self.sum(self.outlet)
        if self.qin<=0 or self.qout<=0:
            raise RuntimeError("Inlet and outlet fluxes must both be positive")
        self.steps=[]; self.snapshots=[]; self.t=0.; self.rejected=0
        self.cumulative_in=0.; self.cumulative_out=0.
        self.cumulative_bp_out=0.; self.cumulative_bp_generated=0.

    def sum(self,x):
        return self.comm.allreduce(float(np.sum(x)),op=MPI.SUM)

    def max(self,x):
        return self.comm.allreduce(float(np.max(x)) if np.size(x) else -np.inf,op=MPI.MAX)

    def min(self,x):
        return self.comm.allreduce(float(np.min(x)) if np.size(x) else np.inf,op=MPI.MIN)

    def linear(self,B,old,robin,source,dt,cin=None):
        A=B.copy()
        self.diagonal.array[:]=self.mass/dt+self.surface*robin
        A.setDiagonal(self.diagonal,addv=PETSc.InsertMode.ADD_VALUES); A.assemble()
        self.rhs.array[:]=self.mass*old/dt+source
        if cin is not None:
            A.zeroRows(self.inlet_global,diag=1.)
            self.rhs.array[self.inlet_local]=cin
        self.ksp.setOperators(A); self.ksp.solve(self.rhs,self.sol)
        value=self.sol.array.real.copy(); its=self.ksp.getIterationNumber()
        A.destroy()
        return value,its

    def operator_total(self,B,values):
        self.sol.array[:]=values; B.mult(self.sol,self.temp)
        return self.sum(self.temp.array)

    def attempt(self,dt):
        p=self.cfg; oldc=self.c_values.copy(); oldb=self.b_values.copy(); old=self.state.copy()
        guess=old.copy(); cg=oldc.copy(); bg=oldb.copy()
        cin=pulse_average(self.t,self.t+dt,p)
        source=self.inlet*cin if p.inlet_condition=="flux" else np.zeros(self.n)
        boundary=cin if p.inlet_condition=="dirichlet" else None
        linear_iterations=0
        for iteration in range(1,p.max_coupling_iterations+1):
            ap,ab=coefficients(cg,guess,p.temperature,p)
            c,its=self.linear(self.B,oldc,ap,source,dt,boundary)
            linear_iterations+=its
            b,its=self.linear(self.Bb,oldb,ab,p.byproduct_yield*self.surface*ap*c,dt)
            linear_iterations+=its
            if min(self.min(c),self.min(b)) < -p.negative_tolerance*p.c0:
                raise ArithmeticError("Negative gas concentration beyond numerical tolerance")
            candidate=implicit_update(old,c,b,dt,p,guess)
            candidate[:,~self.active]=0.
            error=max(self.max(abs(candidate-guess)),self.max(abs(c-cg))/p.c0,
                      self.max(abs(b-bg))/p.c0)
            if error<p.coupling_tolerance:
                # c,b solved with guess. Surface accepted from the same flux:
                # candidate differs only by controlled fixed-point residual.
                break
            w=p.relaxation
            guess=w*candidate+(1-w)*guess
            cg=w*c+(1-w)*cg; bg=w*b+(1-w)*bg
        else:
            raise ArithmeticError("Gas/surface nonlinear coupling did not converge")
        total_occupancy=candidate[0]+candidate[2]
        if self.min(candidate)<-1e-9 or self.max(candidate)>1+1e-9 or self.max(total_occupancy)>1+1e-9:
            raise ArithmeticError("Surface bounds violated")
        theta=coverage(candidate,p); oldtheta=coverage(old,p)
        uptake=self.sum(self.surface*p.gamma*(theta-oldtheta))
        bpads=self.sum(self.surface*p.gamma*(candidate[2]-old[2]))
        out=self.operator_total(self.B,c)
        bpout=self.operator_total(self.Bb,b)
        supplied=self.qin*cin
        if boundary is not None:
            self.sol.array[:]=c; self.B.mult(self.sol,self.temp)
            residual=self.mass*(c-oldc)/dt+self.temp.array+self.surface*ap*c
            supplied=self.sum(residual[self.inlet_local])
        balance=self.sum(self.mass*(c-oldc))+uptake+dt*(out-supplied)
        generated=dt*self.sum(p.byproduct_yield*self.surface*ap*c)
        bp_balance=self.sum(self.mass*(b-oldb))+bpads+dt*bpout-generated
        scale=max(dt*supplied,uptake,self.qin*p.c0*dt*1e-3,1e-30)
        if abs(balance)/scale>2e-5 or abs(bp_balance)/scale>2e-5:
            raise ArithmeticError("Discrete gas/surface mass balance failed")
        self.c_values,self.b_values,self.state=c,b,candidate
        self.c.x.array[:self.n]=c; self.c.x.scatter_forward()
        self.b.x.array[:self.n]=b; self.b.x.scatter_forward()
        self.t+=dt
        self.cumulative_in+=dt*supplied; self.cumulative_out+=dt*out
        self.cumulative_bp_out+=dt*bpout; self.cumulative_bp_generated+=generated
        total_ads=self.sum(self.surface*p.gamma*theta)
        gas=self.sum(self.mass*c)
        inventory_error=gas+total_ads+self.cumulative_out-self.cumulative_in
        bp_inventory_error=(self.sum(self.mass*b)+self.sum(self.surface*p.gamma*candidate[2])
                            +self.cumulative_bp_out-self.cumulative_bp_generated)
        sw=self.sum(self.wafer)
        row={"time_s":self.t,"dt_s":dt,"inlet_c_mol_m3":cin,
             "theta_min":self.min(theta[self.wafer_nodes]),
             "theta_mean":self.sum(self.wafer*theta)/sw,
             "theta_max":self.max(theta[self.wafer_nodes]),
             "blocked_mean":self.sum(self.wafer*candidate[2])/sw,
             "c_min_mol_m3":self.min(c),"c_max_mol_m3":self.max(c),
             "byproduct_min_mol_m3":self.min(b),
             "outlet_precursor_c_mol_m3":self.sum(self.outlet*c)/self.qout,
             "outlet_byproduct_c_mol_m3":self.sum(self.outlet*b)/self.qout,
             "outlet_precursor_area_c_mol_m3":self.sum(self.outlet_area*c)/self.sum(self.outlet_area),
             "gas_inventory_mol":gas,"surface_uptake_mol":total_ads,
             "cumulative_in_mol":self.cumulative_in,"cumulative_out_mol":self.cumulative_out,
             "balance_residual_mol":inventory_error,
             "balance_relative":inventory_error/max(self.cumulative_in,1e-30),
             "byproduct_balance_residual_mol":bp_inventory_error,
             "byproduct_balance_relative":bp_inventory_error/max(self.cumulative_bp_generated,1e-30),
             "step_balance_mol":balance,"coupling_iterations":iteration,
             "coupling_error":error,"linear_iterations":linear_iterations,
             "rejected_steps_total":self.rejected,
             "qcm_normalized_mass":self.sum(self.wafer*theta)/sw}
        # Spatial QCM-like finite patches; report normalized mass, no invented film density.
        for j,fraction in enumerate((0.1,0.3,0.5,0.7,0.9)):
            if self.mesh.geometry.dim==2:
                center=p.wafer_start+fraction*(p.wafer_end-p.wafer_start)
                mask=abs(self.coords[:,0]-center)<max(p.length/p.nx,0.012)
            else:
                center=(2*fraction-1)*p.wafer_radius*0.8
                mask=(self.coords[:,0]-center)**2+self.coords[:,1]**2<0.035**2
            weight=self.wafer*mask; area=self.sum(weight)
            row[f"qcm_{j+1}_coverage"]=self.sum(weight*theta)/area if area>0 else None
        self.steps.append(row)
        return row

    def collect(self,fields):
        """Gather owned CG1 DOFs in global order; root-only plotting."""
        chunks=self.comm.gather((self.imap.local_range[0],self.coords,fields),root=0)
        if self.comm.rank:
            return None
        chunks.sort(key=lambda item:item[0])
        return {"coordinates":np.concatenate([a[1] for a in chunks]),
                **{key:np.concatenate([a[2][key] for a in chunks]) for key in fields}}

    def snapshot(self):
        fields={"concentration":self.c_values,"byproduct":self.b_values,
                "theta":coverage(self.state,self.cfg),"blocked":self.state[2],
                "wafer_weight":self.wafer}
        result=self.collect(fields)
        if result is not None:
            result["time"]=self.t
            self.snapshots.append(result)

    def run(self,times=None):
        p=self.cfg
        requested=[0.1,0.2,0.3,0.5,0.7,1.0,p.end_time] if times is None else list(times)
        events=sorted(set([round(t,12) for t in requested if 0<t<=p.end_time]+[p.end_time,
                     p.rise_time,p.pulse_duration,p.pulse_duration+p.rise_time]))
        events=[t for t in events if t>0]
        snapshot_times=np.array(sorted(set([t for t in requested if 0<t<=p.end_time]+[p.end_time])))
        self.snapshot()
        dt=p.dt
        for target in events:
            while self.t<target-1e-11:
                trial=min(dt,target-self.t)
                try:
                    self.attempt(trial)
                except (ArithmeticError,PETSc.Error) as exc:
                    self.rejected+=1; dt=trial/2
                    if dt<p.min_dt:
                        raise RuntimeError(f"dt fell below {p.min_dt:g} at t={self.t:g}: {exc}") from exc
                    continue
                dt=min(p.dt,dt*1.25)
            if np.any(abs(snapshot_times-self.t)<1e-9):
                self.snapshot()
        return self.steps,self.snapshots

    def close(self):
        for item in (self.B,self.Bb,self.rhs,self.sol,self.diagonal,self.temp,self.ksp):
            item.destroy()
