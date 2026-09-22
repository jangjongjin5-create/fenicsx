"""Transient inverse solver on the existing DOLFINx-assembled lumped P1 FEM.

State u=(cp/c_ref, cb/c_ref, theta_p at reactive DOFs). Each BE residual
is scaled by its storage, so dF_n/du_(n-1)=-I, including nonuniform dt.
SciPy SuperLU factors the serial DOLFINx matrices and their transpose.
This is an algebraic solver choice, not a replacement of the FEM operators.
No competitive law or hidden-state input is used in this module.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from config import pulse_average
from physics.kinetics import rate_constants


@dataclass
class Operators:
    B: object
    Bb: object
    mass: np.ndarray
    surface: np.ndarray
    wafer: np.ndarray
    inlet: np.ndarray
    outlet: np.ndarray
    coordinates: np.ndarray
    graph_viscosity: list

    @classmethod
    def from_transport(cls, transport):
        if transport.comm.size != 1:
            raise RuntimeError("Stage 8 currently requires one MPI rank; Stages 1–7 retain MPI support")
        def csr(A):
            indptr, indices, data = A.getValuesCSR()
            return sparse.csr_matrix((data.copy(), indices.copy(), indptr.copy()), shape=A.getSize())
        return cls(csr(transport.B), csr(transport.Bb),
                   *[getattr(transport, name).copy() for name in
                     ("mass", "surface", "wafer", "inlet", "outlet", "coords")],
                   list(transport.graph_viscosity))


@dataclass
class Trajectory:
    times: np.ndarray
    states: np.ndarray
    factors: list
    iterations: list
    residuals: list
    diagnostics: dict


class TransientFEM:
    def __init__(self, operators, cfg, concentration_ref, adapter, tolerance=2e-10):
        if cfg.inlet_condition != "flux":
            raise ValueError("Stage 8 supports the Stage 5 Danckwerts flux inlet")
        self.ops, self.cfg, self.ref, self.adapter = operators, cfg, float(concentration_ref), adapter
        self.n = len(operators.mass)
        self.active = np.flatnonzero(operators.surface > 0)
        self.s = len(self.active)
        self.size = 2*self.n+self.s
        self.kp = rate_constants(cfg.temperature, cfg)[0]
        self.tolerance = tolerance
        if self.s == 0 or np.any(operators.mass <= 0):
            raise ValueError("Positive volume weights and a reactive surface are required")
        # H maps a normalized surface flux [m/s] into each storage-scaled PDE.
        j = np.arange(self.s)
        a = self.active
        w = operators.surface[a]/operators.mass[a]
        rows = np.concatenate([a, self.n+a, 2*self.n+j])
        cols = np.tile(j, 3)
        vals = np.concatenate([w, -cfg.byproduct_yield*w,
                               np.full(self.s, -self.ref/cfg.gamma)])
        self.H = sparse.csc_matrix((vals, (rows, cols)), shape=(self.size, self.s))
        self.G = sparse.block_diag([sparse.diags(1/operators.mass)@operators.B,
                                   sparse.diags(1/operators.mass)@operators.Bb,
                                   sparse.csr_matrix((self.s, self.s))], format="csc")
        self.identity = sparse.eye(self.size, format="csc")
        self._linear_cache = {}

    def local_state(self, u):
        return np.stack([u[self.active], u[self.n+self.active], u[2*self.n:]], axis=1)

    def residual_jacobian(self, u, previous, t0, t1, jacobian=True):
        dt = t1-t0
        xi = self.local_state(u)
        if jacobian:
            g, dg = self.adapter.value_tangent(xi)
        else:
            g = self.adapter.value(xi)
        if not np.all(np.isfinite(g)) or np.any(g < 0) or np.any(g > 1):
            raise ArithmeticError("Nonfinite or unbounded local closure")
        a, b, theta = xi.T
        base = self.kp*a*(1-theta)
        r = base*g
        key = round(dt, 13)
        if key not in self._linear_cache:
            self._linear_cache[key] = self.identity+dt*self.G
        A = self._linear_cache[key]
        residual = A@u-previous+dt*(self.H@r)
        residual[:self.n] -= dt*self.ops.inlet/self.ops.mass*pulse_average(t0,t1,self.cfg)/self.ref
        if not jacobian:
            return residual
        dr = base[:, None]*dg
        dr[:, 0] += self.kp*(1-theta)*g
        dr[:, 2] -= self.kp*a*g
        rows = np.tile(np.arange(self.s), 3)
        cols = np.concatenate([self.active, self.n+self.active, np.arange(2*self.n,self.size)])
        D = sparse.csc_matrix((dr.T.ravel(), (rows, cols)), shape=(self.s,self.size))
        return residual, (A+dt*self.H@D).tocsc()

    def admissible(self, u):
        return (np.all(np.isfinite(u)) and np.min(u) >= -1e-10
                and np.max(u[2*self.n:]) <= 1+1e-10)

    def step(self, previous, t0, t1, retain_factor):
        u = previous.copy()
        for iteration in range(40):
            residual, K = self.residual_jacobian(u, previous, t0, t1)
            norm = np.linalg.norm(residual, ord=np.inf)
            if not np.isfinite(norm):
                raise ArithmeticError(f"Nonfinite residual at t={t1:g}")
            if norm < self.tolerance:
                if not self.admissible(u):
                    raise ArithmeticError(f"State bounds violated at t={t1:g}")
                return u, splu(K) if retain_factor else None, iteration, norm
            try:
                factor = splu(K)
                increment = factor.solve(-residual)
            except RuntimeError as exc:
                raise ArithmeticError(f"Singular nonlinear Jacobian at t={t1:g}") from exc
            alpha = 1.
            for _ in range(30):
                trial = u+alpha*increment
                if self.admissible(trial):
                    trial_norm = np.linalg.norm(self.residual_jacobian(trial,previous,t0,t1,False), ord=np.inf)
                    if trial_norm <= (1-1e-4*alpha)*norm or trial_norm < self.tolerance:
                        u = trial
                        break
                alpha *= .5
            else:
                raise ArithmeticError(f"Newton line search failed at t={t1:g}; residual={norm:.3e}")
        raise ArithmeticError(f"Newton exceeded 40 iterations at t={t1:g}")

    def forward(self, times, retain_factors=False):
        times = np.asarray(times, dtype=float)
        if times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError("A strictly increasing, fixed time grid starting at zero is required")
        states = np.zeros((len(times), self.size))
        factors, iterations, residuals = [], [], []
        for i in range(1,len(times)):
            u, factor, count, residual = self.step(states[i-1], times[i-1], times[i], retain_factors)
            states[i] = u
            factors.append(factor)
            iterations.append(count)
            residuals.append(residual)
        diagnostics = self.diagnostics(times, states, iterations, residuals)
        if max(diagnostics["precursor_mass_relative"], diagnostics["byproduct_mass_relative"]) > 2e-5:
            raise ArithmeticError(f"Mass conservation failed: {diagnostics}")
        return Trajectory(times, states, factors, iterations, residuals, diagnostics)

    def adjoint(self, trajectory, loss_state_gradient):
        """K_n^T lambda_n = L_(u_n) + lambda_(n+1); gradient=-sum F_phi^T lambda.

        The +lambda_(n+1) term is essential. Only local g outputs receive
        parameter cotangents; dg/dxi already enters K, not an independent VJP.
        """
        lam_next = np.zeros(self.size)
        all_xi, all_cotangent = [], []
        maximum_residual = 0.
        for i in range(len(trajectory.times)-1, 0, -1):
            u = trajectory.states[i]
            rhs = loss_state_gradient[i]+lam_next
            factor = trajectory.factors[i-1]
            _, K = self.residual_jacobian(u, trajectory.states[i-1],
                                          trajectory.times[i-1], trajectory.times[i])
            if factor is None:
                factor = splu(K)
            lam = factor.solve(rhs, trans="T")
            maximum_residual = max(maximum_residual,
                float(np.linalg.norm(K.T@lam-rhs)/max(np.linalg.norm(rhs), 1e-30)))
            xi = self.local_state(u)
            base = self.kp*xi[:,0]*(1-xi[:,2])
            cotangent = -(trajectory.times[i]-trajectory.times[i-1])*base*(self.H.T@lam)
            all_xi.append(xi)
            all_cotangent.append(cotangent)
            lam_next = lam
        self.adapter.vjp(np.concatenate(all_xi), np.concatenate(all_cotangent))
        return maximum_residual

    def diagnostics(self, times, states, iterations, residuals):
        cp, cb, th = states[:,:self.n]*self.ref, states[:,self.n:2*self.n]*self.ref, states[:,2*self.n:]
        dt = np.diff(times)
        injected = np.cumsum([self.ops.inlet.sum()*pulse_average(a,b,self.cfg)*(b-a)
                              for a,b in zip(times[:-1],times[1:])])
        uptake = th@(self.ops.surface[self.active]*self.cfg.gamma)
        outp = np.cumsum(dt*(cp[1:]@self.ops.outlet))
        outb = np.cumsum(dt*(cb[1:]@self.ops.outlet))
        rp = cp[1:]@self.ops.mass+uptake[1:]+outp-injected
        rb = cb[1:]@self.ops.mass+outb-self.cfg.byproduct_yield*uptake[1:]
        scale = max(float(injected[-1]),1e-30)
        return dict(precursor_min_mol_m3=float(cp.min()), precursor_max_mol_m3=float(cp.max()),
                    byproduct_min_mol_m3=float(cb.min()), byproduct_max_mol_m3=float(cb.max()),
                    theta_min=float(th.min()), theta_max=float(th.max()),
                    precursor_mass_relative=float(np.max(abs(rp))/scale),
                    byproduct_mass_relative=float(np.max(abs(rb))/scale),
                    mass_normalizer="total injected precursor", max_newton_iterations=max(iterations),
                    total_newton_iterations=sum(iterations), maximum_residual=max(residuals),
                    min_dt=float(dt.min()), max_dt=float(dt.max()),
                    time_grid="fixed, no state-dependent step adaptation", clipping=False)
