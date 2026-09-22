import numpy as np
from .diagnostics import front_position


def metrics(x,numerical,reference):
    order=np.argsort(x); x=np.asarray(x)[order]
    n=np.asarray(numerical)[order]; a=np.asarray(reference)[order]
    e=n-a; length=x[-1]-x[0]
    trap=np.trapezoid
    error=trap(e*e,x); norm=trap(a*a,x)
    nf,af=front_position(x,n),front_position(x,a)
    return {"rmse":float(np.sqrt(error/length)),
            "mae":float(trap(abs(e),x)/length),
            "relative_L2":float(np.sqrt(error/norm)) if norm>1e-24 else None,
            "x50_fem_m":nf,"x50_analytic_m":af,
            "x50_error_m":abs(nf-af) if nf is not None and af is not None else None,
            "front_status":"both_cross" if nf is not None and af is not None else "one_or_both_outside_domain"}
