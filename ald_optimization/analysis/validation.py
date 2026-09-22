import numpy as np
from .diagnostics import front_position


def metrics(x, numerical, reference):
    """
    Compare numerical and reference surface-coverage profiles.

    Returns:
        RMSE
        MAE
        relative L2 error
        FEM / analytical x50 front positions
        front-position error
    """

    order = np.argsort(x)

    x = np.asarray(x)[order]
    n = np.asarray(numerical)[order]
    a = np.asarray(reference)[order]

    e = n - a

    length = x[-1] - x[0]

    error = np.trapezoid(
        e * e,
        x
    )

    norm = np.trapezoid(
        a * a,
        x
    )

    nf = front_position(
        x,
        n
    )

    af = front_position(
        x,
        a
    )

    return {
        "rmse": float(
            np.sqrt(error / length)
        ),

        "mae": float(
            np.trapezoid(
                np.abs(e),
                x
            ) / length
        ),

        "relative_L2": (
            float(
                np.sqrt(error / norm)
            )
            if norm > 1e-24
            else None
        ),

        "x50_fem_m": nf,

        "x50_analytic_m": af,

        "x50_error_m": (
            abs(nf - af)
            if nf is not None
            and af is not None
            else None
        ),

        "front_status": (
            "both_cross"
            if nf is not None
            and af is not None
            else "one_or_both_outside_domain"
        )
    }
