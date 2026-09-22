"""MPI-safe evaluation on an actual FEM mesh, not a reconstructed interpolant."""

import numpy as np
from dolfinx import geometry


def sample(function, points):
    m = function.function_space.mesh

    tree = geometry.bb_tree(
        m,
        m.topology.dim
    )

    candidates = geometry.compute_collisions_points(
        tree,
        points
    )

    collisions = geometry.compute_colliding_cells(
        m,
        candidates,
        points
    )

    inds = []
    cells = []

    owned = (
        m.topology
        .index_map(m.topology.dim)
        .size_local
    )

    for i in range(len(points)):
        hits = collisions.links(i)
        hits = hits[hits < owned]

        if len(hits):
            inds.append(i)
            cells.append(hits[0])

    count = (
        int(np.prod(function.ufl_shape))
        if function.ufl_shape
        else 1
    )

    local = np.full(
        (len(points), count),
        np.nan
    )

    if inds:
        local[inds] = function.eval(
            points[inds],
            np.asarray(cells, dtype=np.int32)
        ).reshape(-1, count)

    chunks = m.comm.gather(
        local,
        root=0
    )

    if m.comm.rank:
        return None

    out = np.full_like(
        local,
        np.nan
    )

    for c in chunks:
        ok = np.isfinite(c[:, 0])
        out[ok] = c[ok]

    return out


def grid(cfg, dim, nx=101, ny=51):

    if dim == 2:

        X, Y = np.meshgrid(
            np.linspace(
                0,
                cfg.length,
                nx
            ),
            np.linspace(
                0,
                cfg.height,
                ny
            )
        )

        points = np.column_stack([
            X.ravel(),
            Y.ravel(),
            np.zeros(X.size)
        ])

    else:

        X, Y = np.meshgrid(
            np.linspace(
                -cfg.chamber_radius,
                cfg.chamber_radius,
                nx
            ),
            np.linspace(
                -cfg.chamber_radius,
                cfg.chamber_radius,
                nx
            )
        )

        points = np.column_stack([
            X.ravel(),
            Y.ravel(),
            np.full(
                X.size,
                cfg.height / 2
            )
        ])

    return X, Y, points


def triangles(solver):

    if solver.mesh.geometry.dim != 2:
        return None

    n = (
        solver.mesh.topology
        .index_map(2)
        .size_local
    )

    local = np.array([
        solver.imap.local_to_global(
            solver.V.dofmap.cell_dofs(i)
        )
        for i in range(n)
    ])

    chunks = solver.comm.gather(
        local,
        root=0
    )

    return (
        np.concatenate(chunks)
        if solver.comm.rank == 0
        else None
    )


def wafer_triangles(solver):

    from dolfinx import fem
    from geometry.reactor_2d import WAFER

    if solver.mesh.geometry.dim != 3:
        return None

    facets = solver.reactor.tags.find(
        WAFER
    )

    owned = (
        solver.mesh.topology
        .index_map(2)
        .size_local
    )

    facets = facets[
        facets < owned
    ]

    rows = []

    for facet in facets:

        dofs = fem.locate_dofs_topological(
            solver.V,
            2,
            np.array(
                [facet],
                dtype=np.int32
            )
        )

        rows.append(
            solver.imap.local_to_global(
                dofs
            )
        )

    chunks = solver.comm.gather(
        np.asarray(
            rows,
            dtype=np.int64
        ).reshape(-1, 3),
        root=0
    )

    return (
        np.concatenate(chunks)
        if solver.comm.rank == 0
        else None
    )
