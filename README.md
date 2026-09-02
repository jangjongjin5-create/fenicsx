## FEniCSx Environment

This project uses **FEniCSx / DOLFINx** for finite element simulations.

### Installation

Using Conda/Mamba is recommended.

```bash
# Create environment
mamba create -n fenicsx -c conda-forge \
    fenics-dolfinx \
    mpi4py \
    petsc4py \
    gmsh \
    pyvista \
    numpy \
    scipy \
    matplotlib

# Activate environment
mamba activate fenicsx
```

If `mamba` is not installed:

```bash
conda install -n base -c conda-forge mamba
```

### Core Libraries

- **DOLFINx** — Finite element solver
- **UFL** — Variational formulation
- **PETSc / petsc4py** — Linear and nonlinear solvers
- **MPI / mpi4py** — Parallel computing
- **Gmsh** — Mesh generation
- **PyVista** — FEM visualization
- **NumPy / SciPy** — Numerical computation
- **Matplotlib** — Plotting

### Installation Check

```bash
python -c "import dolfinx; print('DOLFINx:', dolfinx.__version__)"
```

MPI test:

```bash
mpirun -np 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"
```
