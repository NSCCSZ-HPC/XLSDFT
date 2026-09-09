# XLSDFT

XLSDFT is a real-space density functional theory software package for
large-scale electronic structure calculations. It combines MPI and OpenMP
parallelism with optimized stencil, eigensolver, density, Poisson, and force
calculation components.

This source release contains the core program and the LX2 build configuration.
Documentation, examples, pseudopotentials, and validation cases can be added as
separate top-level sections in later releases.

## Contents

- `src/`: source code and build files.
- `src/include/`: public and internal headers used by the solver.
- `src/Makefile`: build entry.
- `src/Makefile.config.LX2`: compiler flags, feature switches, and dependency paths.
- `src/objects.mk`: object list for the executable.

## Requirements

The build environment needs:

- MPI C++ compiler wrapper, such as `mpicxx`.
- C++17-capable compiler.
- OpenMP runtime.
- FFTW-compatible transform library.
- JSON library and SStructMG dependencies.
- HBM-related libraries.
- LVTX BLAS header tree for the optimized backend.

## Build

Build from the source directory:

```shell
cd src
cp Makefile.config.LX2 Makefile.config
make clean
make -j
```

The executable is generated as:

```shell
src/a.out
```

`Makefile.config.LX2` contains default dependency locations for the current LX2
environment. If a library is installed somewhere else, override the path before
building:

```shell
export LVTX_BLAS_ROOT=/path/to/lvtx_blas
export HBMROOT=/path/to/memory
export SOFT_HOME=/path/to/dependencies
export JSON_HOME=/path/to/jsoncpp
export SSTRUCTMG_HOME=/path/to/sstructmg

cd src
cp Makefile.config.LX2 Makefile.config
make -j
```

The OpenMP team size used by the optimized kernels is controlled by `NT`:

```shell
make -j NT=36
```

This build parameter expands to compile-time thread-count macros such as:

```make
CPPFLAGS += -DNT=36
```

Use the same thread count at run time:

```shell
export OMP_NUM_THREADS=36
export CHEFSI_USE_OPT=1
```

## Run

Run the executable with MPI from a calculation directory:

```shell
mpirun -np <num_processes> /path/to/XLSDFT/src/a.out <input options>
```

Input files define the calculation parameters, atomic structure, and referenced
pseudopotentials. Detailed input documentation and example calculations are not
included in this source-only release.

## Output

XLSDFT writes results in the run directory. Depending on the calculation setup,
outputs can include SCF iteration logs, energies, forces, timing data, and
restart-related files.
