## XLSDFT Installation And Usage

### (1) Brief

XLSDFT is a real-space density functional theory software package for large-scale
electronic structure calculations. The code is parallelized with MPI and OpenMP
and includes real-space stencil kernels, eigensolver routines, density-matrix
solvers, Poisson solver interfaces, and force evaluation modules.

This release provides the source code and build configurations. Documentation,
examples, pseudopotentials, and validation cases may be added in later releases.

### (2) Package Contents

The current package contains:

* `src/` -- source code and build files.
* `src/include/` -- header files.
* `src/Makefile` -- main build script.
* `src/Makefile.config.LX2` -- LX2 build configuration.
* `src/Makefile.config.x86` -- x86 build configuration.
* `src/objects.mk` -- object list for the default executable.

### (3) Installation

Prerequisites:

* MPI C++ compiler wrapper, e.g. `mpicxx`.
* C++17 compiler support.
* OpenMP runtime.
* BLAS/LAPACK/ScaLAPACK-compatible numerical libraries.
* FFTW-compatible transform library when FFT-based paths are enabled.
* JSON library and multigrid solver library when multigrid support is enabled.
* HBM-related libraries when HBM support is enabled.

To build on LX2:

```shell
$ cd src
$ cp Makefile.config.LX2 Makefile.config
$ make clean; make -j
```

To build on x86:

```shell
$ cd src
$ cp Makefile.config.x86 Makefile.config
$ make clean; make -j
```

If dependency locations are different from the defaults, either edit
the temporary `Makefile.config` generated above, or override paths before
compiling:

```shell
$ export MKLROOT=/path/to/mkl
$ export OPENBLASROOT=/path/to/openblas
$ export LAPACKROOT=/path/to/lapack
$ export CBLASROOT=/path/to/cblas
$ export SCALAPACKROOT=/path/to/scalapack
$ export HBMROOT=/path/to/hbm
$ export SOFT_HOME=/path/to/dependencies
$ export JSON_HOME=/path/to/jsoncpp
$ export HYPRE_F64_HOME=/path/to/hypre
$ export SSTRUCTMG_HOME=/path/to/sstructmg
```

The executable is generated as:

```shell
src/a.out
```

`Makefile.config` is a local temporary build file generated from one of the
provided configuration templates. It is not part of the source package.

### (4) Input Files

Input files describe calculation parameters, atomic information, and referenced
pseudopotentials. Example inputs and detailed parameter documentation are not
included in this source-only release.

### (5) Execution

After compilation, run XLSDFT with MPI:

```shell
$ mpirun -np <num_processes> ./src/a.out <input options>
```

The exact input options depend on the calculation setup.

### (6) Output

Output is written in the run directory. Depending on the calculation type and
input settings, output may include SCF iteration data, energies, forces, timing
information, and restart-related files.
