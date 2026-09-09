OBJSC = main.o args.o filesys.o tools.o geometry.o vertices.o\
		stencil.o stencil_default.o stencil_specialization.o stencil_sve.o stencil_sme.o index.o\
		linalg.o linalg_sve.o linalg_default.o arr.o atom.o control.o\
		scf.o poisson_solver.o parallel_vertices.o mixing.o\
		density_solver.o preparation.o effective_potential_loc_solver.o\
		exchange_correlation_solver.o pseudo_charge_solver.o\
		eigen_solver.o chefsi_double.o chefsi.o chefsi_opt.o chefsi_layout.o chefsi_layout_opt.o \
		laplacian_opt.o nloc_opt.o axpy_opt.o \
		sp_gemm_opt.o sp_syrk_opt.o sr_gemm_opt.o transpose.o \
		hamiltonian.o smearing.o\
		spin.o lanczos.o aar.o nloc_projector.o\
		effective_potential_nloc.o force_solver.o\
		density_matrix_solver.o xlsdft.o\
		wrapper_hypre.o wrapper_sstructmg.o \
		$(BACKEND_OBJSC)
