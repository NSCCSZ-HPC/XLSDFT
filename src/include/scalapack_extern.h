#ifndef _SCALAPACK_EXTERN_
#define _SCALAPACK_EXTERN_
#include <mpi.h>

extern "C" {

	#if (defined(USE_MKL) || defined(USE_SCALAPACK))
	int Csys2blacs_handle(MPI_Comm comm);
	void Cblacs_pinfo(int *mypnum, int *nprocs);
	void Cblacs_get(int ConTxt, int what, int *val);
	void Cblacs_gridinit(int *ConTxt, char *order, int nprow, int npcol);
	void Cblacs_gridinfo(int ConTxt, int *nprow, int *npcol, int *myrow, int *mycol);
	void Cblacs_gridexit(int ConTxt);
	void Cblacs_exit(int NotDone);
	void Cblacs_gridmap(int *ConTxt, int *usermap, int ldup, int nprow0, int npcol0);
	int Cblacs_pnum(int ConTxt, int prow, int pcol);
	void Cblacs_pcoord(int ConTxt, int nodenum, int *prow, int *pcol);
	void Cblacs_barrier(int ConTxt, char *scope);
	#endif

	#ifdef USE_MKL
		#include <mkl.h>
		#include <mkl_blacs.h>
		#include <mkl_pblas.h>
		#include <mkl_scalapack.h>
	#else
		#ifdef USE_SCALAPACK
		int numroc_(const int* n, const int* nb, const int* iproc,
    	            const int* isrcproc, const int* nprocs);
		void descinit_(int* desc, const int* m, const int* n,
    	           	   const int* mb, const int* nb, const int* irsrc,
    	           	   const int* icsrc, const int* ictxt, const int* lld,
    	           	   int* info);
		void psgemr2d_(const int *m, const int *n,
               		   const float *a, const int *ia, const int *ja, const int *desca,
               		   float *b, const int *ib, const int *jb, const int *descb,
               		   const int *ictxt);
		void pdgemr2d_(const int *m, const int *n,
               		   const double *a, const int *ia, const int *ja, const int *desca,
               		   double *b, const int *ib, const int *jb, const int *descb,
               		   const int *ictxt);
		void pssyrk_(const char *uplo, const char *trans, const int *n, const int *k,
					 const float *alpha, const float *a, const int *ia, const int *ja,
					 const int *desca, const float *beta, float *c, const int *ic,
					 const int *jc, const int *descc );
		void pdsyrk_(const char *uplo, const char *trans, const int *n, const int *k,
					 const double *alpha, const double *a, const int *ia, const int *ja,
					 const int *desca, const double *beta, double *c, const int *ic,
					 const int *jc, const int *descc );
		void psgemm_(const char *transa, const char *transb, const int *m, const int *n,
					 const int *k, const float *alpha, const float *a, const int *ia,
					 const int *ja, const int *desca, const float *b, const int *ib,
					 const int *jb, const int *descb, const float *beta, float *c,
					 const int *ic, const int *jc, const int *descc );
		void pdgemm_(const char *transa, const char *transb, const int *m, const int *n,
					 const int *k, const double *alpha, const double *a, const int *ia,
					 const int *ja, const int *desca, const double *b, const int *ib,
					 const int *jb, const int *descb, const double *beta, double *c,
					 const int *ic, const int *jc, const int *descc );
		void pssygvx_(const int* ibtype, const char* jobz, const char* range, const char* uplo,
					  const int* n, float* a, const int* ia, const int* ja, const int* desca,
					  float* b, const int* ib, const int* jb, const int* descb, const float* vl,
					  const float* vu, const int* il, const int* iu, const float* abstol,
					  int* m, int* nz, float* w, const float* orfac, float* z, const int* iz,
					  const int* jz, const int* descz, float* work, const int* lwork, int* iwork,
					  const int* liwork, int* ifail, int* iclustr, float* gap, int* info);
		void pdsygvx_(const int* ibtype, const char* jobz, const char* range, const char* uplo,
					  const int* n, double* a, const int* ia, const int* ja, const int* desca,
					  double* b, const int* ib, const int* jb, const int* descb, const double* vl,
					  const double* vu, const int* il, const int* iu, const double* abstol,
					  int* m, int* nz, double* w, const double* orfac, double* z, const int* iz,
					  const int* jz, const int* descz, double* work, const int* lwork, int* iwork,
					  const int* liwork, int* ifail, int* iclustr, double* gap, int* info);
		#endif
	#endif
}

#endif //_SCALAPACK_EXTERN_
