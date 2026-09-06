#include "tools.h"

#include <mpi.h>
#include <vector>
#include <set>
uint Tools::get_node_number() {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char hostname[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(hostname, &name_len);

    int max_len;
    MPI_Allreduce(&name_len, &max_len, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    std::vector<char> all_names(world_size * (max_len+1));
    MPI_Gather(hostname, max_len+1, MPI_CHAR,
               all_names.data(), max_len+1, MPI_CHAR,
               0, MPI_COMM_WORLD);
    
    int node_number = 0;
    if (world_rank == 0) {
        std::set<std::string> unique_hosts;
        for (int i = 0; i < world_size; i++) {
            std::string h(&all_names[i*(max_len+1)]);
            unique_hosts.insert(h);
        }
        node_number = unique_hosts.size();
    }
    // MPI_Bcast(&node_number, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return node_number;
}

// #include <fstream>
// #ifdef USE_OPENMP
// #include <omp.h>
// #endif
// void Tools::print_rankfile() {
//     int world_rank, world_size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);

//     char hostname[MPI_MAX_PROCESSOR_NAME];
//     int name_len;
//     MPI_Get_processor_name(hostname, &name_len);

//     #ifdef USE_OPENMP

//     std::ofstream fout("rankfile.txt", std::ios::app);

//     #pragma omp parallel
//     {
//         int tid = omp_get_thread_num();
//         cpu_set_t mask;
//         CPU_ZERO(&mask);
//         sched_getaffinity(0, sizeof(mask), &mask);

//         int core = -1;
//         for (int i = 0; i < CPU_SETSIZE; i++) {
//             if (CPU_ISSET(i, &mask)) { core = i; break; }
//         }

//         #pragma omp critical
//         {
//             fout << "rank " << world_rank << ":" << tid
//                  << " = " << hostname
//                  << " slot=" << core
//                  << std::endl;
//         }
//     }

//     fout.close();

//     #else //USE_OPENMP

//     cpu_set_t mask;
//     CPU_ZERO(&mask);
//     sched_getaffinity(0, sizeof(mask), &mask);

//     int core = -1;
//     for (int i = 0; i < CPU_SETSIZE; i++) {
//         if (CPU_ISSET(i, &mask)) { core = i; break; }
//     }

//     std::ofstream fout("rankfile.txt", std::ios::app);
//     fout << "rank " << world_rank 
//          << " = " << hostname 
//          << " slot=" << core << std::endl;
//     fout.close();

//     #endif
//     return;
// }

int Tools::group_of(const int& i, const int& n) {
if (i >= 0)
    return i / n;
else
    // return (i - 1) / n;
    return - ((-i + 1) / n);
}

template<typename T>
void Tools::sort(T const* const& __restrict__ X, const int& len, T* const& __restrict__ Xsorted, int* const& __restrict__ Ind) {
    std::iota(Ind, Ind+len, 0);
    std::memcpy(Xsorted, X, sizeof(T)*len);
    std::stable_sort(Ind, Ind+len, [X](int i1, int i2) {return X[i1] < X[i2];});
    std::stable_sort(Xsorted, Xsorted+len);
    return;
}
template void Tools::sort<float>(float const* const& X, const int& len, float* const& Xsorted, int* const& Ind);
template void Tools::sort<double>(double const* const& X, const int& len, double* const& Xsorted, int* const& Ind);

/**
 * @brief Binary search to find which interval a given number 
 *        is located. list[...] is assumed to be monotonically
 *        increasing.
 * 
 * @ref https://github.com/SPARC-X/SPARC/blob/c1c792e66835b4f550f8c4eb2b1fe7e395a1e60e/src/tools.c#L653
 */
template<typename T1, typename T2>
int Tools::binary_interval_search(T1 const* const& __restrict__ list, const int len, const T2 x) {
    int first = 0;
	int last = len - 1;
	int middle = (first + last) / 2;
	while (first + 1 < last) {
		if (list[middle]<= x && x < list[middle+1]) {
			return middle;
		}
		if (list[middle] < x) {
			first = middle;
		} else {
			last = middle;
		}
		middle = (first + last)/2;
	}
	return first;
}
template int Tools::binary_interval_search<double, float>(double const* const& list, const int len, const float x);
template int Tools::binary_interval_search<double, double>(double const* const& list, const int len, const double x);

/**
 * @brief Solves a tridiagonal system using Gauss Elimination.
 * 
 * @ref https://github.com/SPARC-X/SPARC/blob/c1c792e66835b4f550f8c4eb2b1fe7e395a1e60e/src/tools.c#L413
 */
template<typename T> void Tools::tridiag_gen(T const* const& __restrict__ A, T const* const& __restrict__ B,
                                             T const* const& __restrict__ C, T* const __restrict__ D, const int& len) {
    int i;
    T b, *F;
    F = new T[len];
    assert(F != nullptr && "Memory allocation failed!\n");
    
    // Gauss elimination; forward substitution
    b = B[0];
    assert(fabs(b) >= 1e-14 && "Divide by zero in tridiag_gen");

    D[0] = D[0]/b;
    for (i = 1; i<len; i++) {
        F[i] = C[i-1] / b;
        b= B[i] - A[i] * F[i];
        assert(fabs(b) >= 1e-14 && "Divide by zero in tridiag_gen");
        D[i] = (D[i] - D[i-1] * A[i])/b;
    }
    // backsubstitution 
    for (i = len-2; i >= 0; i--)
        D[i] -= (D[i+1]*F[i+1]);

    delete [] F;
    return;
}
template void Tools::tridiag_gen<float>(float const* const& A, float const* const& B,  float const* const& C, float* const D, const int& len);
template void Tools::tridiag_gen<double>(double const* const& A, double const* const& B,  double const* const& C, double* const D, const int& len);

/**
 * @brief Calculates derivatives of a tabulated function required for spline interpolation.
 * 
 * @ref https://github.com/SPARC-X/SPARC/blob/c1c792e66835b4f550f8c4eb2b1fe7e395a1e60e/src/tools.c#L372
 */
template<typename T>
void Tools::getYD_gen(T const* const& __restrict__ X, T const* const& __restrict__ Y, T* const& __restrict__ YD, const int& len) {
    int i;
    T h0,h1,r0,r1,*A,*B,*C;

    A = new T[len];
    B = new T[len];
    C = new T[len];
    assert(A != nullptr && B != nullptr && C != nullptr && "Memory allocation failed!\n");

    h0 =  X[1]-X[0]; h1 = X[2]-X[1];
    r0 = (Y[1]-Y[0])/h0; r1=(Y[2]-Y[1])/h1;
    B[0] = h1*(h0+h1);
    C[0] = (h0+h1)*(h0+h1);
    YD[0] = r0*(3*h0*h1 + 2*h1*h1) + r1*h0*h0;

    for(i=1;i<len-1;i++) {
        h0 = X[i]-X[i-1]; h1=X[i+1]-X[i];
        r0 = (Y[i]-Y[i-1])/h0;  r1=(Y[i+1]-Y[i])/h1;
        A[i] = h1;
        B[i] = 2*(h0+h1);
        C[i] = h0;
        YD[i] = 3*(r0*h1 + r1*h0);
    }

    A[i] = (h0+h1)*(h0+h1);
    B[i] = h0*(h0+h1);
    YD[i] = r0*h1*h1 + r1*(3*h0*h1 + 2*h0*h0);

    Tools::tridiag_gen(A,B,C,YD,len);

    delete [] A;
    delete [] B;
    delete [] C;
    return;
}
template void Tools::getYD_gen<float>(float const* const& X, float const* const& Y, float* const& YD, const int& len);
template void Tools::getYD_gen<double>(double const* const& X, double const* const& Y, double* const& YD, const int& len);

/**
 * @brief Cubic spline evaluation from precalculated data.
 * 
 * @ref https://github.com/SPARC-X/SPARC/blob/c1c792e66835b4f550f8c4eb2b1fe7e395a1e60e/src/tools.c#L450
 */
template<typename T1, typename T2>
void Tools::SplineInterp(T1 const* const& __restrict__ X1, T1 const* const& __restrict__ Y1, const int& len1,
                         T2 const* const& __restrict__ X2, T2* const& __restrict__ Y2, const int& len2, T1 const* const& __restrict__ YD) {
    if (len2 <= 0) return;
    int i,j;
    T2 A0,A1,A2,A3,x,dx,dy,p1,p2,p3;    
    assert(!(X2[0]<X1[0] || X2[len2-1]>X1[len1-1]));
    // if(X2[0]<X1[0] || X2[len2-1]>X1[len1-1]) {
    //     printf("First input X in table=%lf, last input X in table=%lf, "
    //            "interpolate at x[first]=%lf, x[last]=%lf\n",X1[0],X1[len1-1],X2[0],X2[len2-1]);
    //     printf("Out of range in spline interpolation!\n");
    //     exit(EXIT_FAILURE);
    // }
    // p1 is left endpoint of the interval
    // p2 is resampling position
    // p3 is right endpoint of interval
    // j is input index of current interval  
    A0 = A1 = A2 = A3 = 0.0;
    p1 = p3 = X2[0]-1;  // force coefficient initialization  
    for (i = j = 0; i < len2; i++) {
        // check if in new interval
        p2 = X2[i];
        if (p2 > p3) {
            //find interval which contains p2 
            for (; j<len1 && p2>X1[j]; j++);
            if (p2 < X1[j]) j--;
            p1 = X1[j]; 
            p3 = X1[j+1]; 
            // coefficients
            dx = 1.0 / (X1[j+1] - X1[j]);
            dy = (Y1[j+1] - Y1[j]) * dx;
            A0 = Y1[j];
            A1 = YD[j];
            A2 = dx * (3.0 * dy - 2.0 * YD[j] - YD[j+1]);
            A3 = dx * dx * (-2.0*dy + YD[j] + YD[j+1]);  
        }
        // use Horner's rule to calculate cubic polynomial
        x = p2-p1;
        Y2[i] = ((A3*x + A2) * x + A1) * x + A0;     
    }
    return;
}
template void Tools::SplineInterp<double, float>(double const* const& X1, double const* const& Y1, const int& len1,
                                                 float const* const& X2, float* const& Y2, const int& len2, double const* const& YD);
template void Tools::SplineInterp<double, double>(double const* const& X1, double const* const& Y1, const int& len1,
                                                  double const* const& X2, double* const& Y2, const int& len2, double const* const& YD);

/**
 * @brief Sort input values and then apply spline interpolation.
 * 
 * @ref https://github.com/SPARC-X/SPARC/blob/c1c792e66835b4f550f8c4eb2b1fe7e395a1e60e/src/tools.c#L517
 */
template<typename T1, typename T2>
void Tools::SortSplineInterp(T1 const* const& __restrict__ X1, T1 const* const& __restrict__ Y1, const int& len1,
                             T2 const* const& __restrict__ X2, T2* const& __restrict__ Y2, const int& len2, T1 const* const& __restrict__ YD) {
    int i, *Ind_sort, len_interp;
    T2 *Y2_sort;
    
    // first sort X2 in ascending order
    // NOTE: we're using the memory of Y2 to store sorted X2 temporarily to save memory!
    Ind_sort = new int[len2];
    Tools::sort(X2, len2, Y2, Ind_sort);
    
    Y2_sort = new T2[len2];
    assert(Y2_sort != nullptr && "Memory allocation failed!");
    
    len_interp = len2;
    while (len_interp >= 1 && Y2[len_interp-1] > X1[len1-1]) len_interp--;
    
    // apply cubic spline interpolation to find Y2_sort, note here Y2 stores sorted values of X2
    Tools::SplineInterp(X1, Y1, len1, Y2, Y2_sort, len_interp, YD);
    
    // rearrange Y2_sort to original order
    for (i = 0; i < len_interp; i++) {
        Y2[Ind_sort[i]] = Y2_sort[i];
    }
    
    delete [] Y2_sort;
    delete [] Ind_sort;
    return;
}
template void Tools::SortSplineInterp<double, float>(double const* const& X1, double const* const& Y1, const int& len1,
                                                     float const* const& X2, float* const& Y2, const int& len2, double const* const& YD);
template void Tools::SortSplineInterp<double, double>(double const* const& X1, double const* const& Y1, const int& len1,
                                                      double const* const& X2, double* const& Y2, const int& len2, double const* const& YD);

/**
 * @brief Cubic spline evaluation from precalculated data. This assumes
 *        X1 is a uniformly increasing grid.
 * 
 * @ref https://github.com/SPARC-X/SPARC/blob/c1c792e66835b4f550f8c4eb2b1fe7e395a1e60e/src/tools.c#L552
 */
template<typename T1, typename T2>
void Tools::SplineInterpUniform(T1 const* const& __restrict__ X1, T1 const* const& __restrict__ Y1, const int& len1,
                                T2 const* const& __restrict__ X2, T2* const& __restrict__ Y2, const int& len2, T1 const* const& __restrict__ YD) {
    if (len2 <= 0 || len1 < 2) return;
	int i,j;
	T2 A0,A1,A2,A3,x,dx,dy,p1,p2,p3;
	// p1 is left endpoint of the interval
	// p2 is resampling position
	// p3 is right endpoint of interval
	// j is index of current interval  
	T2 X1_max = X1[len1-1];
	T2 delta_x1;
	delta_x1 = (X1[1] - X1[0]); // assuming len1 >= 2
	p3 = X2[0]-1;  // force coefficient initialization
	for (i = 0; i < len2; i++) {
		// check if in new interval
		p2 = X2[i];
		// assume X1 is a uniform grid with len1>=2
		if (p2 > X1_max) {
			j = len1 - 2;
			continue; // comment this to enable interpolation out of range
		} else {
			j = floor((p2 - X1[0]) / delta_x1);
		}
		// interval j = (X1[j], X1[j+1])
		p1 = X1[j];
		p3 = X1[j+1];
		// coefficients
		dx = 1.0 / (p3 - p1);
		dy = (Y1[j+1] - Y1[j]) * dx;
		A0 = Y1[j];
		A1 = YD[j];
		A2 = dx * (3.0 * dy - 2.0 * YD[j] - YD[j+1]);
		A3 = dx * dx * (-2.0*dy + YD[j] + YD[j+1]);
		// use Horner's rule to calculate cubic polynomial
		x = p2-p1;
		Y2[i] = ((A3*x + A2) * x + A1) * x + A0;
	}
    return;
}
template void Tools::SplineInterpUniform<double, float>(double const* const& X1, double const* const& Y1, const int& len1,
                                                        float const* const& X2, float* const& Y2, const int& len2, double const* const& YD);
template void Tools::SplineInterpUniform<double, double>(double const* const& X1, double const* const& Y1, const int& len1,
                                                         double const* const& X2, double* const& Y2, const int& len2, double const* const& YD);

/**
 * @brief Cubic spline evaluation from precalculated data. This function
 *        assumes X1 is a monotically increasing grid (but not necessarily
 *        uniform).
 *        This function runs faster than SortSplineInterp above in general.
 *        This function assumes that X1 is monotically increasing, and 
 *        therefore one can use a binary search approach to find the 
 *        interval any point in X2 is located.
 * 
 * @ref https://github.com/SPARC-X/SPARC/blob/c1c792e66835b4f550f8c4eb2b1fe7e395a1e60e/src/tools.c#L605 
 */
template<typename T1, typename T2>
void Tools::SplineInterpNonuniform(T1 const* const& __restrict__ X1, T1 const* const& __restrict__ Y1, const int& len1,
                                   T2 const* const& __restrict__ X2, T2* const& __restrict__ Y2, const int& len2, T1 const* const& __restrict__ YD) {
    Tools::SortSplineInterp(X1, Y1, len1, X2, Y2, len2, YD);
    return;

    if (len2 <= 0) return;
    int i,j;
    T2 A0,A1,A2,A3,x,dx,dy,p1,p2,p3;
    // p1 is left endpoint of the interval
    // p2 is resampling position
    // p3 is right endpoint of interval
    // j is input index of current interval  
    T2 X1_max = X1[len1-1];
    p3 = X2[0]-1;  // force coefficient initialization  
    for (i = 0; i < len2; i++) {
        // check if in new interval
        p2 = X2[i];
        if (p2 > X1_max) {
        	j = len1 - 2;
        	continue; // comment this to enable interpolation out of range
        } else {
        	j = Tools::binary_interval_search(X1, len1, p2);
        }
        p1 = X1[j];
        p3 = X1[j+1];
        // coefficients
        dx = 1.0 / (p3 - p1);
        dy = (Y1[j+1] - Y1[j]) * dx;
        A0 = Y1[j];
        A1 = YD[j];
        A2 = dx * (3.0 * dy - 2.0 * YD[j] - YD[j+1]);
        A3 = dx * dx * (-2.0*dy + YD[j] + YD[j+1]);
        // use Horner's rule to calculate cubic polynomial
        x = p2-p1;
        Y2[i] = ((A3*x + A2) * x + A1) * x + A0;
    } 
    return;
}
template void Tools::SplineInterpNonuniform<double, float>(double const* const& X1, double const* const& Y1, const int& len1,
                                                           float const* const& X2, float* const& Y2, const int& len2, double const* const& YD);
template void Tools::SplineInterpNonuniform<double, double>(double const* const& X1, double const* const& Y1, const int& len1,
                                                            double const* const& X2, double* const& Y2, const int& len2, double const* const& YD);

/**
 * @brief   Calculate real spherical harmonics for given positions and given l and m. 
 *
 *          Only for l = 0, 1, ..., 6.
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/tools.c#L1211
 */
template<typename T>
void Tools::RealSphericalHarmonic(T* const& __restrict__ Ylm, const int& l, const int& m, const int& length,
                                  T const* const& __restrict__ x, T const* const& __restrict__ y, T const* const& __restrict__ z,
                                  T const* const& __restrict__ r) {
    // only l=0,1,2,3,4,5,6 implemented for now

    //double pi=M_PI;
    T p;                   	  
    int i; 
    
    /* l = 0 */
    T C00 = 0.282094791773878; // 0.5*sqrt(1/pi)
    /* l = 1 */
    T C1m1 = 0.488602511902920; // sqrt(3/(4*pi))
    T C10 = 0.488602511902920; // sqrt(3/(4*pi))
    T C1p1 = 0.488602511902920; // sqrt(3/(4*pi))
    /* l = 2 */
    T C2m2 = 1.092548430592079; // 0.5*sqrt(15/pi)
    T C2m1 = 1.092548430592079; // 0.5*sqrt(15/pi)  
    T C20 =  0.315391565252520; // 0.25*sqrt(5/pi)
    T C2p1 = 1.092548430592079; // 0.5*sqrt(15/pi)  
    T C2p2 = 0.546274215296040; // 0.25*sqrt(15/pi)
    /* l = 3 */
    T C3m3 = 0.590043589926644; // 0.25*sqrt(35/(2*pi))   
    T C3m2 = 2.890611442640554; // 0.5*sqrt(105/(pi))
    T C3m1 = 0.457045799464466; // 0.25*sqrt(21/(2*pi))
    T C30 =  0.373176332590115; // 0.25*sqrt(7/pi)
    T C3p1 = 0.457045799464466; // 0.25*sqrt(21/(2*pi))
    T C3p2 = 1.445305721320277; //  0.25*sqrt(105/(pi))
    T C3p3 = 0.590043589926644; //  0.25*sqrt(35/(2*pi))
    /* l = 4 */
    T C4m4 = 2.503342941796705; // (3.0/4.0)*sqrt(35.0/pi)
    T C4m3 = 1.770130769779930; // (3.0/4.0)*sqrt(35.0/(2.0*pi))   
    T C4m2 = 0.946174695757560; // (3.0/4.0)*sqrt(5.0/pi)
    T C4m1 = 0.669046543557289; // (3.0/4.0)*sqrt(5.0/(2.0*pi))
    T C40 =  0.105785546915204; // (3.0/16.0)*sqrt(1.0/pi)
    T C4p1 = 0.669046543557289; // (3.0/4.0)*sqrt(5.0/(2.0*pi))
    T C4p2 = 0.473087347878780; // (3.0/8.0)*sqrt(5.0/(pi))
    T C4p3 = 1.770130769779930; // (3.0/4.0)*sqrt(35.0/(2.0*pi))
    T C4p4 = 0.625835735449176; // (3.0/16.0)*sqrt(35.0/pi) 
    /* l = 5 */
    T C5m5 = 0.656382056840170; // (3.0*sqrt(2.0*77.0/pi)/32.0)
    T C5m4 = 2.075662314881042; // (3.0/16.0)*sqrt(385.0/pi)
    T C5m3 = 0.489238299435250; // (sqrt(2.0*385.0/pi)/32.0)
    T C5m2 = 4.793536784973324; // (1.0/8.0)*sqrt(1155.0/pi)*2.0
    T C5m1 = 0.452946651195697; // (1.0/16.0)*sqrt(165.0/pi)
    T C50 =  0.116950322453424; // (1.0/16.0)*sqrt(11.0/pi)
    T C5p1 = 0.452946651195697; // (1.0/16.0)*sqrt(165.0/pi)
    T C5p2 = 2.396768392486662; // (1.0/8.0)*sqrt(1155.0/pi)
    T C5p3 = 0.489238299435250; // (sqrt(2.0*385.0/pi)/32.0)
    T C5p4 = 2.075662314881042; // (3.0/16.0)*sqrt(385.0/pi)
    T C5p5 = 0.656382056840170; // (3.0*sqrt(2.0)/32.0)*sqrt(77.0/pi)
    /* l = 6 */
    T C6m6 = 0.683184105191914; // (sqrt(2.0*3003.0/pi)/64.0)
    T C6m5 = 2.366619162231752; // (3.0/32.0)*sqrt(2.0*1001.0/pi)
    T C6m4 = 0.504564900728724; // (3.0/32.0)*sqrt(91.0/pi)
    T C6m3 = 0.921205259514923; // (sqrt(2.0*1365.0/pi)/32.0)
    T C6m2 = 0.460602629757462; // (sqrt(2.0*1365/pi)/64.0)
    T C6m1 = 0.582621362518731; // (sqrt(273.0/pi)/16.0)
    T C60 =  0.0635692022676284;// (sqrt(13.0/pi)/32.0)
    T C6p1 = 0.582621362518731; // (sqrt(273.0/pi)/16.0)
    T C6p2 = 0.460602629757462; // (sqrt(2.0*1365.0/pi)/64.0)
    T C6p3 = 0.921205259514923; // (sqrt(2.0*1365.0/pi)/32.0)
    T C6p4 = 0.504564900728724; // (3.0/32.0)*sqrt(91.0/pi)
    T C6p5 = 2.366619162231752; // (3.0/32.0)*sqrt(2.0*1001.0/pi)
    T C6p6 = 0.683184105191914; // (sqrt(2.0*3003.0/pi)/64.0)
    
    switch (l)
    {
        /* l = 0 */
        case 0:
            for (i = 0; i < length; i++) Ylm[i] = C00;
            break;
        
        /* l = 1 */
        case 1: 
            switch (m) 
            {
                case -1: /* m = -1 */
                    for (i = 0; i < length; i++) 
                        Ylm[i] = C1m1 * (y[i] / r[i]);
                    break;
                
                case 0: /* m = 0 */
                    for (i = 0; i < length; i++)
                        Ylm[i] = C10 * (z[i] / r[i]);
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < length; i++)
                        Ylm[i] = C1p1 * (x[i] / r[i]);
                    break;
                
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); break;
            }
            break;
        
        /* l = 2 */
        case 2: 
            switch (m) 
            {      
                case -2: /* m = -2 */
                    for (i = 0; i < length; i++)
                        Ylm[i] = C2m2 * (x[i]*y[i])/(r[i]*r[i]);
                    break;

                case -1: /* m = -1 */
                    for (i = 0; i < length; i++)
                        Ylm[i] = C2m1*(y[i]*z[i])/(r[i]*r[i]);
                    break;

                case 0: /* m = 0 */
                    for (i = 0; i < length; i++)
                        Ylm[i] = C20*(-x[i]*x[i] - y[i]*y[i] + 2.0*z[i]*z[i])/(r[i]*r[i]);
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < length; i++)
                        Ylm[i] = C2p1*(z[i]*x[i])/(r[i]*r[i]);
                    break;
                
                case 2: /* m = 2 */
                    for (i = 0; i < length; i++)
                        Ylm[i] = C2p2*(x[i]*x[i] - y[i]*y[i])/(r[i]*r[i]);
                    break;
                
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;

        /* l = 3 */
        case 3: 
            switch (m) 
            {   
                case -3: /* m = -3 */
                    for (i = 0; i < length; i++)
                        Ylm[i] = C3m3*(3*x[i]*x[i] - y[i]*y[i])*y[i]/(r[i]*r[i]*r[i]);
                    break;
               
                case -2: /* m = -2 */
                    for (i = 0; i < length; i++)
                        Ylm[i] = C3m2*(x[i]*y[i]*z[i])/(r[i]*r[i]*r[i]);
                    break;

                case -1: /* m = -1 */
                    for (i = 0; i < length; i++)
                        Ylm[i] = C3m1*y[i]*(4*z[i]*z[i] - x[i]*x[i] - y[i]*y[i])/(r[i]*r[i]*r[i]);
                    break;

                case 0: /* m = 0 */
                    for (i = 0; i < length; i++)
                        Ylm[i] = C30*z[i]*(2*z[i]*z[i]-3*x[i]*x[i]-3*y[i]*y[i])/(r[i]*r[i]*r[i]);
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < length; i++)
                        Ylm[i] = C3p1*x[i]*(4*z[i]*z[i] - x[i]*x[i] - y[i]*y[i])/(r[i]*r[i]*r[i]);
                    break;
                
                case 2: /* m = 2 */
                    for (i = 0; i < length; i++)
                        Ylm[i] = C3p2*z[i]*(x[i]*x[i] - y[i]*y[i])/(r[i]*r[i]*r[i]);
                    break;
                
                case 3: /* m = 3 */
                    for (i = 0; i < length; i++)
                        Ylm[i] = C3p3*x[i]*(x[i]*x[i]-3*y[i]*y[i])/(r[i]*r[i]*r[i]);
                    break;
                
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;
        
        /* l = 4 */
        case 4: 
            switch (m) 
            {     
                case -4: /* m = -4 */
                    for (i = 0; i < length; i++)
                        Ylm[i]=C4m4*(x[i]*y[i]*(x[i]*x[i]-y[i]*y[i]))/(r[i]*r[i]*r[i]*r[i]);
                    break;
                
                case -3: /* m = -3 */
                    for (i = 0; i < length; i++)
                        Ylm[i]=C4m3*(3.0*x[i]*x[i]-y[i]*y[i])*y[i]*z[i]/(r[i]*r[i]*r[i]*r[i]);
                    break;
               
                case -2: /* m = -2 */
                    for (i = 0; i < length; i++)
                        Ylm[i]=C4m2*x[i]*y[i]*(7.0*z[i]*z[i]-r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]);
                    break;

                case -1: /* m = -1 */
                    for (i = 0; i < length; i++)
                        Ylm[i]=C4m1*y[i]*z[i]*(7.0*z[i]*z[i]-3.0*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]);
                    break;

                case 0: /* m = 0 */
                    for (i = 0; i < length; i++)
                        Ylm[i]=C40*(35.0*z[i]*z[i]*z[i]*z[i]-30.0*z[i]*z[i]*r[i]*r[i]+3.0*r[i]*r[i]*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]);
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < length; i++)
                        Ylm[i]=C4p1*x[i]*z[i]*(7.0*z[i]*z[i]-3.0*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]);
                    break;
                
                case 2: /* m = 2 */
                    for (i = 0; i < length; i++)
                        Ylm[i]=C4p2*(x[i]*x[i]-y[i]*y[i])*(7.0*z[i]*z[i]-r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]);
                    break;
                
                case 3: /* m = 3 */
                    for (i = 0; i < length; i++)
                        Ylm[i]=C4p3*(x[i]*x[i]-3.0*y[i]*y[i])*x[i]*z[i]/(r[i]*r[i]*r[i]*r[i]);
                    break;
                
                case 4: /* m = 4 */
                    for (i = 0; i < length; i++)
                        Ylm[i]=C4p4*(x[i]*x[i]*(x[i]*x[i]-3.0*y[i]*y[i]) - y[i]*y[i]*(3.0*x[i]*x[i]-y[i]*y[i]))/(r[i]*r[i]*r[i]*r[i]);
                    break;
                    
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;
      
        /* l = 5 */
        case 5: 
            //p = sqrt(x[i]*x[i]+y[i]*y[i]);
            switch (m) 
            {   
                case -5: /* m = -5 */
                    for (i = 0; i < length; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5m5*(8.0*x[i]*x[i]*x[i]*x[i]*y[i]-4.0*x[i]*x[i]*y[i]*y[i]*y[i] + 4.0*pow(y[i],5)-3.0*y[i]*p*p*p*p)/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case -4: /* m = -4 */
                    for (i = 0; i < length; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5m4*(4.0*x[i]*x[i]*x[i]*y[i] - 4.0*x[i]*y[i]*y[i]*y[i])*z[i]/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case -3: /* m = -3 */
                    for (i = 0; i < length; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5m3*(3.0*y[i]*p*p - 4.0*y[i]*y[i]*y[i])*(9.0*z[i]*z[i]-r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
               
                case -2: /* m = -2 */
                    for (i = 0; i < length; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5m2*x[i]*y[i]*(3.0*z[i]*z[i]*z[i]-z[i]*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;

                case -1: /* m = -1 */
                    for (i = 0; i < length; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5m1*y[i]*(21.0*z[i]*z[i]*z[i]*z[i] - 14.0*r[i]*r[i]*z[i]*z[i]+r[i]*r[i]*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;

                case 0: /* m = 0 */
                    for (i = 0; i < length; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C50*(63.0*z[i]*z[i]*z[i]*z[i]*z[i] -70.0*z[i]*z[i]*z[i]*r[i]*r[i] + 15.0*z[i]*r[i]*r[i]*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < length; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5p1*x[i]*(21.0*z[i]*z[i]*z[i]*z[i] - 14.0*r[i]*r[i]*z[i]*z[i]+r[i]*r[i]*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 2: /* m = 2 */
                    for (i = 0; i < length; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5p2*(x[i]*x[i]-y[i]*y[i])*(3.0*z[i]*z[i]*z[i] - r[i]*r[i]*z[i])/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 3: /* m = 3 */
                    for (i = 0; i < length; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5p3*(4.0*x[i]*x[i]*x[i]-3.0*p*p*x[i])*(9.0*z[i]*z[i]-r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 4: /* m = 4 */
                    for (i = 0; i < length; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5p4*(4.0*(x[i]*x[i]*x[i]*x[i]+y[i]*y[i]*y[i]*y[i])-3.0*p*p*p*p)*z[i]/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 5: /* m = 5 */
                    for (i = 0; i < length; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C5p5*(4.0*x[i]*x[i]*x[i]*x[i]*x[i] + 8.0*x[i]*y[i]*y[i]*y[i]*y[i] -4.0*x[i]*x[i]*x[i]*y[i]*y[i] -3.0*x[i]*p*p*p*p)/(r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                    
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;

        /* l = 6 */
        case 6: 
            //p = sqrt(x[i]*x[i]+y[i]*y[i]);
            switch (m) 
            {   
                case -6: /* m = -6 */
                    for (i = 0; i < length; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6m6*(12.0*pow(x[i],5)*y[i]+12.0*x[i]*pow(y[i],5) - 8.0*x[i]*x[i]*x[i]*y[i]*y[i]*y[i]-6.0*x[i]*y[i]*pow(p,4))/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case -5: /* m = -5 */
                    for (i = 0; i < length; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6m5*(8.0*pow(x[i],4)*y[i] - 4.0*x[i]*x[i]*y[i]*y[i]*y[i] + 4.0*pow(y[i],5) -3.0*y[i]*pow(p,4))*z[i]/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case -4: /* m = -4 */
                    for (i = 0; i < length; i++)
                        Ylm[i] = C6m4*(4.0*x[i]*x[i]*x[i]*y[i] -4.0*x[i]*y[i]*y[i]*y[i])*(11.0*z[i]*z[i]-r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    break;
                
                case -3: /* m = -3 */
                    for (i = 0; i < length; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6m3*(-4.0*y[i]*y[i]*y[i] + 3.0*y[i]*p*p)*(11.0*z[i]*z[i]*z[i] - 3.0*z[i]*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
               
                case -2: /* m = -2 */
                    for (i = 0; i < length; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6m2*(2.0*x[i]*y[i])*(33.0*pow(z[i],4)-18.0*z[i]*z[i]*r[i]*r[i] + pow(r[i],4))/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;

                case -1: /* m = -1 */
                    for (i = 0; i < length; i++)
                        Ylm[i] = C6m1*y[i]*(33.0*pow(z[i],5)-30.0*z[i]*z[i]*z[i]*r[i]*r[i] +5.0*z[i]*pow(r[i],4))/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    break;

                case 0: /* m = 0 */
                    for (i = 0; i < length; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C60*(231.0*pow(z[i],6)-315*pow(z[i],4)*r[i]*r[i] + 105.0*z[i]*z[i]*pow(r[i],4) -5.0*pow(r[i],6))/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 1: /* m = 1 */
                    for (i = 0; i < length; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6p1*x[i]*(33.0*pow(z[i],5)-30.0*z[i]*z[i]*z[i]*r[i]*r[i] +5.0*z[i]*pow(r[i],4))/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 2: /* m = 2 */
                    for (i = 0; i < length; i++) {
                        //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6p2*(x[i]*x[i]-y[i]*y[i])*(33.0*pow(z[i],4) - 18.0*z[i]*z[i]*r[i]*r[i] + pow(r[i],4))/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 3: /* m = 3 */
                    for (i = 0; i < length; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6p3*(4.0*x[i]*x[i]*x[i] -3.0*x[i]*p*p)*(11.0*z[i]*z[i]*z[i] - 3.0*z[i]*r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 4: /* m = 4 */
                    for (i = 0; i < length; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6p4*(4.0*pow(x[i],4)+4.0*pow(y[i],4) -3.0*pow(p,4))*(11.0*z[i]*z[i] -r[i]*r[i])/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 5: /* m = 5 */
                    for (i = 0; i < length; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6p5*(4.0*pow(x[i],5) + 8.0*x[i]*pow(y[i],4)-4.0*x[i]*x[i]*x[i]*y[i]*y[i]-3.0*x[i]*pow(p,4))*z[i]/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                
                case 6: /* m = 6 */
                    for (i = 0; i < length; i++) {
                        p = sqrt(x[i]*x[i]+y[i]*y[i]);
                        Ylm[i] = C6p6*(4.0*pow(x[i],6)-4.0*pow(y[i],6) +12.0*x[i]*x[i]*pow(y[i],4)-12.0*pow(x[i],4)*y[i]*y[i] + 3.0*y[i]*y[i]*pow(p,4)-3.0*x[i]*x[i]*pow(p,4))/(r[i]*r[i]*r[i]*r[i]*r[i]*r[i]);
                    }
                    break;
                    
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;
        
        default: printf("<l> must be an integer between 0 and 6!\n"); break;
    }
    
    if (l > 0) {
        for (i = 0; i < length; i++) {
            if (r[i] < 1e-10) Ylm[i] = 0.0;
        }
    }
    return;
}
template void Tools::RealSphericalHarmonic<float>(float* const& Ylm, const int& l, const int& m, const int& length,
                                                  float const* const& x, float const* const& y, float const* const& z,
                                                  float const* const& r);
template void Tools::RealSphericalHarmonic<double>(double* const& Ylm, const int& l, const int& m, const int& length,
                                                   double const* const& x, double const* const& y, double const* const& z,
                                                   double const* const& r);

/**
 * @brief   Calculate real spherical harmonics for given positions and given l and m. 
 *
 *          Only for l = 0, 1, ..., 6.
 * @ref https://github.com/xuqimen/SPARC/blob/c92931afaa4c4cbef1bbc90f7db49e2a259b8f90/src/tools.c#L1211
 */
template<typename T>
void Tools::RealSphericalHarmonic2(T* const& __restrict__ Ylm, const int& l, const int& m, const int& length,
                                   T const* const& __restrict__ x_r, T const* const& __restrict__ y_r, T const* const& __restrict__ z_r,
                                   T const* const& __restrict__ r) {
    // only l=0,1,2,3,4,5,6 implemented for now

    //double pi=M_PI;
    // T p;
    T p2;
    int i;
    
    // /* l = 0 */
    // T C00 = 0.282094791773878; // 0.5*sqrt(1/pi)
    // /* l = 1 */
    // T C1m1 = 0.488602511902920; // sqrt(3/(4*pi))
    // T C10 = 0.488602511902920; // sqrt(3/(4*pi))
    // T C1p1 = 0.488602511902920; // sqrt(3/(4*pi))
    // /* l = 2 */
    // T C2m2 = 1.092548430592079; // 0.5*sqrt(15/pi)
    // T C2m1 = 1.092548430592079; // 0.5*sqrt(15/pi)  
    // T C20 =  0.315391565252520; // 0.25*sqrt(5/pi)
    // T C2p1 = 1.092548430592079; // 0.5*sqrt(15/pi)  
    // T C2p2 = 0.546274215296040; // 0.25*sqrt(15/pi)
    // /* l = 3 */
    // T C3m3 = 0.590043589926644; // 0.25*sqrt(35/(2*pi))   
    // T C3m2 = 2.890611442640554; // 0.5*sqrt(105/(pi))
    // T C3m1 = 0.457045799464466; // 0.25*sqrt(21/(2*pi))
    // T C30 =  0.373176332590115; // 0.25*sqrt(7/pi)
    // T C3p1 = 0.457045799464466; // 0.25*sqrt(21/(2*pi))
    // T C3p2 = 1.445305721320277; //  0.25*sqrt(105/(pi))
    // T C3p3 = 0.590043589926644; //  0.25*sqrt(35/(2*pi))
    // /* l = 4 */
    // T C4m4 = 2.503342941796705; // (3.0/4.0)*sqrt(35.0/pi)
    // T C4m3 = 1.770130769779930; // (3.0/4.0)*sqrt(35.0/(2.0*pi))   
    // T C4m2 = 0.946174695757560; // (3.0/4.0)*sqrt(5.0/pi)
    // T C4m1 = 0.669046543557289; // (3.0/4.0)*sqrt(5.0/(2.0*pi))
    // T C40 =  0.105785546915204; // (3.0/16.0)*sqrt(1.0/pi)
    // T C4p1 = 0.669046543557289; // (3.0/4.0)*sqrt(5.0/(2.0*pi))
    // T C4p2 = 0.473087347878780; // (3.0/8.0)*sqrt(5.0/(pi))
    // T C4p3 = 1.770130769779930; // (3.0/4.0)*sqrt(35.0/(2.0*pi))
    // T C4p4 = 0.625835735449176; // (3.0/16.0)*sqrt(35.0/pi) 
    // /* l = 5 */
    // T C5m5 = 0.656382056840170; // (3.0*sqrt(2.0*77.0/pi)/32.0)
    // T C5m4 = 2.075662314881042; // (3.0/16.0)*sqrt(385.0/pi)
    // T C5m3 = 0.489238299435250; // (sqrt(2.0*385.0/pi)/32.0)
    // T C5m2 = 4.793536784973324; // (1.0/8.0)*sqrt(1155.0/pi)*2.0
    // T C5m1 = 0.452946651195697; // (1.0/16.0)*sqrt(165.0/pi)
    // T C50 =  0.116950322453424; // (1.0/16.0)*sqrt(11.0/pi)
    // T C5p1 = 0.452946651195697; // (1.0/16.0)*sqrt(165.0/pi)
    // T C5p2 = 2.396768392486662; // (1.0/8.0)*sqrt(1155.0/pi)
    // T C5p3 = 0.489238299435250; // (sqrt(2.0*385.0/pi)/32.0)
    // T C5p4 = 2.075662314881042; // (3.0/16.0)*sqrt(385.0/pi)
    // T C5p5 = 0.656382056840170; // (3.0*sqrt(2.0)/32.0)*sqrt(77.0/pi)
    // /* l = 6 */
    // T C6m6 = 0.683184105191914; // (sqrt(2.0*3003.0/pi)/64.0)
    // T C6m5 = 2.366619162231752; // (3.0/32.0)*sqrt(2.0*1001.0/pi)
    // T C6m4 = 0.504564900728724; // (3.0/32.0)*sqrt(91.0/pi)
    // T C6m3 = 0.921205259514923; // (sqrt(2.0*1365.0/pi)/32.0)
    // T C6m2 = 0.460602629757462; // (sqrt(2.0*1365/pi)/64.0)
    // T C6m1 = 0.582621362518731; // (sqrt(273.0/pi)/16.0)
    // T C60 =  0.0635692022676284;// (sqrt(13.0/pi)/32.0)
    // T C6p1 = 0.582621362518731; // (sqrt(273.0/pi)/16.0)
    // T C6p2 = 0.460602629757462; // (sqrt(2.0*1365.0/pi)/64.0)
    // T C6p3 = 0.921205259514923; // (sqrt(2.0*1365.0/pi)/32.0)
    // T C6p4 = 0.504564900728724; // (3.0/32.0)*sqrt(91.0/pi)
    // T C6p5 = 2.366619162231752; // (3.0/32.0)*sqrt(2.0*1001.0/pi)
    // T C6p6 = 0.683184105191914; // (sqrt(2.0*3003.0/pi)/64.0)
    
    switch (l)
    {
        /* l = 0 */
        case 0: 
            {
                T C00 = 0.282094791773878; // 0.5*sqrt(1/pi)
                for (i = 0; i < length; i++) Ylm[i] = C00;
                break;
            }
        
        /* l = 1 */
        case 1: 
            switch (m) 
            {
                case -1: /* m = -1 */
                    {
                        T C1m1 = 0.488602511902920; // sqrt(3/(4*pi))
                        for (i = 0; i < length; i++) 
                            Ylm[i] = C1m1 * (y_r[i]);
                        break;
                    }
                
                case 0:  /* m = 0 */
                    {
                        T C10 = 0.488602511902920; // sqrt(3/(4*pi))
                        for (i = 0; i < length; i++)
                            Ylm[i] = C10 * (z_r[i]);
                        break;
                    }
                
                case 1: /* m = 1 */
                    {
                        T C1p1 = 0.488602511902920; // sqrt(3/(4*pi))
                        for (i = 0; i < length; i++)
                            Ylm[i] = C1p1 * (x_r[i]);
                        break;
                    }
                
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); break;
            }
            break;
        
        /* l = 2 */
        case 2: 
            switch (m) 
            {      
                case -2: /* m = -2 */
                    {
                        T C2m2 = 1.092548430592079; // 0.5*sqrt(15/pi)
                        for (i = 0; i < length; i++)
                            Ylm[i] = C2m2 * (x_r[i]*y_r[i]);
                        break;
                    }

                case -1: /* m = -1 */
                    {
                        T C2m1 = 1.092548430592079; // 0.5*sqrt(15/pi)
                        for (i = 0; i < length; i++)
                            Ylm[i] = C2m1*(y_r[i]*z_r[i]);
                        break;
                    }

                case 0: /* m = 0 */
                    {
                        T C20 =  0.315391565252520; // 0.25*sqrt(5/pi)
                        for (i = 0; i < length; i++)
                            Ylm[i] = C20*(-x_r[i]*x_r[i] - y_r[i]*y_r[i] + (T)2.0*z_r[i]*z_r[i]);
                        break;
                    }
                
                case 1: /* m = 1 */
                    {
                        T C2p1 = 1.092548430592079; // 0.5*sqrt(15/pi)
                        for (i = 0; i < length; i++)
                            Ylm[i] = C2p1*(z_r[i]*x_r[i]);
                        break;
                    }
                
                case 2: /* m = 2 */
                    {
                        T C2p2 = 0.546274215296040; // 0.25*sqrt(15/pi)
                        for (i = 0; i < length; i++)
                            Ylm[i] = C2p2*(x_r[i]*x_r[i] - y_r[i]*y_r[i]);
                        break;
                    }
                
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;

        /* l = 3 */
        case 3: 
            switch (m) 
            {   
                case -3: /* m = -3 */
                    {
                        T C3m3 = 0.590043589926644; // 0.25*sqrt(35/(2*pi))   
                        for (i = 0; i < length; i++)
                            Ylm[i] = C3m3*((T)3.0*x_r[i]*x_r[i] - y_r[i]*y_r[i])*y_r[i];
                        break;
                    }
               
                case -2: /* m = -2 */
                    {
                        T C3m2 = 2.890611442640554; // 0.5*sqrt(105/(pi))
                        for (i = 0; i < length; i++)
                            Ylm[i] = C3m2*(x_r[i]*y_r[i]*z_r[i]);
                        break;
                    }

                case -1: /* m = -1 */
                    {
                        T C3m1 = 0.457045799464466; // 0.25*sqrt(21/(2*pi))
                        for (i = 0; i < length; i++)
                            Ylm[i] = C3m1*y_r[i]*((T)4.0*z_r[i]*z_r[i] - x_r[i]*x_r[i] - y_r[i]*y_r[i]);
                        break;
                    }

                case 0: /* m = 0 */
                    {
                        T C30 =  0.373176332590115; // 0.25*sqrt(7/pi)
                        for (i = 0; i < length; i++)
                            Ylm[i] = C30*z_r[i]*((T)2.0*z_r[i]*z_r[i]-(T)3.0*x_r[i]*x_r[i]-(T)3.0*y_r[i]*y_r[i]);
                        break;
                    }
                
                case 1: /* m = 1 */
                    {
                        T C3p1 = 0.457045799464466; // 0.25*sqrt(21/(2*pi))
                        for (i = 0; i < length; i++)
                            Ylm[i] = C3p1*x_r[i]*((T)4.0*z_r[i]*z_r[i] - x_r[i]*x_r[i] - y_r[i]*y_r[i]);
                        break;
                    }
                
                case 2: /* m = 2 */
                    {
                        T C3p2 = 1.445305721320277; //  0.25*sqrt(105/(pi))
                        for (i = 0; i < length; i++)
                            Ylm[i] = C3p2*z_r[i]*(x_r[i]*x_r[i] - y_r[i]*y_r[i]);
                        break;
                    }
                    
                
                case 3: /* m = 3 */
                    {
                        T C3p3 = 0.590043589926644; //  0.25*sqrt(35/(2*pi))
                        for (i = 0; i < length; i++)
                            Ylm[i] = C3p3*x_r[i]*(x_r[i]*x_r[i]-(T)3.0*y_r[i]*y_r[i]);
                        break;
                    }
                
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;
        
        /* l = 4 */
        case 4: 
            switch (m) 
            {     
                case -4: /* m = -4 */
                    {
                        T C4m4 = 2.503342941796705; // (3.0/4.0)*sqrt(35.0/pi)
                        for (i = 0; i < length; i++)
                            Ylm[i]=C4m4*(x_r[i]*y_r[i]*(x_r[i]*x_r[i]-y_r[i]*y_r[i]));
                        break;
                    }
                
                case -3: /* m = -3 */
                    {
                        T C4m3 = 1.770130769779930; // (3.0/4.0)*sqrt(35.0/(2.0*pi))
                        for (i = 0; i < length; i++)
                            Ylm[i]=C4m3*((T)3.0*x_r[i]*x_r[i]-y_r[i]*y_r[i])*y_r[i]*z_r[i];
                        break;
                    }
               
                case -2: /* m = -2 */
                    {
                        T C4m2 = 0.946174695757560; // (3.0/4.0)*sqrt(5.0/pi)
                        for (i = 0; i < length; i++)
                            Ylm[i]=C4m2*x_r[i]*y_r[i]*((T)7.0*z_r[i]*z_r[i]-(T)1.0);
                        break;
                    }

                case -1: /* m = -1 */
                    {
                        T C4m1 = 0.669046543557289; // (3.0/4.0)*sqrt(5.0/(2.0*pi))
                        for (i = 0; i < length; i++)
                            Ylm[i]=C4m1*y_r[i]*z_r[i]*((T)7.0*z_r[i]*z_r[i]-(T)3.0);
                        break;
                    }

                case 0: /* m = 0 */
                    {
                        T C40 =  0.105785546915204; // (3.0/16.0)*sqrt(1.0/pi)
                        for (i = 0; i < length; i++)
                            Ylm[i]=C40*((T)35.0*z_r[i]*z_r[i]*z_r[i]*z_r[i]-(T)30.0*z_r[i]*z_r[i]+(T)3.0);
                        break;
                    }
                
                case 1: /* m = 1 */
                    {
                        T C4p1 = 0.669046543557289; // (3.0/4.0)*sqrt(5.0/(2.0*pi))
                        for (i = 0; i < length; i++)
                            Ylm[i]=C4p1*x_r[i]*z_r[i]*((T)7.0*z_r[i]*z_r[i]-(T)3.0);
                        break;
                    }
                
                case 2: /* m = 2 */
                    {
                        T C4p2 = 0.473087347878780; // (3.0/8.0)*sqrt(5.0/(pi))
                        for (i = 0; i < length; i++)
                            Ylm[i]=C4p2*(x_r[i]*x_r[i]-y_r[i]*y_r[i])*((T)7.0*z_r[i]*z_r[i]-(T)1.0);
                        break;
                    }
                
                case 3: /* m = 3 */
                    {
                        T C4p3 = 1.770130769779930; // (3.0/4.0)*sqrt(35.0/(2.0*pi))
                        for (i = 0; i < length; i++)
                            Ylm[i]=C4p3*(x_r[i]*x_r[i]-(T)3.0*y_r[i]*y_r[i])*x_r[i]*z_r[i];
                        break;
                    }
                
                case 4: /* m = 4 */
                    {
                        T C4p4 = 0.625835735449176; // (3.0/16.0)*sqrt(35.0/pi)
                        for (i = 0; i < length; i++)
                            Ylm[i]=C4p4*(x_r[i]*x_r[i]*(x_r[i]*x_r[i]-(T)3.0*y_r[i]*y_r[i]) - y_r[i]*y_r[i]*((T)3.0*x_r[i]*x_r[i]-y_r[i]*y_r[i]));
                        break;
                    }
                    
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;
      
        /* l = 5 */
        case 5: 
            //p = sqrt(x[i]*x[i]+y[i]*y[i]);
            switch (m) 
            {   
                case -5: /* m = -5 */
                    {
                        T C5m5 = 0.656382056840170; // (3.0*sqrt(2.0*77.0/pi)/32.0)
                        for (i = 0; i < length; i++) {
                            // p = sqrt(x_r[i]*x_r[i]+y_r[i]*y_r[i]);
                            // Ylm[i] = C5m5*((T)8.0*x_r[i]*x_r[i]*x_r[i]*x_r[i]*y_r[i]-(T)4.0*x_r[i]*x_r[i]*y_r[i]*y_r[i]*y_r[i] + (T)4.0*pow(y_r[i],5)-(T)3.0*y_r[i]*p*p*p*p);
                            p2 = x_r[i]*x_r[i]+y_r[i]*y_r[i];
                            Ylm[i] = C5m5*((T)8.0*x_r[i]*x_r[i]*x_r[i]*x_r[i]*y_r[i]-(T)4.0*x_r[i]*x_r[i]*y_r[i]*y_r[i]*y_r[i] + (T)4.0*pow(y_r[i],5)-(T)3.0*y_r[i]*p2*p2);
                        }
                        break;
                    }
                
                case -4: /* m = -4 */
                    {
                        T C5m4 = 2.075662314881042; // (3.0/16.0)*sqrt(385.0/pi)
                        for (i = 0; i < length; i++) {
                            //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                            Ylm[i] = C5m4*((T)4.0*x_r[i]*x_r[i]*x_r[i]*y_r[i] - (T)4.0*x_r[i]*y_r[i]*y_r[i]*y_r[i])*z_r[i];
                        }
                        break;
                    }
                
                case -3: /* m = -3 */
                    {
                        T C5m3 = 0.489238299435250; // (sqrt(2.0*385.0/pi)/32.0)
                        for (i = 0; i < length; i++) {
                            // p = sqrt(x_r[i]*x_r[i]+y_r[i]*y_r[i]);
                            // Ylm[i] = C5m3*((T)3.0*y_r[i]*p*p - (T)4.0*y_r[i]*y_r[i]*y_r[i])*((T)9.0*z_r[i]*z_r[i]-(T)1.0);
                            p2 = x_r[i]*x_r[i]+y_r[i]*y_r[i];
                            Ylm[i] = C5m3*((T)3.0*y_r[i]*p2 - (T)4.0*y_r[i]*y_r[i]*y_r[i])*((T)9.0*z_r[i]*z_r[i]-(T)1.0);
                        }
                        break;
                    }
               
                case -2: /* m = -2 */
                    {
                        T C5m2 = 4.793536784973324; // (1.0/8.0)*sqrt(1155.0/pi)*2.0
                        for (i = 0; i < length; i++) {
                            //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                            Ylm[i] = C5m2*x_r[i]*y_r[i]*((T)3.0*z_r[i]*z_r[i]*z_r[i]-z_r[i]);
                        }
                        break;
                    }

                case -1: /* m = -1 */
                    {
                        T C5m1 = 0.452946651195697; // (1.0/16.0)*sqrt(165.0/pi)
                        for (i = 0; i < length; i++) {
                            //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                            Ylm[i] = C5m1*y_r[i]*((T)21.0*z_r[i]*z_r[i]*z_r[i]*z_r[i] - (T)14.0*z_r[i]*z_r[i] + (T)1.0);
                        }
                        break;
                    }

                case 0: /* m = 0 */
                    {
                        T C50 =  0.116950322453424; // (1.0/16.0)*sqrt(11.0/pi)
                        for (i = 0; i < length; i++) {
                            //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                            Ylm[i] = C50*((T)63.0*z_r[i]*z_r[i]*z_r[i]*z_r[i]*z_r[i] -(T)70.0*z_r[i]*z_r[i]*z_r[i] + (T)15.0*z_r[i]);
                        }
                        break;
                    }
                
                case 1: /* m = 1 */
                    {
                        T C5p1 = 0.452946651195697; // (1.0/16.0)*sqrt(165.0/pi)
                        for (i = 0; i < length; i++) {
                            //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                            Ylm[i] = C5p1*x_r[i]*((T)21.0*z_r[i]*z_r[i]*z_r[i]*z_r[i] - (T)14.0*z_r[i]*z_r[i] + (T)1.0);
                        }
                        break;
                    }
                
                case 2: /* m = 2 */
                    {
                        T C5p2 = 2.396768392486662; // (1.0/8.0)*sqrt(1155.0/pi)
                        for (i = 0; i < length; i++) {
                            //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                            Ylm[i] = C5p2*(x_r[i]*x_r[i]-y_r[i]*y_r[i])*((T)3.0*z_r[i]*z_r[i]*z_r[i] - z_r[i]);
                        }
                        break;
                    }
                
                case 3: /* m = 3 */
                    {
                        T C5p3 = 0.489238299435250; // (sqrt(2.0*385.0/pi)/32.0)
                        for (i = 0; i < length; i++) {
                            // p = sqrt(x_r[i]*x_r[i]+y_r[i]*y_r[i]);
                            // Ylm[i] = C5p3*((T)4.0*x_r[i]*x_r[i]*x_r[i]-(T)3.0*p*p*x_r[i])*((T)9.0*z_r[i]*z_r[i] - (T)1.0);
                            p2 = x_r[i]*x_r[i]+y_r[i]*y_r[i];
                            Ylm[i] = C5p3*((T)4.0*x_r[i]*x_r[i]*x_r[i]-(T)3.0*p2*x_r[i])*((T)9.0*z_r[i]*z_r[i] - (T)1.0);
                        }
                        break;
                    }
                
                case 4: /* m = 4 */
                    {
                        T C5p4 = 2.075662314881042; // (3.0/16.0)*sqrt(385.0/pi)
                        for (i = 0; i < length; i++) {
                            // p = sqrt(x_r[i]*x_r[i]+y_r[i]*y_r[i]);
                            // Ylm[i] = C5p4*((T)4.0*(x_r[i]*x_r[i]*x_r[i]*x_r[i]+y_r[i]*y_r[i]*y_r[i]*y_r[i])-(T)3.0*p*p*p*p)*z_r[i];
                            p2 = x_r[i]*x_r[i]+y_r[i]*y_r[i];
                            Ylm[i] = C5p4*((T)4.0*(x_r[i]*x_r[i]*x_r[i]*x_r[i]+y_r[i]*y_r[i]*y_r[i]*y_r[i])-(T)3.0*p2*p2)*z_r[i];
                        }
                        break;
                    }
                
                case 5: /* m = 5 */
                    {
                        T C5p5 = 0.656382056840170; // (3.0*sqrt(2.0)/32.0)*sqrt(77.0/pi)
                        for (i = 0; i < length; i++) {
                            // p = sqrt(x_r[i]*x_r[i]+y_r[i]*y_r[i]);
                            // Ylm[i] = C5p5*((T)4.0*x_r[i]*x_r[i]*x_r[i]*x_r[i]*x_r[i] + (T)8.0*x_r[i]*y_r[i]*y_r[i]*y_r[i]*y_r[i] -(T)4.0*x_r[i]*x_r[i]*x_r[i]*y_r[i]*y_r[i] -(T)3.0*x_r[i]*p*p*p*p);
                            p2 = x_r[i]*x_r[i]+y_r[i]*y_r[i];
                            Ylm[i] = C5p5*((T)4.0*x_r[i]*x_r[i]*x_r[i]*x_r[i]*x_r[i] + (T)8.0*x_r[i]*y_r[i]*y_r[i]*y_r[i]*y_r[i] -(T)4.0*x_r[i]*x_r[i]*x_r[i]*y_r[i]*y_r[i] -(T)3.0*x_r[i]*p2*p2);
                        }
                        break;
                    }
                    
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;

        /* l = 6 */
        case 6: 
            //p = sqrt(x[i]*x[i]+y[i]*y[i]);
            switch (m) 
            {   
                case -6: /* m = -6 */
                    {
                        T C6m6 = 0.683184105191914; // (sqrt(2.0*3003.0/pi)/64.0)
                        for (i = 0; i < length; i++) {
                            // p = sqrt(x_r[i]*x_r[i]+y_r[i]*y_r[i]);
                            // Ylm[i] = C6m6*((T)12.0*pow(x_r[i],5)*y_r[i]+(T)12.0*x_r[i]*pow(y_r[i],5) - (T)8.0*x_r[i]*x_r[i]*x_r[i]*y_r[i]*y_r[i]*y_r[i]-(T)6.0*x_r[i]*y_r[i]*pow(p,4));
                            p2 = x_r[i]*x_r[i]+y_r[i]*y_r[i];
                            Ylm[i] = C6m6*((T)12.0*pow(x_r[i],5)*y_r[i]+(T)12.0*x_r[i]*pow(y_r[i],5) - (T)8.0*x_r[i]*x_r[i]*x_r[i]*y_r[i]*y_r[i]*y_r[i]-(T)6.0*x_r[i]*y_r[i]*pow(p2,2));
                        }
                        break;
                    }
                
                case -5: /* m = -5 */
                    {
                        T C6m5 = 2.366619162231752; // (3.0/32.0)*sqrt(2.0*1001.0/pi)
                        for (i = 0; i < length; i++) {
                            // p = sqrt(x_r[i]*x_r[i]+y_r[i]*y_r[i]);
                            // Ylm[i] = C6m5*((T)8.0*pow(x_r[i],4)*y_r[i] - (T)4.0*x_r[i]*x_r[i]*y_r[i]*y_r[i]*y_r[i] + (T)4.0*pow(y_r[i],5) -(T)3.0*y_r[i]*pow(p,4))*z_r[i];
                            p2 = x_r[i]*x_r[i]+y_r[i]*y_r[i];
                            Ylm[i] = C6m5*((T)8.0*pow(x_r[i],4)*y_r[i] - (T)4.0*x_r[i]*x_r[i]*y_r[i]*y_r[i]*y_r[i] + (T)4.0*pow(y_r[i],5) -(T)3.0*y_r[i]*pow(p2,2))*z_r[i];
                        }
                        break;
                    }
                
                case -4: /* m = -4 */
                    {
                        T C6m4 = 0.504564900728724; // (3.0/32.0)*sqrt(91.0/pi)
                        for (i = 0; i < length; i++)
                            Ylm[i] = C6m4*((T)4.0*x_r[i]*x_r[i]*x_r[i]*y_r[i] -(T)4.0*x_r[i]*y_r[i]*y_r[i]*y_r[i])*((T)11.0*z_r[i]*z_r[i]-(T)1.0);
                        break;
                    }
                
                case -3: /* m = -3 */
                    {
                        T C6m3 = 0.921205259514923; // (sqrt(2.0*1365.0/pi)/32.0)
                        for (i = 0; i < length; i++) {
                            // p = sqrt(x_r[i]*x_r[i]+y_r[i]*y_r[i]);
                            // Ylm[i] = C6m3*(-(T)4.0*y_r[i]*y_r[i]*y_r[i] + (T)3.0*y_r[i]*p*p)*((T)11.0*z_r[i]*z_r[i]*z_r[i] - (T)3.0*z_r[i]);
                            p2 = x_r[i]*x_r[i]+y_r[i]*y_r[i];
                            Ylm[i] = C6m3*(-(T)4.0*y_r[i]*y_r[i]*y_r[i] + (T)3.0*y_r[i]*p2)*((T)11.0*z_r[i]*z_r[i]*z_r[i] - (T)3.0*z_r[i]);
                        }
                        break;
                    }
               
                case -2: /* m = -2 */
                    {
                        T C6m2 = 0.460602629757462; // (sqrt(2.0*1365/pi)/64.0)
                        for (i = 0; i < length; i++) {
                            //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                            Ylm[i] = C6m2*((T)2.0*x_r[i]*y_r[i])*((T)33.0*pow(z_r[i],4)-(T)18.0*z_r[i]*z_r[i] + (T)1.0);
                        }
                        break;
                    }

                case -1: /* m = -1 */
                    {
                        T C6m1 = 0.582621362518731; // (sqrt(273.0/pi)/16.0)
                        for (i = 0; i < length; i++)
                            Ylm[i] = C6m1*y_r[i]*((T)33.0*pow(z_r[i],5)-(T)30.0*z_r[i]*z_r[i]*z_r[i] +(T)5.0*z_r[i]);
                        break;
                    }

                case 0: /* m = 0 */
                    {
                        T C60 =  0.0635692022676284;// (sqrt(13.0/pi)/32.0)
                        for (i = 0; i < length; i++) {
                            //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                            Ylm[i] = C60*((T)231.0*pow(z_r[i],6)-(T)315.0*pow(z_r[i],4) + (T)105.0*z_r[i]*z_r[i] -(T)5.0);
                        }
                        break;
                    }
                
                case 1: /* m = 1 */
                    {
                        T C6p1 = 0.582621362518731; // (sqrt(273.0/pi)/16.0)
                        for (i = 0; i < length; i++) {
                            //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                            Ylm[i] = C6p1*x_r[i]*((T)33.0*pow(z_r[i],5)-(T)30.0*z_r[i]*z_r[i]*z_r[i] +(T)5.0*z_r[i]);
                        }
                        break;
                    }
                
                case 2: /* m = 2 */
                    {
                        T C6p2 = 0.460602629757462; // (sqrt(2.0*1365.0/pi)/64.0)
                        for (i = 0; i < length; i++) {
                            //p = sqrt(x[i]*x[i]+y[i]*y[i]);
                            Ylm[i] = C6p2*(x_r[i]*x_r[i]-y_r[i]*y_r[i])*((T)33.0*pow(z_r[i],4) - (T)18.0*z_r[i]*z_r[i] + (T)1.0);
                        }
                        break;
                    }
                
                case 3: /* m = 3 */
                    {
                        T C6p3 = 0.921205259514923; // (sqrt(2.0*1365.0/pi)/32.0)
                        for (i = 0; i < length; i++) {
                            // p = sqrt(x_r[i]*x_r[i]+y_r[i]*y_r[i]);
                            // Ylm[i] = C6p3*((T)4.0*x_r[i]*x_r[i]*x_r[i] -(T)3.0*x_r[i]*p*p)*((T)11.0*z_r[i]*z_r[i]*z_r[i] - (T)3.0*z_r[i]);
                            p2 = x_r[i]*x_r[i]+y_r[i]*y_r[i];
                            Ylm[i] = C6p3*((T)4.0*x_r[i]*x_r[i]*x_r[i] -(T)3.0*x_r[i]*p2)*((T)11.0*z_r[i]*z_r[i]*z_r[i] - (T)3.0*z_r[i]);
                        }
                        break;
                    }
                
                case 4: /* m = 4 */
                    {
                        T C6p4 = 0.504564900728724; // (3.0/32.0)*sqrt(91.0/pi)
                        for (i = 0; i < length; i++) {
                            // p = sqrt(x_r[i]*x_r[i]+y_r[i]*y_r[i]);
                            // Ylm[i] = C6p4*((T)4.0*pow(x_r[i],4)+(T)4.0*pow(y_r[i],4) -(T)3.0*pow(p,4))*((T)11.0*z_r[i]*z_r[i] -(T)1.0);
                            p2 = x_r[i]*x_r[i]+y_r[i]*y_r[i];
                            Ylm[i] = C6p4*((T)4.0*pow(x_r[i],4)+(T)4.0*pow(y_r[i],4) -(T)3.0*pow(p2,2))*((T)11.0*z_r[i]*z_r[i] -(T)1.0);
                        }
                        break;
                    }
                
                case 5: /* m = 5 */
                    {
                        T C6p5 = 2.366619162231752; // (3.0/32.0)*sqrt(2.0*1001.0/pi)
                        for (i = 0; i < length; i++) {
                            // p = sqrt(x_r[i]*x_r[i]+y_r[i]*y_r[i]);
                            // Ylm[i] = C6p5*((T)4.0*pow(x_r[i],5) + (T)8.0*x_r[i]*pow(y_r[i],4)-(T)4.0*x_r[i]*x_r[i]*x_r[i]*y_r[i]*y_r[i]-(T)3.0*x_r[i]*pow(p,4))*z_r[i];
                            p2 = x_r[i]*x_r[i]+y_r[i]*y_r[i];
                            Ylm[i] = C6p5*((T)4.0*pow(x_r[i],5) + (T)8.0*x_r[i]*pow(y_r[i],4)-(T)4.0*x_r[i]*x_r[i]*x_r[i]*y_r[i]*y_r[i]-(T)3.0*x_r[i]*pow(p2,2))*z_r[i];
                        }
                        break;
                    }
                
                case 6: /* m = 6 */
                    {
                        T C6p6 = 0.683184105191914; // (sqrt(2.0*3003.0/pi)/64.0)
                        for (i = 0; i < length; i++) {
                            // p = sqrt(x_r[i]*x_r[i]+y_r[i]*y_r[i]);
                            // Ylm[i] = C6p6*((T)4.0*pow(x_r[i],6)-(T)4.0*pow(y_r[i],6) +(T)12.0*x_r[i]*x_r[i]*pow(y_r[i],4)-(T)12.0*pow(x_r[i],4)*y_r[i]*y_r[i] + (T)3.0*y_r[i]*y_r[i]*pow(p,4)-(T)3.0*x_r[i]*x_r[i]*pow(p,4));
                            p2 = x_r[i]*x_r[i]+y_r[i]*y_r[i];
                            Ylm[i] = C6p6*((T)4.0*pow(x_r[i],6)-(T)4.0*pow(y_r[i],6) +(T)12.0*x_r[i]*x_r[i]*pow(y_r[i],4)-(T)12.0*pow(x_r[i],4)*y_r[i]*y_r[i] + (T)3.0*(y_r[i]*y_r[i] - x_r[i]*x_r[i])*pow(p2,2));
                        }
                        break;
                    }
                    
                /* incorrect m */
                default: printf("<m> must be an integer between %d and %d!\n", -l, l); 
                         break;
            }
            break;
        
        default: printf("<l> must be an integer between 0 and 6!\n"); break;
    }
    
    if (l > 0) {
        for (i = 0; i < length; i++) {
            if (r[i] < 1e-10) Ylm[i] = 0.0;
        }
    }
    return;
}
template void Tools::RealSphericalHarmonic2<float>(float* const& Ylm, const int& l, const int& m, const int& length,
                                                   float const* const& x_r, float const* const& y_r, float const* const& z_r,
                                                   float const* const& r);
template void Tools::RealSphericalHarmonic2<double>(double* const& Ylm, const int& l, const int& m, const int& length,
                                                    double const* const& x_r, double const* const& y_r, double const* const& z_r,
                                                    double const* const& r);

template<typename T>
bool Tools::is_uniform(T const* const& __restrict__ x, const int& length) {
    T dx0 = x[1] - x[0];
    T dxi;
    for (int i = 2; i < length; i++) {
        dxi = x[i] - x[i-1];
        assert(fabs(dxi) > 1e-12);
        if (fabs(dxi - dx0) > 1e-12) return false;
    }
    return true;
}
template bool Tools::is_uniform<float>(float const* const& r, const int& length);
template bool Tools::is_uniform<double>(double const* const& r, const int& length);

template<typename T>
void Tools::simpson_antideriv(T* const y, T const* const x, T const h, const uint64_t length, const bool if_periodic) {
    if (length == 0) return;
    // y_0 = 0
    y[0] = T(0);

    if (length == 1) return;

    // y_1 = y_0 + h/2 * (x_0 + x_1)
    y[1] = y[0] + T(0.5) * h * (x[0] + x[1]);

    if (length == 2) return;

    uint64_t i = 0;

    // Simpson:
    // y_{i+2} = y_i + h/3 * (x_i + 4*x_{i+1} + x_{i+2})
    //
    // Midpoint fill:
    // y_{i+1} = 0.5 * (y_i + y_{i+2})
    for (; i + 2 < length; i += 2)
    {
        const T integral = (h / T(3)) *
            (x[i] + T(4) * x[i + 1] + x[i + 2]);

        y[i + 2] = y[i] + integral;
        y[i + 1] = T(0.5) * (y[i] + y[i + 2]);
    }

    // If one interval is left, use trapezoidal rule:
    // y_{N-1} = y_{N-2} + h/2 * (x_{N-2} + x_{N-1})
    if (i + 1 < length)
    {
        y[length - 1] = y[length - 2]
                      + T(0.5) * h * (x[length - 2] + x[length - 1]);
    }

    // For a generic antiderivative, do NOT force y to be periodic.
    // if_periodic only means x[length] should be treated as x[0]
    // when a caller explicitly needs the closing interval.
    //
    // This function returns y[0..length-1], so no closing interval is added here.

    (void)if_periodic;

    return;
}
template void Tools::simpson_antideriv<float>(float* const y, float const* const x, float const h, const uint64_t length, const bool if_periodic);
template void Tools::simpson_antideriv<double>(double* const y, double const* const x, double const h, const uint64_t length, const bool if_periodic);

template<typename T>
void Tools::simpson_double_antideriv(T* const y, T const* const x, T const h, const uint64_t length, const bool if_periodic) {
    if (length == 0) return;

    // y_0 = 0
    y[0] = T(0);

    if (length == 1) return;

    // g_0 = 0
    T g_i = T(0);

    if (length == 2)
    {
        // First integral:
        // g_1 = g_0 + h/2 * (x_0 + x_1)
        T g_1 = g_i + T(0.5) * h * (x[0] + x[1]);

        // Second integral:
        // y_1 = y_0 + h/2 * (g_0 + g_1)
        y[1] = y[0] + T(0.5) * h * (g_i + g_1);

        return;
    }

    uint64_t i = 0;

    for (; i + 2 < length; i += 2)
    {
        // First integral by Simpson:
        // g_{i+2} = g_i + h/3 * (x_i + 4*x_{i+1} + x_{i+2})
        T g_i2 = g_i
               + (h / T(3)) * (
                     x[i]
                   + T(4) * x[i + 1]
                   + x[i + 2]
                 );

        // Midpoint fill for first integral:
        // g_{i+1} = 0.5 * (g_i + g_{i+2})
        T g_i1 = T(0.5) * (g_i + g_i2);

        // Second integral by Simpson:
        // y_{i+2} = y_i + h/3 * (g_i + 4*g_{i+1} + g_{i+2})
        y[i + 2] = y[i]
                 + (h / T(3)) * (
                       g_i
                     + T(4) * g_i1
                     + g_i2
                   );

        // Midpoint fill for second integral:
        // y_{i+1} = 0.5 * (y_i + y_{i+2})
        y[i + 1] = T(0.5) * (y[i] + y[i + 2]);

        // Move to next even point
        g_i = g_i2;
    }

    // If one interval is left, use trapezoidal rule
    if (i + 1 < length)
    {
        // g_{i+1} = g_i + h/2 * (x_i + x_{i+1})
        T g_i1 = g_i + T(0.5) * h * (x[i] + x[i + 1]);

        // y_{i+1} = y_i + h/2 * (g_i + g_{i+1})
        y[i + 1] = y[i] + T(0.5) * h * (g_i + g_i1);
    }

    // Generic antiderivative: do not force periodicity on y.
    (void)if_periodic;

    return;
}
template void Tools::simpson_double_antideriv<float>(float* const y, float const* const x, float const h, const uint64_t length, const bool if_periodic);
template void Tools::simpson_double_antideriv<double>(double* const y, double const* const x, double const h, const uint64_t length, const bool if_periodic);

template<typename T>
void Tools::second_deriv_fd(T* const y, T const* const f, T const* const coef, const uint64_t FDn, const uint64_t length, const bool if_periodic) {
    if (length == 0) return;

    if (if_periodic)
    {
        // Left boundary: periodic wrap is needed
        for (uint64_t i = 0; i < FDn && i < length; ++i)
        {
            T sum = coef[0] * f[i];

            for (uint64_t k = 1; k <= FDn; ++k)
            {
                const uint64_t il = (i >= k) ? (i - k) : (length + i - k);
                const uint64_t ir = (i + k < length) ? (i + k) : (i + k - length);

                sum += coef[k] * (f[il] + f[ir]);
            }

            y[i] = sum;
        }

        // Interior: no boundary check, no wrap
        if (length > 2 * FDn)
        {
            for (uint64_t i = FDn; i < length - FDn; ++i)
            {
                T sum = coef[0] * f[i];

                for (uint64_t k = 1; k <= FDn; ++k)
                {
                    sum += coef[k] * (f[i - k] + f[i + k]);
                }

                y[i] = sum;
            }
        }

        // Right boundary: periodic wrap is needed
        const uint64_t right_start = (length > FDn) ? (length - FDn) : 0;

        for (uint64_t i = right_start; i < length; ++i)
        {
            // Avoid double-computing when length <= 2*FDn
            if (i < FDn) continue;

            T sum = coef[0] * f[i];

            for (uint64_t k = 1; k <= FDn; ++k)
            {
                const uint64_t il = (i >= k) ? (i - k) : (length + i - k);
                const uint64_t ir = (i + k < length) ? (i + k) : (i + k - length);

                sum += coef[k] * (f[il] + f[ir]);
            }

            y[i] = sum;
        }
    }
    else
    {
        // Left boundary: out-of-domain values are zero
        for (uint64_t i = 0; i < FDn && i < length; ++i)
        {
            T sum = coef[0] * f[i];

            for (uint64_t k = 1; k <= FDn; ++k)
            {
                if (i >= k) sum += coef[k] * f[i - k];
                if (i + k < length) sum += coef[k] * f[i + k];
            }

            y[i] = sum;
        }

        // Interior: no boundary check
        if (length > 2 * FDn)
        {
            for (uint64_t i = FDn; i < length - FDn; ++i)
            {
                T sum = coef[0] * f[i];

                for (uint64_t k = 1; k <= FDn; ++k)
                {
                    sum += coef[k] * (f[i - k] + f[i + k]);
                }

                y[i] = sum;
            }
        }

        // Right boundary: out-of-domain values are zero
        const uint64_t right_start = (length > FDn) ? (length - FDn) : 0;

        for (uint64_t i = right_start; i < length; ++i)
        {
            if (i < FDn) continue;

            T sum = coef[0] * f[i];

            for (uint64_t k = 1; k <= FDn; ++k)
            {
                if (i >= k) sum += coef[k] * f[i - k];
                if (i + k < length) sum += coef[k] * f[i + k];
            }

            y[i] = sum;
        }
    }
    return;
}
template void Tools::second_deriv_fd<float>(float* const y, float const* const f, float const* const coef, const uint64_t FDn, const uint64_t length, const bool if_periodic);
template void Tools::second_deriv_fd<double>(double* const y, double const* const f, double const* const coef, const uint64_t FDn, const uint64_t length, const bool if_periodic);

std::string Tools::time_cost(const std::chrono::steady_clock::time_point& begin, const std::chrono::steady_clock::time_point& end) {
    // unsigned long int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    // if (milliseconds < 60000) {
    //     return std::to_string(milliseconds) + " [ms]";
    // } else if (milliseconds < 60000000) {
    //     return std::to_string(milliseconds / 1000) + " [s]";
    // } else {
    //     return std::to_string(milliseconds / 60000) + " [minutes]";
    // }
    unsigned long int microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    if (microseconds < 60000000) {
        return std::to_string(double(microseconds) / 1000) + " [ms]";
    } else if (microseconds < 60000000000) {
        return std::to_string(double(microseconds) / 1000000) + " [s]";
    } else {
        return std::to_string(double(microseconds) / 60000000) + " [minutes]";
    }
}

/**
 * @brief   Find root of a function using Brent's method.
 *
 * @ref     W.H. Press, Numerical recepies 3rd edition: The art of scientific
 *          computing, Cambridge university press, 2007.
 * @ref     https://github.com/SPARC-X/SPARC/blob/a2ecc61b19090f9c99dd0f5eedb18cb588d58884/src/relax.c#L1367
 */
template<typename T>
T Tools::BrentsFun(std::function<T(const T)> func, const T lower_bound, const T upper_bound,
                                    const uint max_iter, const T tol) {
    T a = lower_bound;
    T b = upper_bound;
    T fa = func(a);
    T fb = func(b);
    T c = T(0);

    uint ext_loop_count = 0;
    while (fa * fb > 0.0 && ext_loop_count++ < 10) {
        T w = b - a;
        a -= w / 2.0;
        b += w / 2.0;
        c = b;
        fa = func(a);
        fb = func(b);
    }

    if (fa * fb > T(0.0)) {
        assert(0 && "Cannot find the chemical potential in the given range!");
    }

    T fc = fb;
    T e = T(0);
    T d = T(0);
    T tol1 = T(0);
    T xm = T(0);
    T s = T(0);
    T p = T(0);
    T q = T(0);
    T r = T(0);
    T tol1q = T(0);
    T min1 = T(0);
    T min2 = T(0);
    T eq = T(0);
    #define EPSILON 1e-16
    #define SIGN(a, b) ((b) > T(0.0) ? std::fabs((a)) : -std::fabs((a)))
    for (uint iter = 1; iter <= max_iter; iter++) {
        if ((fb > T(0.0) && fc > T(0.0)) || (fb < T(0.0) && fc < T(0.0))) {
            c = a;
            fc = fa;
            e = d = b - a;
        }
        if (std::fabs(fc) < std::fabs(fb)) {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        tol1 = 2.0 * T(EPSILON) * std::fabs(b) + 0.5 * tol;
        xm = 0.5 * (c - b);
        if (std::fabs(xm) <= tol1 || std::fabs(fb) < T(EPSILON)) {
            return b;
        }
        if (std::fabs(e) >= tol1 && std::fabs(fa) > std::fabs(fb)) {
            // attempt inverse quadratic interpolation
            s = fb / fa;
            if (a == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                q = fa / fc;
                r = fb / fc;
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) {
                // check whether in bounds
                q = -q;
            }
            p = fabs(p);
            tol1q = tol1 * q;
            min1 = 3.0 * xm * q - fabs(tol1q);
            eq = e * q;
            min2 = fabs(eq);
            if (2.0 * p < (min1 < min2 ? min1 : min2)) {
                // accept interpolation
                e = d;
                d = p / q;
            } else {
                // Bounds decreasing too slowly, use bisection
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        // move last best guess to a
        a = b;
        fa = fb;

        if (fabs(d) > tol1) {
            // evaluate new trial root
            b += d;
        } else {
            b += SIGN(tol1, xm);
        }
        fb = func(b);
    }
    #undef EPSILON
    #undef SIGN

    return T(0.0);
}
template float Tools::BrentsFun<float>(std::function<float(const float)> func, const float lower_bound, const float upper_bound,
                                                    const uint max_iter, const float tol);
template double Tools::BrentsFun<double>(std::function<double(const double)> func, const double lower_bound, const double upper_bound,
                                                    const uint max_iter, const double tol);

/**
 * @brief   Find root of a function using Batched Bisection (Polysection) method.
 * 
 * @param func        Batched function evaluator: func(in_array, out_array, n)
 * @param lower_bound Initial left bound (a)
 * @param upper_bound Initial right bound (b)
 * @param max_iter    Maximum number of iterations
 * @param tol         Tolerance for the root
 * @param n           Batch size (number of points evaluated per iteration)
 * @return            The estimated root
 */
template<typename T>
T Tools::BatchedBisection(std::function<void(const T* const in, T* const out, const uint n)> func, 
                    const T lower_bound, const T upper_bound,
                    const uint max_iter, const T tol, const uint n) {
    if (n == 0) {
        throw std::invalid_argument("Batch size 'n' must be at least 1.");
    }

    // Allocate memory once to avoid reallocation inside the loop
    std::vector<T> x(n + 2);
    std::vector<T> y(n + 2);

    // 1. Initial Check: Evaluate bounds to ensure they bracket the root
    x[0] = lower_bound;
    x[1] = upper_bound;
    func(x.data(), y.data(), 2);
    
    if (y[0] == T(0.0)) return x[0];
    if (y[1] == T(0.0)) return x[1];

    if (std::signbit(y[0]) == std::signbit(y[1])) {
        throw std::runtime_error("Root is not bracketed by initial bounds!");
    }

    T a = x[0];
    T fa = y[0];
    T b = x[1];
    T fb = y[1];

    // 2. Main Iteration Loop
    for (unsigned int iter = 1; iter <= max_iter; iter++) {
        T width = b - a;
        
        // Check convergence
        if (std::abs(width) <= tol) {
            return a + width / T(2.0); // Return midpoint
        }

        // Generate 'n' evenly spaced inner points
        T step = width / T(n + 1);
        
        x[0] = a;
        y[0] = fa;
        for (unsigned int i = 1; i <= n; i++) {
            x[i] = a + i * step;
        }
        x[n + 1] = b;
        y[n + 1] = fb;

        // Evaluate the 'n' inner points in one batch
        // Pass the pointers offset by 1 to only compute inner points
        func(&x[1], &y[1], n);

        // Find the sub-interval that contains the root (sign change)
        bool found = false;
        for (unsigned int i = 0; i <= n; i++) {
            if (y[i] == T(0.0)) {
                return x[i]; // Exact root hit
            }
            
            if (std::signbit(y[i]) != std::signbit(y[i + 1])) {
                a = x[i];
                fa = y[i];
                b = x[i + 1];
                fb = y[i + 1];
                found = true;
                break;
            }
        }

        if (!found) {
            // Should theoretically never happen for continuous functions
            throw std::runtime_error("Lost the root during polysection! Function might be discontinuous.");
        }
    }

    return a + (b - a) / T(2.0); // Return best guess if max_iter reached
}
template float Tools::BatchedBisection<float>(
    std::function<void(const float* const in, float* const out, const uint n)> func, 
    const float lower_bound, const float upper_bound,
    const uint max_iter, const float tol, const uint n);
template double Tools::BatchedBisection<double>(
    std::function<void(const double* const in, double* const out, const uint n)> func, 
    const double lower_bound, const double upper_bound,
    const uint max_iter, const double tol, const uint n);

template<typename T>
T Tools::BatchedBrent(std::function<void(const T* const, T* const, const unsigned int)> func, 
                const T lower_bound, const T upper_bound,
                const unsigned int max_iter, const T tol, const unsigned int n) {
    if (n < 1) throw std::invalid_argument("Batch size must be >= 1");

    struct Point { T x; T y; };
    std::vector<T> in_x(n), out_y(n);
    std::vector<Point> pts(n + 2);

    // 1. 初始化端点
    T L = lower_bound, R = upper_bound;
    in_x[0] = L; in_x[1] = R;
    func(in_x.data(), out_y.data(), 2);
    T fL = out_y[0], fR = out_y[1];
    
    if (fL == T(0.0)) return L;
    if (fR == T(0.0)) return R;
    if (std::signbit(fL) == std::signbit(fR)) throw std::runtime_error("Root not bracketed!");

    // C 点用于逆二次插值 (IQI)，初始设为 L (退化为割线法)
    T C = L, fC = fL; 

    for (unsigned int iter = 1; iter <= max_iter; iter++) {
        if (std::abs(R - L) <= tol) return L + (R - L) / T(2.0);

        // 记录当前的最佳估计点 (用于下一次迭代的 C 点)
        Point old_best = std::abs(fL) < std::abs(fR) ? Point{L, fL} : Point{R, fR};

        // 2. 尝试 插值预测 (IQI / 割线法)
        T guess_x;
        bool can_iqi = (fL != fC) && (fR != fC) && (fL != fR);
        if (can_iqi) {
            // 逆二次插值 (Inverse Quadratic Interpolation)
            guess_x = L * fR * fC / ((fL - fR) * (fL - fC))
                    + R * fL * fC / ((fR - fL) * (fR - fC))
                    + C * fL * fR / ((fC - fL) * (fC - fR));
        } else if (fL != fR) {
            // 割线法 (Secant Method)
            guess_x = R - fR * (R - L) / (fR - fL);
        } else {
            guess_x = L - 1.0; // 强制失效
        }

        // 3. 验证预测点是否在区间内（并且不要太靠近边缘，防止停滞）
        bool use_guess = false;
        T min_x = std::min(L, R), max_x = std::max(L, R);
        T safe_margin = (max_x - min_x) * T(0.05); // 至少距离边界 5%
        if (guess_x > min_x + safe_margin && guess_x < max_x - safe_margin) {
            use_guess = true;
        }

        // 4. 生成批处理计算点
        unsigned int n_even = use_guess ? (n - 1) : n;
        unsigned int idx = 0;
        
        if (use_guess) in_x[idx++] = guess_x; // 将最佳预测点放入 Batch
        
        // 剩余的点用于传统的均分（提供兜底保障）
        T step = (R - L) / T(n_even + 1);
        for (unsigned int i = 1; i <= n_even; i++) {
            in_x[idx++] = L + i * step;
        }

        // 5. 批量评估
        func(in_x.data(), out_y.data(), n);

        // 6. 收集所有点并排序
        pts[0] = {L, fL};
        pts[1] = {R, fR};
        for (unsigned int i = 0; i < n; i++) pts[i + 2] = {in_x[i], out_y[i]};
        
        std::sort(pts.begin(), pts.end(), [](const Point& p1, const Point& p2) {
            return p1.x < p2.x;
        });

        // 7. 寻找新的根包络
        for (unsigned int i = 0; i < pts.size() - 1; i++) {
            if (pts[i].y == T(0.0)) return pts[i].x;
            if (pts[i + 1].y == T(0.0)) return pts[i + 1].x;
            
            if (std::signbit(pts[i].y) != std::signbit(pts[i + 1].y)) {
                L = pts[i].x; fL = pts[i].y;
                R = pts[i + 1].x; fR = pts[i + 1].y;
                break;
            }
        }

        // 8. 更新 C 点
        C = old_best.x; fC = old_best.y;
    }

    return L + (R - L) / T(2.0);
}
template float Tools::BatchedBrent<float>(std::function<void(const float* const, float* const, const unsigned int)> func, 
                const float lower_bound, const float upper_bound,
                const unsigned int max_iter, const float tol, const unsigned int n);
template double Tools::BatchedBrent<double>(std::function<void(const double* const, double* const, const unsigned int)> func, 
                const double lower_bound, const double upper_bound,
                const unsigned int max_iter, const double tol, const unsigned int n);

/**
 * @brief   Global Spectral Histogram Root Finder for Electronic Chemical Potential.
 *          Combines 1-shot global DOS compression, local zero-comm solve,
 *          machine-precision-safe Newton correction, and Batched Exponential/Bisection fallback.
 */
template<typename T>
T Tools::SpectralHistogramRoot(
    std::function<void(const T L, const T R, const uint num_bins, T* const out_hist)> hist_func,
    std::function<void(const T* const in, T* const out_f, T* const out_df, const uint n)> exact_func_df,
    std::function<void(const T* const in, T* const out_f, const uint n)> exact_func_f,
    std::function<T(const T delta_E)> local_smear_eval,
    const T lower_bound, const T upper_bound,
    const uint max_iter, const T tol, const T target_charge,
    const uint num_bins, const uint fallback_n,
    std::function<T(const T chemical_potential)> local_extra_eval)
{
    // =========================================================================
    // 阶段 1: 1 次 Allreduce 构建全能谱全局直方图
    // =========================================================================
    std::vector<T> H(num_bins, T(0.0));
    hist_func(lower_bound, upper_bound, num_bins, H.data());

    T bin_width = (upper_bound - lower_bound) / T(num_bins);
    std::vector<T> bin_centers(num_bins);
    for (uint i = 0; i < num_bins; i++) {
        bin_centers[i] = lower_bound + (T(i) + T(0.5)) * bin_width;
    }

    // =========================================================================
    // 阶段 2: 本地 0 通信求解粗根 (纯二分法：绝对稳健、无除以0风险、耗时极低)
    // =========================================================================
    auto local_f = [&](T mu) {
        T f_mod = -target_charge;
        for (uint i = 0; i < num_bins; i++) {
            if (H[i] <= T(0.0)) continue;
            f_mod += H[i] * local_smear_eval(bin_centers[i] - mu);
        }
        if (local_extra_eval) {
            f_mod += local_extra_eval(mu);
        }
        return f_mod;
    };

    T a_loc = lower_bound;
    T b_loc = upper_bound;
    T x_local = (a_loc + b_loc) / T(2.0);

    // 55 次二分法可将区间压缩 2^55 ≈ 3.6e16 倍，直接逼近 IEEE 754 双精度极限
    for (int iter = 0; iter < 55; iter++) {
        x_local = (a_loc + b_loc) / T(2.0);
        
        // 区间已足够狭窄，提前终止
        if ((b_loc - a_loc) <= T(1e-13)) break; 

        T f_m = local_f(x_local);
        if (std::abs(f_m) <= T(1e-13)) break;

        // 利用单调递增性安全收缩括号
        if (f_m < T(0.0)) a_loc = x_local;
        else              b_loc = x_local;
    }

    // =========================================================================
    // 阶段 3: 精确能级 Newton 修正 (严格监控异常，遇错即移交 Fallback)
    // =========================================================================
    T x_curr = x_local;
    T x_prev1 = lower_bound - T(999.0);
    T x_prev2 = lower_bound - T(999.0);
    T f_curr = T(0.0);

    // 相对残差保护: 当总电荷极大时，绝对误差受限于双精度极限 (ULP) 避免过度苛求死锁
    T effective_tol = std::max(tol, target_charge * std::numeric_limits<T>::epsilon() * T(2.0));

    for (int corr_step = 0; corr_step < 10; corr_step++) {
        T df_curr;
        exact_func_df(&x_curr, &f_curr, &df_curr, 1);

        // 成功判定 1: 残差已达到精度要求 (带隙中 df_curr 可能等于 0，只要 f_curr == 0 即为合法解)
        if (std::abs(f_curr) <= effective_tol) {
            return x_curr;
        }

        // 异常判定 1: 带隙保护。导数极小，牛顿法失效，交给 Fallback
        if (df_curr <= T(1e-12)) {
            break; 
        }

        T step = f_curr / df_curr;
        T x_next = x_curr - step;

        // 异常判定 2: 步长极限。当前点与下一点之间已无双精度浮点数 (1 ULP limit)
        if (x_next == x_curr || std::abs(step) <= std::abs(x_curr) * std::numeric_limits<T>::epsilon()) {
            break; 
        }

        // 异常判定 3: 2-周期防死锁。捕获到 A <-> B 相邻浮点数无限横跳
        if (x_next == x_prev2 || x_next == x_prev1) {
            break; 
        }

        // 异常判定 4: 物理越界
        if (x_next < lower_bound || x_next > upper_bound) {
            break; 
        }

        x_prev2 = x_prev1;
        x_prev1 = x_curr;
        x_curr = x_next;
    }

    // =========================================================================
    // 阶段 4: Batched Exponential Bracket Expansion (极速指数探边)
    // =========================================================================
    // 当非金属体系牛顿法陷入带隙或双精度极限时，基于最后的 x_curr 进行单向指数级探边。
    
    T L_fb = x_curr;
    T R_fb = x_curr;
    
    // 利用严格单调递增性判定搜索方向 (若 f_curr < 0, 说明根在右边)
    T dir = (f_curr < T(0.0)) ? T(1.0) : T(-1.0); 
    // 高精度微调步长 (设定为 1e-10 极微小探针，配合指数膨胀可快速覆盖微观到宏观)
    T base_step = T(1e-10) * dir; 
    
    bool bracket_found = false;
    std::vector<T> bx(fallback_n), bf(fallback_n);
    T x_anchor = x_curr; 

    // 最多尝试 5 轮批量探测 (5 * fallback_n 个点)
    for (int exp_iter = 0; exp_iter < 5; exp_iter++) { 
        // 1. 构建 Batch 点的坐标
        for (uint i = 0; i < fallback_n; i++) {
            // 使用极速位移运算替代昂贵的 std::pow，按 2 的指数次幂膨胀
            uint64_t power = exp_iter * fallback_n + i;
            T multiplier = (power < 62) ? T(1ULL << power) : std::pow(T(2.0), T(power)); 
            T shift = base_step * multiplier;
            bx[i] = std::max(lower_bound, std::min(upper_bound, x_anchor + shift));
        }

        // 2. 真正的批量并行评估 (仅产生 1 次 AllReduce)
        exact_func_f(bx.data(), bf.data(), fallback_n);

        // 3. 寻找符号翻转点，圈定包络区间
        for (uint i = 0; i < fallback_n; i++) {
            if (std::abs(bf[i]) <= effective_tol) {
                return bx[i]; // 运气爆棚，探测点正好命中根
            }
            
            // f_curr 记录的是序列中前一个点的函数值
            if (std::signbit(f_curr) != std::signbit(bf[i])) {
                L_fb = std::min(x_curr, bx[i]);
                R_fb = std::max(x_curr, bx[i]);
                bracket_found = true;
                break;
            }
            x_curr = bx[i];
            f_curr = bf[i];
        }

        if (bracket_found) break;
    }

    if (!bracket_found) {
        // 极端异常兜底：若探出物理边界都没找到异号，退回全域
        L_fb = lower_bound;
        R_fb = upper_bound;
    }

    // =========================================================================
    // 阶段 5: 在紧致包络中执行最终的 Batched Bisection (绝对稳健收尾)
    // =========================================================================
    return Tools::BatchedBisection<T>(exact_func_f, L_fb, R_fb, max_iter, tol, fallback_n);
}

// 显式模板实例化
template float Tools::SpectralHistogramRoot<float>(
    std::function<void(const float, const float, const uint, float* const)>,
    std::function<void(const float* const, float* const, float* const, const uint)>,
    std::function<void(const float* const, float* const, const uint)>,
    std::function<float(const float)>, const float, const float, const uint, const float, const float, const uint, const uint,
    std::function<float(const float)>);

template double Tools::SpectralHistogramRoot<double>(
    std::function<void(const double, const double, const uint, double* const)>,
    std::function<void(const double* const, double* const, double* const, const uint)>,
    std::function<void(const double* const, double* const, const uint)>,
    std::function<double(const double)>, const double, const double, const uint, const double, const double, const uint, const uint,
    std::function<double(const double)>);

void Tools::calc_mpi_statistics(
    double const* const local_values,
    const int nitems,
    const MPI_Comm comm,
    double* const max_values,
    double* const min_values,
    double* const mean_values,
    double* const stddev_values,
    double* const reciprocal_mean_values)
{
    assert(local_values != nullptr);
    assert(max_values != nullptr);
    assert(min_values != nullptr);
    assert(mean_values != nullptr);
    assert(stddev_values != nullptr);
    assert(reciprocal_mean_values != nullptr);
    assert(nitems > 0);
    assert(comm != MPI_COMM_NULL);

    int comm_size = 0;
    MPI_Comm_size(comm, &comm_size);

    /*
     * Pack:
     *   [0, nitems)           : x
     *   [nitems, 2*nitems)    : x^2
     *   [2*nitems, 3*nitems)  : 1/x
     */
    std::vector<double> local_sum_data(3 * nitems);
    std::vector<double> global_sum_data(3 * nitems);

    for (int i = 0; i < nitems; ++i) {
        const double x = local_values[i];

        /*
         * mean(1/x) requires x != 0.
         */
        // assert(x != 0.0);

        local_sum_data[i]              = x;
        local_sum_data[nitems + i]     = x * x;
        local_sum_data[2 * nitems + i] = 1.0 / x;
    }

    /*
     * SUM:
     *   sum(x)
     *   sum(x^2)
     *   sum(1/x)
     */
    MPI_Allreduce(
        local_sum_data.data(),
        global_sum_data.data(),
        3 * nitems,
        MPI_DOUBLE,
        MPI_SUM,
        comm);

    /*
     * MAX
     */
    MPI_Allreduce(
        local_values,
        max_values,
        nitems,
        MPI_DOUBLE,
        MPI_MAX,
        comm);

    /*
     * MIN
     */
    MPI_Allreduce(
        local_values,
        min_values,
        nitems,
        MPI_DOUBLE,
        MPI_MIN,
        comm);

    const double inv_comm_size = 1.0 / static_cast<double>(comm_size);

    for (int i = 0; i < nitems; ++i) {
        const double mean =
            global_sum_data[i] * inv_comm_size;

        const double mean_square =
            global_sum_data[nitems + i] * inv_comm_size;

        /*
         * Population standard deviation:
         *
         * std = sqrt(E[x^2] - E[x]^2)
         *
         * max(..., 0) avoids tiny negative values caused by
         * floating-point roundoff.
         */
        const double variance =
            std::max(0.0, mean_square - mean * mean);

        mean_values[i] = mean;
        stddev_values[i] = std::sqrt(variance);

        reciprocal_mean_values[i] =
            global_sum_data[2 * nitems + i] * inv_comm_size;
    }
}

void Tools::calc_mpi_statistics(
    double const* const local_values,
    const int nitems,
    const MPI_Comm comm,
    double* const max_values,
    double* const min_values,
    double* const mean_values,
    double* const stddev_values,
    double* const reciprocal_mean_values,
    int* const max_ranks)
{
    assert(local_values != nullptr);
    assert(max_values != nullptr);
    assert(min_values != nullptr);
    assert(mean_values != nullptr);
    assert(stddev_values != nullptr);
    assert(reciprocal_mean_values != nullptr);
    assert(max_ranks != nullptr);
    assert(nitems > 0);
    assert(comm != MPI_COMM_NULL);

    int comm_size = 0;
    int comm_rank = 0;

    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    /*
     * Pack:
     *   [0, nitems)           : x
     *   [nitems, 2*nitems)    : x^2
     *   [2*nitems, 3*nitems)  : 1/x
     */
    std::vector<double> local_sum_data(3 * nitems);
    std::vector<double> global_sum_data(3 * nitems);

    for (int i = 0; i < nitems; ++i) {
        const double x = local_values[i];

        local_sum_data[i]              = x;
        local_sum_data[nitems + i]     = x * x;
        local_sum_data[2 * nitems + i] = 1.0 / x;
    }

    /*
     * SUM:
     *   sum(x)
     *   sum(x^2)
     *   sum(1/x)
     */
    MPI_Allreduce(
        local_sum_data.data(),
        global_sum_data.data(),
        3 * nitems,
        MPI_DOUBLE,
        MPI_SUM,
        comm);

    /*
     * MAX + rank of MAX
     */
    struct Double_int {
        double value;
        int rank;
    };

    std::vector<Double_int> local_maxloc(nitems);
    std::vector<Double_int> global_maxloc(nitems);

    for (int i = 0; i < nitems; ++i) {
        local_maxloc[i].value = local_values[i];
        local_maxloc[i].rank  = comm_rank;
    }

    MPI_Allreduce(
        local_maxloc.data(),
        global_maxloc.data(),
        nitems,
        MPI_DOUBLE_INT,
        MPI_MAXLOC,
        comm);

    for (int i = 0; i < nitems; ++i) {
        max_values[i] = global_maxloc[i].value;
        max_ranks[i]  = global_maxloc[i].rank;
    }

    /*
     * MIN
     */
    MPI_Allreduce(
        local_values,
        min_values,
        nitems,
        MPI_DOUBLE,
        MPI_MIN,
        comm);

    const double inv_comm_size =
        1.0 / static_cast<double>(comm_size);

    for (int i = 0; i < nitems; ++i) {
        const double mean =
            global_sum_data[i] * inv_comm_size;

        const double mean_square =
            global_sum_data[nitems + i] * inv_comm_size;

        const double variance =
            std::max(0.0, mean_square - mean * mean);

        mean_values[i] = mean;

        stddev_values[i] =
            std::sqrt(variance);

        reciprocal_mean_values[i] =
            global_sum_data[2 * nitems + i] * inv_comm_size;
    }
}

void Tools::calc_mpi_statistics(
    double const* const local_values,
    const int nitems,
    const MPI_Comm comm,
    double* const max_values,
    double* const min_values,
    double* const mean_values,
    double* const stddev_values,
    double* const reciprocal_mean_values,
    int* const max_ranks,
    int* const min_ranks)
{
    assert(local_values != nullptr);

    assert(max_values != nullptr);
    assert(min_values != nullptr);
    assert(mean_values != nullptr);
    assert(stddev_values != nullptr);
    assert(reciprocal_mean_values != nullptr);

    assert(max_ranks != nullptr);
    assert(min_ranks != nullptr);

    assert(nitems > 0);
    assert(comm != MPI_COMM_NULL);


    int comm_size = 0;
    int comm_rank = 0;

    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);



    /*
     * Sum:
     *
     *   sum(x)
     *   sum(x^2)
     *   sum(1/x)
     */
    std::vector<double> local_sum_data(3 * nitems);
    std::vector<double> global_sum_data(3 * nitems);


    for (int i = 0; i < nitems; ++i) {

        const double x = local_values[i];

        local_sum_data[i] =
            x;

        local_sum_data[nitems + i] =
            x * x;

        local_sum_data[2 * nitems + i] =
            1.0 / x;
    }


    MPI_Allreduce(
        local_sum_data.data(),
        global_sum_data.data(),
        3 * nitems,
        MPI_DOUBLE,
        MPI_SUM,
        comm);



    /*
     * Max value + rank
     */
    struct Double_int
    {
        double value;
        int rank;
    };


    std::vector<Double_int> local_maxloc(nitems);
    std::vector<Double_int> global_maxloc(nitems);

    std::vector<Double_int> local_minloc(nitems);
    std::vector<Double_int> global_minloc(nitems);



    for (int i = 0; i < nitems; ++i) {

        local_maxloc[i].value = local_values[i];
        local_maxloc[i].rank  = comm_rank;

        local_minloc[i].value = local_values[i];
        local_minloc[i].rank  = comm_rank;
    }



    MPI_Allreduce(
        local_maxloc.data(),
        global_maxloc.data(),
        nitems,
        MPI_DOUBLE_INT,
        MPI_MAXLOC,
        comm);


    MPI_Allreduce(
        local_minloc.data(),
        global_minloc.data(),
        nitems,
        MPI_DOUBLE_INT,
        MPI_MINLOC,
        comm);



    for (int i = 0; i < nitems; ++i) {

        max_values[i] =
            global_maxloc[i].value;

        max_ranks[i] =
            global_maxloc[i].rank;


        min_values[i] =
            global_minloc[i].value;

        min_ranks[i] =
            global_minloc[i].rank;
    }



    const double inv_comm_size =
        1.0 / static_cast<double>(comm_size);



    for (int i = 0; i < nitems; ++i) {

        const double mean =
            global_sum_data[i] *
            inv_comm_size;


        const double mean_square =
            global_sum_data[nitems + i] *
            inv_comm_size;


        const double variance =
            std::max(
                0.0,
                mean_square - mean * mean);



        mean_values[i] =
            mean;


        stddev_values[i] =
            std::sqrt(variance);


        reciprocal_mean_values[i] =
            global_sum_data[2 * nitems + i] *
            inv_comm_size;
    }
}
