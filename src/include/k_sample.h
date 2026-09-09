#ifndef _K_SAMPLE_H_
#define _K_SAMPLE_H_

#include <iostream>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <sstream>
#include <climits>
#include <mpi.h>
#include <cstddef>
#include "filesys.h"
#include "atom.h"
#include "geometry.h"
#include "control.h"

class K_sample
{
public:
    const K_sample_control& k_sample_control;
    const Geometry& geometry;
    double G1[3] = {0.0, 0.0, 0.0};           // reciprocal lattice vector v^{T} . G = 2\pi
    double G2[3] = {0.0, 0.0, 0.0};           // reciprocal lattice vector v^{T} . G = 2\pi
    double G3[3] = {0.0, 0.0, 0.0};           // reciprocal lattice vector v^{T} . G = 2\pi
    int nkpt = 0;
    Array_0D<double> k_weights;
    Array_0D<double> relative_k1s;
    Array_0D<double> relative_k2s;
    Array_0D<double> relative_k3s;
    K_sample(const K_sample_control& k_sample_control, const Geometry& geometry);
    K_sample(const K_sample& other);
    ~K_sample();
    K_sample& operator=(const K_sample& other);
    K_sample& operator=(K_sample&& other);
    void cal_Gs();
    void generate_kpoints();
    void init();
    void show() const;
};














#endif //_K_SAMPLE_H_
