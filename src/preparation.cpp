#include "preparation.h"


Preparation::Preparation() {}

Preparation::~Preparation() {}

Preparation::Preparation(const char* fname) {
    this->init(Path(fname));
}
Preparation::Preparation(const Path& fname) {
    this->init(fname);
}

#include <filesystem>
void Preparation::init(const Path& fname) {

    int rank;
    int size;
    int rank_local;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int chunk = 16 * 32;
    int color = rank/chunk;

    const std::string& path = fname + ".inpt.2";
    if (rank == 0) this->if_input_file2_exists = std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
    MPI_Bcast(&(this->if_input_file2_exists), 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &comm);
    MPI_Comm_rank(comm, &rank_local);

    #ifdef USE_OPENMP
    Linalg::init_omp_env();
    #endif
    #ifdef USE_HBM
    #ifdef USE_HUGEPAGE_HBM
    #else
    hbw_set_policy(HBW_POLICY_BIND);
    // hbw_set_policy(HBW_POLICY_PREFERRED);
    #endif
    #endif
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if (rank_local == 0) {
        this->input_file.read(fname + ".inpt");
        this->ion_file.read(fname + ".ion");
        this->geometry.init(this->input_file, this->ion_file);
        this->control.init(this->input_file, this->geometry);
        if (this->if_input_file2_exists) {
            this->input_file2.read(fname + ".inpt.2");
            this->control2.init(this->input_file2, this->geometry);
        }
    }
    this->geometry.bcast(comm, 0);
    this->control.bcast(comm, 0);
    if (this->if_input_file2_exists) this->control2.bcast(comm, 0);
    if (this->control.flow_control.scf_flag == 1) {
        this->psp8_files.resize(this->geometry.natom_type);
        for (uint i = 0; i < this->geometry.natom_type; ++i) {
            if (rank_local == 0) {
                this->psp8_files[i].read(this->ion_file.pseudo_pot_files[i]);
            }
            this->psp8_files[i].bcast(comm, 0);
            this->set_charge_cut(this->psp8_files[i]);
        }
    } else {
        assert(this->control.flow_control.scf_flag == 1);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    MPI_Comm_free(&comm);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        this->control.print(std::cout);
        std::cout << "The Preparation init took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

void Preparation::set_charge_cut(Psp8_file& psp8_file) {
    // Info: Working in the four upper quadrants only (from symmetry) for nonorthogonal and 1 quadrant for orthogonal systems
// TODO: rb_max depends on finite difference order as well, improve this function
#define RB_MAX(h) (h) < 1.5 ? (h)*10+10 : 20*(h)-9.5
    double Rbmax_x, Rbmax_y, Rbmax_z;
    int rank, nx, ny, nz, FDn, i, j, k;
    int count, Ncube_x, Ncube_y, Ncube_z;
    double dx, dy, dz;
    double Bint, val, rchrg; 
    double rb_cur_x, rb_cur_y, rb_cur_z, rb_prev_x, rb_prev_y, rb_prev_z, error_cur, error_prev;

    assert(this->geometry.cell_type <= 2);


    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    dx = this->control.mesh_control.delta_x;
    dy = this->control.mesh_control.delta_y;
    dz = this->control.mesh_control.delta_z;

    double TOL_PSEUDOCHARGE = this->control.scf_control.tolerance*0.001;
    
    // if Rbmax too small, change the function RB_MAX() or override Rbmax later!
    Rbmax_x = RB_MAX(dx);
    Rbmax_y = RB_MAX(dy);
    Rbmax_z = RB_MAX(dz);
    Atom atom(0.0,0.0,0.0,0);
    
    /******************************************************
     *  If Rbmax is not big enough, override Rbmax below  *
     ******************************************************/
    // Rbmax_x = 15;
    // Rbmax_y = 15;
    // Rbmax_z = 15;
    nx = ceil(Rbmax_x / dx - 1e-12);
    ny = ceil(Rbmax_y / dy - 1e-12);
    nz = ceil(Rbmax_z / dz - 1e-12);

    Stencil<double> stencil;
    stencil.init(this->geometry.cell_type, this->control.stencil_control.order,
                 this->control.mesh_control.delta_x, this->control.mesh_control.delta_y, this->control.mesh_control.delta_z);
    FDn = stencil.FDn;
    rchrg = psp8_file.rgrid_local_potential[psp8_file.mmax-1];
    Vertices_3D vertices(0, nx, 0, ny, 0, nz);
    Vertices_3D ex_vertices = vertices.generate_ex_vertices(FDn);
    Array_3D<double> ex_R = atom.generate_R_array<double>(this->geometry.cell_type, ex_vertices, dx, dy, dz);
    uint ex_len = ex_R.length;
    Array_3D<double> V_ex(ex_vertices);

    Ncube_x = 0;
    Ncube_y = 0;
    Ncube_z = 0;
    // apply cubic spline interpolation
    if (psp8_file.is_rgrid_local_potential_uniform) {
        Tools::SplineInterpUniform(psp8_file.rgrid_local_potential.data(), psp8_file.local_potential_R.data(), psp8_file.mmax, 
                            ex_R.data, V_ex.data, ex_len, psp8_file.local_potential_RD.data());
    } else {
        Tools::SplineInterpNonuniform(psp8_file.rgrid_local_potential.data(), psp8_file.local_potential_R.data(), psp8_file.mmax, 
                            ex_R.data, V_ex.data, ex_len, psp8_file.local_potential_RD.data());
    }

    double* const& __restrict__ V_ex_data = V_ex.data;
    double const* const& __restrict__ ex_R_data = ex_R.data;
    const double mzion = -psp8_file.zion;
    for (i = 0; i < (int)ex_len; i++) {
        if (ex_R_data[i] < 1e-12) {
            V_ex_data[i] = psp8_file.local_potential[0];
        } else if (ex_R_data[i] > rchrg) {
            V_ex_data[i] = mzion / ex_R_data[i];
        } else {
            V_ex_data[i] =  V_ex_data[i]/ ex_R_data[i];
        }
    }
    /* calculate pseudocharge density */
    // Array_3D<double> b = V_ex.calc_laplacian(vertices, stencil);
    Array_3D<double> b(vertices);
    double const* const& __restrict__ b_data = b.data;
    Stencil_method::calc_laplacian(V_ex.data, ex_vertices, stencil, vertices, b.data, vertices);

    /* go over different radii by BISECTION */
    double temp, lowerbd_x, lowerbd_y, lowerbd_z, upperbd_x, upperbd_y, upperbd_z, rtol_x, rtol_y, rtol_z, rc;
    int errflag = 0;
    //lloc = pSPARC->localPsd[ityp];
    rc = *std::max_element(psp8_file.r_core.begin(),psp8_file.r_core.end());

    
    rb_cur_x = rb_cur_z = rc;
    rb_cur_y = rc;
    rb_prev_x = rb_prev_z = rb_cur_x;
    rb_prev_y = rb_cur_y;
    error_cur = 100;
    error_prev = 100;
    lowerbd_x = lowerbd_z = rc;
    lowerbd_y = rc;
    upperbd_x = Rbmax_x;
    upperbd_y = Rbmax_y;
    upperbd_z = Rbmax_z;
    rtol_x = 1.0 * dx;
    rtol_y = 1.0 * dy;
    rtol_z = 1.0 * dz;
    count = 0;
    while (count < 20) {
        if (error_cur < this->control.scf_control.tolerance*0.001) {
            if (error_prev >= TOL_PSEUDOCHARGE) {
                if (fabs(rb_cur_x - rb_prev_x) < rtol_x 
                    && fabs(rb_cur_y - rb_prev_y) < rtol_y
                    && fabs(rb_cur_z - rb_prev_z) < rtol_z) {                  
                    break;
                } else {
                    // update error
                    error_prev = error_cur;
                }
                temp = rb_cur_x;
                // update upper bond
                upperbd_x = std::min(upperbd_x, rb_cur_x);
                if (fabs(rb_cur_x - rb_prev_x) >= rtol_x) {
                    // update rb
                    rb_cur_x = 0.5 * (rb_cur_x + rb_prev_x);
                }
                rb_prev_x = temp; 
                temp = rb_cur_y;
                // update upper bond
                upperbd_y = std::min(upperbd_y, rb_cur_y);
                if (fabs(rb_cur_y - rb_prev_y) >= rtol_y) {
                    // update rb
                    rb_cur_y = 0.5 * (rb_cur_y + rb_prev_y);     
                }
                rb_prev_y = temp; 
                temp = rb_cur_z;
                // update upper bond
                upperbd_z = std::min(upperbd_z, rb_cur_z);
                if (fabs(rb_cur_z - rb_prev_z) >= rtol_z) {
                    // update rb
                    rb_cur_z = 0.5 * (rb_cur_z + rb_prev_z);     
                }
                rb_prev_z = temp; 
            } else {
                if (fabs(rb_cur_x - lowerbd_x) < rtol_x 
                    && fabs(rb_cur_y - lowerbd_y) < rtol_y
                    && fabs(rb_cur_z - lowerbd_z) < rtol_z) {                  
                    break;
                } else {
                    // update error
                    error_prev = error_cur;
                }
                rb_prev_x = rb_cur_x;
                // update upper bond
                upperbd_x = std::min(upperbd_x, rb_cur_x);
                if (fabs(rb_cur_x - lowerbd_x) >= rtol_x) { 
                    // update rb
                    rb_cur_x = 0.5 * (lowerbd_x + rb_cur_x);
                }
                rb_prev_y = rb_cur_y;
                // update upper bond
                upperbd_y = std::min(upperbd_y, rb_cur_y);	
                if (fabs(rb_cur_y - lowerbd_y) >= rtol_y) { 
                    // update rb
                    rb_cur_y = 0.5 * (lowerbd_y + rb_cur_y);
                }
                rb_prev_z = rb_cur_z;
                // update upper bond
                upperbd_z = std::min(upperbd_z, rb_cur_z);
                if (fabs(rb_cur_z - lowerbd_z) >= rtol_z) { 
                    // update rb
                    rb_cur_z = 0.5 * (lowerbd_z + rb_cur_z);
                }
            }
        } else {
            if (error_prev >= TOL_PSEUDOCHARGE) {
                if (fabs(upperbd_x - rb_cur_x) < rtol_x
                    && fabs(upperbd_y - rb_cur_y) < rtol_y
                    && fabs(upperbd_z - rb_cur_z) < rtol_z) {
                    rb_cur_x = upperbd_x;
                    rb_cur_y = upperbd_y;
                    rb_cur_z = upperbd_z;
                    errflag = 1;
                    break;
                } else {
                    error_prev = error_cur;
                }
                rb_prev_x = rb_cur_x;
                // update lower bound
                lowerbd_x = std::max(lowerbd_x, rb_cur_x);
                if (fabs(upperbd_x - rb_cur_x) >= rtol_x) { 
                    // update rb
                    rb_cur_x = 0.5 * (upperbd_x + rb_cur_x);
                }
                rb_prev_y = rb_cur_y;
                // update lower bound
                lowerbd_y = std::max(lowerbd_y, rb_cur_y);
                if (fabs(upperbd_y - rb_cur_y) >= rtol_y) { 
                    // update rb
                    rb_cur_y = 0.5 * (upperbd_y + rb_cur_y);
                }
                rb_prev_z = rb_cur_z;
                // update lower bound
                lowerbd_z = std::max(lowerbd_z, rb_cur_z);
                if (fabs(upperbd_z - rb_cur_z) >= rtol_z) { 
                    // update rb
                    rb_cur_z = 0.5 * (upperbd_z + rb_cur_z);
                }
            } else {
                if (fabs(rb_prev_x - rb_cur_x) < rtol_x 
                    && fabs(rb_prev_y - rb_cur_y) < rtol_y 
                    && fabs(rb_prev_z - rb_cur_z) < rtol_z) {
                    rb_cur_x = rb_prev_x;
                    rb_cur_y = rb_prev_y;
                    rb_cur_z = rb_prev_z;
                    error_cur = error_prev;
                    break;
                } else {
                    error_prev = error_cur;
                }
                temp = rb_cur_x;
                // update lower bound
                lowerbd_x = std::max(lowerbd_x, rb_cur_x);
                if (fabs(rb_prev_x - rb_cur_x) >= rtol_x) {
                    // update rb
                    rb_cur_x = 0.5 * (rb_cur_x + rb_prev_x);
                }
                rb_prev_x = temp;
                temp = rb_cur_y;
                // update lower bound
                lowerbd_y = std::max(lowerbd_y, rb_cur_y);
                if (fabs(rb_prev_y - rb_cur_y) >= rtol_y) {
                    // update rb
                    rb_cur_y = 0.5 * (rb_cur_y + rb_prev_y);
                }
                rb_prev_y = temp;
                temp = rb_cur_z;
                // update lower bound
                lowerbd_z = std::max(lowerbd_z, rb_cur_z);
                if (fabs(rb_prev_z - rb_cur_z) >= rtol_z) {
                    // update rb
                    rb_cur_z = 0.5 * (rb_cur_z + rb_prev_z);
                }
                rb_prev_z = temp;
            }
        }
        
        Ncube_x=ceil(rb_cur_x/dx-1e-12); 
        Ncube_y=ceil(rb_cur_y/dy-1e-12); 
        Ncube_z=ceil(rb_cur_z/dz-1e-12);
        
        // integrate b over the cell of length rb_cur
        Bint = 0.0;
        for(k = 0; k < Ncube_z; k++)
            for(j = 0; j < Ncube_y; j++)
                for(i = 0; i < Ncube_x; i++) {
                    int index = b.get_vertices().get_index_nocheck(i,j,k);
                    val = b_data[index];
                    if (k == 0) val *= 0.5;
                    if (k == Ncube_z-1) val *= 0.5;
                    if(j == 0) val *= 0.5;
                    if(j == Ncube_y-1) val *= 0.5;
                    if(i == 0) val *= 0.5;
                    if(i == Ncube_x-1) val *= 0.5;
                    Bint += val;
                }
        Bint = Bint / (-4*M_PI) * 8.0 *  this->control.mesh_control.delta_V;
        
        //error_cur = fabs(Bint + pSPARC->Znucl[ityp]) / fabs(pSPARC->Znucl[ityp]); 
        error_cur = fabs(Bint + psp8_file.zion); 
        count++;
    }
    
    // after rb search is done, update rb projected to grid
    Ncube_x=ceil(rb_cur_x/dx-1e-12); 
    Ncube_y=ceil(rb_cur_y/dy-1e-12); 
    Ncube_z=ceil(rb_cur_z/dz-1e-12);
    
    // evaluate the error again if necessary
    if (errflag == 1) {     
        // after rb search is done, update rb projected to grid
        Bint = 0.0;
        // integrate b over cuboidal domain with rb_cur
        for(k = 0; k < Ncube_z; k++)
            for(j = 0; j < Ncube_y; j++)
                for(i = 0; i < Ncube_x; i++) {
                    int index = b.get_vertices().get_index_nocheck(i,j,k);
                    val = b_data[index];
                    if (k == 0) val *= 0.5;
                    if (k == Ncube_z-1) val *= 0.5;
                    if(j == 0) val *= 0.5;
                    if(j == Ncube_y-1) val *= 0.5;
                    if(i == 0) val *= 0.5;
                    if(i == Ncube_x-1) val *= 0.5;
                    Bint += val;
                }
        Bint = Bint / (-4*M_PI) * 8.0 * this->control.mesh_control.delta_V;
        
        //error_cur = fabs(Bint + pSPARC->Znucl[ityp]) / fabs(pSPARC->Znucl[ityp]);
        error_cur = fabs(Bint + psp8_file.zion);
    }

    if (rb_cur_x == Rbmax_x || rb_cur_y == Rbmax_y || rb_cur_z == Rbmax_z) {
        if (rank == 0 && error_cur > TOL_PSEUDOCHARGE) {
            printf("\nerror = %.2e > TOL_PSEUDOCHARGE = %.2e\n",error_cur,TOL_PSEUDOCHARGE);
            printf("\nWARNING: upperbond for pseudocharge radius (Rbmax_?) is not big enough! Rbmax_x = %.1f, Rbmax_y = %.1f, Rbmax_z = %.1f\n\n",Rbmax_x,Rbmax_y,Rbmax_z);
        }
    }

    psp8_file.charge_cut_x = Ncube_x * dx;
    psp8_file.charge_cut_y = Ncube_y * dy;
    psp8_file.charge_cut_z = Ncube_z * dz;

#undef RB_MAX
    return;
}