#include "effective_potential_nloc.h"

template<typename T>
Effective_potential_nloc<T>::Effective_potential_nloc(const Geometry& geometry,
                                                      const Mesh_control& mesh_control,
                                                      const std::vector<Psp8_file>& psp8_files,
                                                      const Domain_parallel_vertices_3D& domain_vertices)
                                                    : geometry(geometry),
                                                      mesh_control(mesh_control),
                                                      psp8_files(psp8_files),
                                                      domain_vertices(domain_vertices) {}

template<typename T>
Effective_potential_nloc<T>::~Effective_potential_nloc() {}

// template<typename T>
// void Effective_potential_nloc<T>::generate_nloc_projectors2() {
//     assert(this->geometry.cell_type <= 2);
//     int atom_index = 0;
//     for (std::vector<Atom>::const_iterator ptr = this->geometry.atoms.cbegin(); ptr != this->geometry.atoms.cend(); ++ptr) {
//         const Psp8_file& psp8_file = this->psp8_files[ptr->type];
//         double rc = psp8_file.r_core[0];
//         double rcbox_x, rcbox_y, rcbox_z;
//         if (this->geometry.cell_type <= 2) {
//             for (int i = 1; i <= psp8_file.lmax; i++) {
//                 if (rc < psp8_file.r_core[i]) rc = psp8_file.r_core[i];
//             }
//             rcbox_x = rc;
//             rcbox_y = rc;
//             rcbox_z = rc;
//         } else {
//             assert(this->geometry.cell_type <= 2);
//         }
//         std::vector<Atom> image_atoms;
//         ptr->generate_valid_images(image_atoms, this->geometry.cell_type, this->domain_vertices, this->mesh_control.is_periodic,
//                                    this->mesh_control.delta_x, this->mesh_control.delta_y, this->mesh_control.delta_z,
//                                    rcbox_x, rcbox_y, rcbox_z);
//         std::vector<Nloc_projector<T>> image_nloc_projectors;
//         image_nloc_projectors.reserve(image_atoms.size());
//         for (std::vector<Atom>::iterator it = image_atoms.begin(); it != image_atoms.end(); ++it) {
//             int cell_shift_x = std::floor(it->x / this->geometry.a);
//             int cell_shift_y = std::floor(it->y / this->geometry.b);
//             int cell_shift_z = std::floor(it->z / this->geometry.c);
//             Vertices_3D atom_vertices =
//                                 this->domain_vertices.get_3D_local_vertices().get_overlap_vertices(it->generate_rc_vertices(
//                                 this->geometry.cell_type, this->mesh_control.delta_x, this->mesh_control.delta_y,
//                                 this->mesh_control.delta_z, rcbox_x, rcbox_y, rcbox_z));
//             if (atom_vertices.get_size() == 0) continue;
//             // image_nloc_projectors.emplace_back(it->generate_nloc_projector_chi<T>(
//             //     this->geometry.cell_type, psp8_file, atom_vertices, this->domain_vertices.get_3D_local_vertices(),
//             //     this->mesh_control.delta_x, this->mesh_control.delta_y, this->mesh_control.delta_z,
//             //     atom_index, cell_shift_x, cell_shift_y, cell_shift_z));

//                 this->nloc_projectors.emplace_back(it->generate_nloc_projector_chi<T>(
//                     this->geometry.cell_type, psp8_file, atom_vertices, this->domain_vertices.get_3D_local_vertices(),
//                     this->mesh_control.delta_x, this->mesh_control.delta_y, this->mesh_control.delta_z,
//                         atom_index, cell_shift_x, cell_shift_y, cell_shift_z));
//                 this->offsets.emplace_back(this->offsets.back());
//         }
//         // this->nloc_projectors.emplace_back(Nloc_projector_method::merge<T>(image_nloc_projectors, psp8_file));
//         // this->offsets.emplace_back((this->nloc_projectors.end() - 1)->ncol + *(this->offsets.end() - 1));

//         this->offsets.back() += (this->nloc_projectors.end() - 1)->ncol;

//         atom_index++;
//     }
//     return;
// }

template<typename T>
void Effective_potential_nloc<T>::generate_nloc_projectors() {
    assert(this->geometry.cell_type <= 2);
    std::vector<Atom> image_atoms;

    std::vector<int> offsets_index;
    offsets_index.reserve(this->geometry.natom + 1);
    offsets_index.emplace_back(0);

    for (std::vector<Atom>::const_iterator ptr = this->geometry.atoms.cbegin(); ptr != this->geometry.atoms.cend(); ++ptr) {
        const Psp8_file& psp8_file = this->psp8_files[ptr->type];
        offsets_index.emplace_back(offsets_index.back()+psp8_file.generate_ncol());
        double rc = psp8_file.r_core[0];
        double rcbox_x = 0.0;
        double rcbox_y = 0.0;
        double rcbox_z = 0.0;
        if (this->geometry.cell_type <= 2) {
            for (int i = 1; i <= psp8_file.lmax; i++) {
                if (rc < psp8_file.r_core[i]) rc = psp8_file.r_core[i];
            }
            rcbox_x = rc;
            rcbox_y = rc;
            rcbox_z = rc;
        } else {
            assert(this->geometry.cell_type <= 2);
        }
        ptr->generate_valid_images(image_atoms, this->geometry.cell_type, this->domain_vertices, this->mesh_control.is_periodic,
                                   this->mesh_control.delta_x, this->mesh_control.delta_y, this->mesh_control.delta_z,
                                   this->geometry.a, this->geometry.b, this->geometry.c,
                                   rcbox_x, rcbox_y, rcbox_z);
    }
    int count = 0;
    for (std::vector<Atom>::iterator it = image_atoms.begin(); it != image_atoms.end(); ++it) {
        const Psp8_file& psp8_file = this->psp8_files[it->type];
        double rc = psp8_file.r_core[0];
        double rcbox_x = 0.0;
        double rcbox_y = 0.0;
        double rcbox_z = 0.0;
        if (this->geometry.cell_type <= 2) {
            for (int i = 1; i <= psp8_file.lmax; i++) {
                if (rc < psp8_file.r_core[i]) rc = psp8_file.r_core[i];
            }
            rcbox_x = rc;
            rcbox_y = rc;
            rcbox_z = rc;
        } else {
            assert(this->geometry.cell_type <= 2);
        }
        Vertices_3D atom_vertices =
                        this->domain_vertices.get_3D_local_vertices().get_overlap_vertices(it->generate_rc_vertices(
                        this->geometry.cell_type, this->mesh_control.delta_x, this->mesh_control.delta_y,
                        this->mesh_control.delta_z, rcbox_x, rcbox_y, rcbox_z));
        if (atom_vertices.get_size() == 0) continue;
        count ++;
    }

    this->offsets.reserve(count+1);

    for (std::vector<Atom>::iterator it = image_atoms.begin(); it != image_atoms.end(); ++it) {
        const Psp8_file& psp8_file = this->psp8_files[it->type];
        double rc = psp8_file.r_core[0];
        double rcbox_x = 0.0;
        double rcbox_y = 0.0;
        double rcbox_z = 0.0;
        if (this->geometry.cell_type <= 2) {
            for (int i = 1; i <= psp8_file.lmax; i++) {
                if (rc < psp8_file.r_core[i]) rc = psp8_file.r_core[i];
            }
            rcbox_x = rc;
            rcbox_y = rc;
            rcbox_z = rc;
        } else {
            assert(this->geometry.cell_type <= 2);
        }
        Vertices_3D atom_vertices =
                            this->domain_vertices.get_3D_local_vertices().get_overlap_vertices(it->generate_rc_vertices(
                            this->geometry.cell_type, this->mesh_control.delta_x, this->mesh_control.delta_y,
                            this->mesh_control.delta_z, rcbox_x, rcbox_y, rcbox_z));
        if (atom_vertices.get_size() == 0) continue;
        int cell_shift_x = std::floor(it->x / this->geometry.a);
        int cell_shift_y = std::floor(it->y / this->geometry.b);
        int cell_shift_z = std::floor(it->z / this->geometry.c);
        bool is_in_domain = false;
        if (this->geometry.cell_type <= 2) {
            double mesh_pos_x = (double)it->x / this->mesh_control.delta_x;
            double mesh_pos_y = (double)it->y / this->mesh_control.delta_y;
            double mesh_pos_z = (double)it->z / this->mesh_control.delta_z;
            if (mesh_pos_x >= (double)this->domain_vertices.get_3D_local_vertices().get_is()
             && mesh_pos_x < (double)(this->domain_vertices.get_3D_local_vertices().get_ie()+1)
             && mesh_pos_y >= (double)this->domain_vertices.get_3D_local_vertices().get_js()
             && mesh_pos_y < (double)(this->domain_vertices.get_3D_local_vertices().get_je()+1)
             && mesh_pos_z >= (double)this->domain_vertices.get_3D_local_vertices().get_ks()
             && mesh_pos_z < (double)(this->domain_vertices.get_3D_local_vertices().get_ke()+1))
             is_in_domain = true;
        } else {
            assert(this->geometry.cell_type <= 2);
        }
        Nloc_projector<T> nloc_projector_temp = it->generate_nloc_projector_chi<T>(
                this->geometry.cell_type, psp8_file, atom_vertices, this->domain_vertices.get_3D_local_vertices(),
                this->mesh_control.delta_x, this->mesh_control.delta_y, this->mesh_control.delta_z,
                    it->index, cell_shift_x, cell_shift_y, cell_shift_z, is_in_domain);
        #ifdef USE_OPENMP
        omp_set_max_active_levels(2);
        #pragma omp parallel num_threads(1)
        this->nloc_projectors.emplace_back(std::move(nloc_projector_temp));
        omp_set_max_active_levels(1);
        #else
        this->nloc_projectors.emplace_back(nloc_projector_temp);
        #endif
        this->offsets.emplace_back(offsets_index[it->index]);
    }
    this->offsets.emplace_back(offsets_index.back());
    return;
}

template<typename T>
void Effective_potential_nloc<T>::generate_nloc_projectors_mp(Memory_pool<T, Fast_memory>& pool_fast,
                                                            Memory_pool<T, Capacity_memory>& pool_cap) {
    (void) pool_fast;
    (void) pool_cap;
    assert(this->geometry.cell_type <= 2);
    std::vector<Atom> image_atoms;

    std::vector<int> offsets_index;
    offsets_index.reserve(this->geometry.natom + 1);
    offsets_index.emplace_back(0);

    for (std::vector<Atom>::const_iterator ptr = this->geometry.atoms.cbegin(); ptr != this->geometry.atoms.cend(); ++ptr) {
        const Psp8_file& psp8_file = this->psp8_files[ptr->type];
        offsets_index.emplace_back(offsets_index.back()+psp8_file.generate_ncol());
        double rc = psp8_file.r_core[0];
        double rcbox_x, rcbox_y, rcbox_z;
        if (this->geometry.cell_type <= 2) {
            for (int i = 1; i <= psp8_file.lmax; i++) {
                if (rc < psp8_file.r_core[i]) rc = psp8_file.r_core[i];
            }
            rcbox_x = rc;
            rcbox_y = rc;
            rcbox_z = rc;
        } else {
            rcbox_x = 0.0;
            rcbox_y = 0.0;
            rcbox_z = 0.0;
            assert(this->geometry.cell_type <= 2);
        }
        ptr->generate_valid_images(image_atoms, this->geometry.cell_type, this->domain_vertices, this->mesh_control.is_periodic,
                                   this->mesh_control.delta_x, this->mesh_control.delta_y, this->mesh_control.delta_z,
                                   this->geometry.a, this->geometry.b, this->geometry.c,
                                   rcbox_x, rcbox_y, rcbox_z);
    }
    int count = 0;
    for (std::vector<Atom>::iterator it = image_atoms.begin(); it != image_atoms.end(); ++it) {
        const Psp8_file& psp8_file = this->psp8_files[it->type];
        double rc = psp8_file.r_core[0];
        double rcbox_x, rcbox_y, rcbox_z;
        if (this->geometry.cell_type <= 2) {
            for (int i = 1; i <= psp8_file.lmax; i++) {
                if (rc < psp8_file.r_core[i]) rc = psp8_file.r_core[i];
            }
            rcbox_x = rc;
            rcbox_y = rc;
            rcbox_z = rc;
        } else {
            rcbox_x = 0.0;
            rcbox_y = 0.0;
            rcbox_z = 0.0;
            assert(this->geometry.cell_type <= 2);
        }
        Vertices_3D atom_vertices =
                        this->domain_vertices.get_3D_local_vertices().get_overlap_vertices(it->generate_rc_vertices(
                        this->geometry.cell_type, this->mesh_control.delta_x, this->mesh_control.delta_y,
                        this->mesh_control.delta_z, rcbox_x, rcbox_y, rcbox_z));
        if (atom_vertices.get_size() == 0) continue;
        count ++;
    }

    this->offsets.reserve(count+1);

    for (std::vector<Atom>::iterator it = image_atoms.begin(); it != image_atoms.end(); ++it) {
        const Psp8_file& psp8_file = this->psp8_files[it->type];
        double rc = psp8_file.r_core[0];
        double rcbox_x, rcbox_y, rcbox_z;
        if (this->geometry.cell_type <= 2) {
            for (int i = 1; i <= psp8_file.lmax; i++) {
                if (rc < psp8_file.r_core[i]) rc = psp8_file.r_core[i];
            }
            rcbox_x = rc;
            rcbox_y = rc;
            rcbox_z = rc;
        } else {
            rcbox_x = 0.0;
            rcbox_y = 0.0;
            rcbox_z = 0.0;
            assert(this->geometry.cell_type <= 2);
        }
        Vertices_3D atom_vertices =
                            this->domain_vertices.get_3D_local_vertices().get_overlap_vertices(it->generate_rc_vertices(
                            this->geometry.cell_type, this->mesh_control.delta_x, this->mesh_control.delta_y,
                            this->mesh_control.delta_z, rcbox_x, rcbox_y, rcbox_z));
        if (atom_vertices.get_size() == 0) continue;
        int cell_shift_x = std::floor(it->x / this->geometry.a);
        int cell_shift_y = std::floor(it->y / this->geometry.b);
        int cell_shift_z = std::floor(it->z / this->geometry.c);
        bool is_in_domain = false;
        if (this->geometry.cell_type <= 2) {
            double mesh_pos_x = (double)it->x / this->mesh_control.delta_x;
            double mesh_pos_y = (double)it->y / this->mesh_control.delta_y;
            double mesh_pos_z = (double)it->z / this->mesh_control.delta_z;
            if (mesh_pos_x >= (double)this->domain_vertices.get_3D_local_vertices().get_is()
             && mesh_pos_x < (double)(this->domain_vertices.get_3D_local_vertices().get_ie()+1)
             && mesh_pos_y >= (double)this->domain_vertices.get_3D_local_vertices().get_js()
             && mesh_pos_y < (double)(this->domain_vertices.get_3D_local_vertices().get_je()+1)
             && mesh_pos_z >= (double)this->domain_vertices.get_3D_local_vertices().get_ks()
             && mesh_pos_z < (double)(this->domain_vertices.get_3D_local_vertices().get_ke()+1))
             is_in_domain = true;
        } else {
            assert(this->geometry.cell_type <= 2);
        }
        Nloc_projector<T> nloc_projector_temp = it->generate_nloc_projector_chi<T>(
                this->geometry.cell_type, psp8_file, atom_vertices, this->domain_vertices.get_3D_local_vertices(),
                this->mesh_control.delta_x, this->mesh_control.delta_y, this->mesh_control.delta_z,
                    it->index, cell_shift_x, cell_shift_y, cell_shift_z, is_in_domain);
        // #ifdef USE_OPENMP
        // omp_set_max_active_levels(2);
        // #pragma omp parallel num_threads(1)
        // this->nloc_projectors.emplace_back(std::move(nloc_projector_temp));
        // omp_set_max_active_levels(1);
        // #else
        // this->nloc_projectors.emplace_back(nloc_projector_temp);
        // #endif
        this->nloc_projectors.emplace_back(Nloc_projector<T>(nloc_projector_temp.psp8_file));
        this->nloc_projectors.back().copy_mp(nloc_projector_temp, pool_fast, pool_cap);
        this->offsets.emplace_back(offsets_index[it->index]);
    }
    this->offsets.emplace_back(offsets_index.back());
    return;
}

template<typename T>
void Effective_potential_nloc<T>::init(const bool print_flag) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    uint size = this->geometry.atoms.size();
    this->nloc_projectors.reserve(size);
    this->offsets.reserve(size + 1);
    // this->offsets.emplace_back(0);
    this->generate_nloc_projectors();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (unlikely(this->domain_vertices.get_comm_rank() == 0 && print_flag)) {
        std::cout << "The Effective_potential_nloc init took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

template<typename T>
void Effective_potential_nloc<T>::init_mp(const bool print_flag,
                                        Memory_pool<T, Fast_memory>& pool_fast,
                                        Memory_pool<T, Capacity_memory>& pool_cap) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    uint size = this->geometry.atoms.size();
    this->nloc_projectors.reserve(size);
    this->offsets.reserve(size + 1);
    // this->offsets.emplace_back(0);
    this->generate_nloc_projectors_mp(pool_fast, pool_cap);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (unlikely(this->domain_vertices.get_comm_rank() == 0 && print_flag)) {
        std::cout << "The Effective_potential_nloc init took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}

template<typename T>
template<typename T2>
void Effective_potential_nloc<T>::init(const Effective_potential_nloc<T2>& effective_potential_nloc) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    this->nloc_projectors.reserve(effective_potential_nloc.nloc_projectors.size());
    for (typename std::vector<Nloc_projector<T2>>::const_iterator it = effective_potential_nloc.nloc_projectors.cbegin();
         it != effective_potential_nloc.nloc_projectors.cend(); ++it) {
        this->nloc_projectors.emplace_back(*it);
    }
    this->offsets = effective_potential_nloc.offsets;
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if (this->domain_vertices.get_comm_rank() == 0) {
        std::cout << "The Effective_potential_nloc init from other took " << Tools::time_cost(begin, end) << "." << std::endl;
    }
    return;
}
template void Effective_potential_nloc<float>::init(const Effective_potential_nloc<float>& effective_potential_nloc);
template void Effective_potential_nloc<double>::init(const Effective_potential_nloc<double>& effective_potential_nloc);
template void Effective_potential_nloc<float>::init(const Effective_potential_nloc<double>& effective_potential_nloc);
template void Effective_potential_nloc<double>::init(const Effective_potential_nloc<float>& effective_potential_nloc);

template<typename T>
void Effective_potential_nloc<T>::destructor() {
    std::vector<Nloc_projector<T>>().swap(this->nloc_projectors);
    std::vector<uint>().swap(this->offsets);
    return;
}

template<typename T>
void Effective_potential_nloc<T>::destructor_mp() {
    for (size_t i = 0; i < this->nloc_projectors.size(); i++) {
        this->nloc_projectors[i].destructor_mp();
    }
    std::vector<Nloc_projector<T>>().swap(this->nloc_projectors);
    std::vector<uint>().swap(this->offsets);
    return;
}

template<typename T>
void Effective_potential_nloc<T>::show(std::ostream& output) const {
    uint size = this->nloc_projectors.size();
    output << size << " nloc_projectors." << std::endl;
    for (uint i = 0; i < size; i++) {
        output << "for " << i << " nloc_projector:" << std::endl;
        output << "offset = " << this->offsets[i] << std::endl;
        this->nloc_projectors[i].show(output);
    }
    return;
}

template class Effective_potential_nloc<float>;
template class Effective_potential_nloc<double>;
