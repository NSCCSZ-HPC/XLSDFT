#ifdef USE_SSTRUCTMG
#include "wrapper_sstructmg.h"


#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

namespace {

    constexpr int solve_num_threads = 16;
    // The linked Semi-StructMG build uses MULTISTAGE_COMM.  Keep the probe
    // model explicit here because this wrapper is compiled separately from
    // the Semi-StructMG library and does not necessarily inherit its macro.
    constexpr bool sstructmg_uses_multistage_comm = true;
    // Keep this value shared by config generation and the pre-setup simulator.
    constexpr long long sstructmg_coarsest_size = 400;

template<typename idx_t>
int generated_num_levels(const idx_t *glb_dim) {
    long long dims[3] = {glb_dim[0], glb_dim[1], glb_dim[2]};
    int num_levels = 1;
    while (dims[0] * dims[1] * dims[2] > sstructmg_coarsest_size) {
        for (int d = 0; d < 3; ++d) dims[d] = (dims[d] + 1) / 2;
        ++num_levels;
    }
    return num_levels;
}

struct ForcedLinkBox {
    long long beg[3] = {0, 0, 0};
    long long end[3] = {0, 0, 0};
};

template<typename idx_t>
ForcedLinkBox make_forced_link_box(const idx_t *begs, const idx_t *ends) {
    ForcedLinkBox box;
    for (int d = 0; d < 3; ++d) {
        box.beg[d] = begs[d];
        box.end[d] = ends[d];
    }
    return box;
}

ForcedLinkBox coarsen_forced_link_box(const ForcedLinkBox &fine) {
    ForcedLinkBox coarse;
    for (int d = 0; d < 3; ++d) {
        coarse.beg[d] = (fine.beg[d] + 1) >> 1;
        coarse.end[d] = (fine.end[d] + 1) >> 1;
    }
    return coarse;
}

bool forced_box_active(const ForcedLinkBox &box) {
    return box.end[0] > box.beg[0] && box.end[1] > box.beg[1] && box.end[2] > box.beg[2];
}

enum ForcedIntervalRelation : unsigned {
    FORCED_DISJOINT = 0,
    FORCED_OWNED_OVERLAP = 1,
    FORCED_FACE_WITHIN_HALO = 2
};

inline unsigned forced_interval_relation_at_shift(
    long long target_beg, long long target_end,
    long long source_beg, long long source_end, long long halo
) {
    if (target_beg < source_end && source_beg < target_end)
        return FORCED_OWNED_OVERLAP;
    // Half-open boxes separated by exactly halo cells have zero halo overlap.
    if ((source_end <= target_beg && target_beg - source_end < halo) ||
        (source_beg >= target_end && source_beg - target_end < halo))
        return FORCED_FACE_WITHIN_HALO;
    return FORCED_DISJOINT;
}

inline unsigned forced_interval_relation(
    long long target_beg, long long target_end,
    long long source_beg, long long source_end,
    long long halo, bool periodic, long long extent
) {
    unsigned relation = forced_interval_relation_at_shift(
        target_beg, target_end, source_beg, source_end, halo);
    if (periodic) {
        relation |= forced_interval_relation_at_shift(
            target_beg, target_end, source_beg - extent, source_end - extent, halo);
        relation |= forced_interval_relation_at_shift(
            target_beg, target_end, source_beg + extent, source_end + extent, halo);
    }
    return relation;
}

bool forced_boxes_within_halo(const ForcedLinkBox &target, const ForcedLinkBox &source,
                              long long halo, const bool *periodic,
                              const long long extent[3]) {
    for (int d = 0; d < 3; ++d)
        if (forced_interval_relation(target.beg[d], target.end[d],
                                     source.beg[d], source.end[d],
                                     halo, periodic[d], extent[d]) == FORCED_DISJOINT)
            return false;
    return true;
}

bool forced_boxes_overlap(const ForcedLinkBox &a, const ForcedLinkBox &b) {
    return a.beg[0] < b.end[0] && b.beg[0] < a.end[0] &&
           a.beg[1] < b.end[1] && b.beg[1] < a.end[1] &&
           a.beg[2] < b.end[2] && b.beg[2] < a.end[2];
}

// MULTISTAGE_COMM links a face pair in one axis only when the owned intervals
// overlap in the other two.  Periodic shifts are independent by dimension, so
// three 1-D relation checks are exactly equivalent to enumerating 27 images.
bool forced_multistage_link(const ForcedLinkBox &target, const ForcedLinkBox &source,
                            long long halo, const bool *periodic,
                            const long long extent[3]) {
    unsigned relation[3];
    for (int d = 0; d < 3; ++d) {
        relation[d] = forced_interval_relation(
            target.beg[d], target.end[d], source.beg[d], source.end[d],
            halo, periodic[d], extent[d]);
        if (relation[d] == FORCED_DISJOINT) return false;
    }
    return ((relation[0] & FORCED_FACE_WITHIN_HALO) &&
            (relation[1] & FORCED_OWNED_OVERLAP) &&
            (relation[2] & FORCED_OWNED_OVERLAP)) ||
           ((relation[1] & FORCED_FACE_WITHIN_HALO) &&
            (relation[0] & FORCED_OWNED_OVERLAP) &&
            (relation[2] & FORCED_OWNED_OVERLAP)) ||
           ((relation[2] & FORCED_FACE_WITHIN_HALO) &&
            (relation[0] & FORCED_OWNED_OVERLAP) &&
            (relation[1] & FORCED_OWNED_OVERLAP));
}

struct ForcedLinkLevel {
    ForcedLinkBox local_box;
    long long extent[3];
};

template<typename idx_t>
double force_link_neighbors(
    MPI_Comm comm, const idx_t *glb_dim, const bool *is_period,
    const idx_t *glb_begs, const idx_t *glb_ends,
    const idx_t halo_len
) {
    int my_pid = 0, num_proc = 0;
    MPI_Comm_rank(comm, &my_pid);
    MPI_Comm_size(comm, &num_proc);
    const double calc_beg_time = wall_time();
    const int num_levels = generated_num_levels(glb_dim);
    std::vector<ForcedLinkLevel> levels(num_levels);
    const int local_offset = my_pid * 3;
    levels[0].local_box = make_forced_link_box(
        glb_begs + local_offset, glb_ends + local_offset);
    for (int d = 0; d < 3; ++d) levels[0].extent[d] = glb_dim[d];
    for (int level = 1; level < num_levels; ++level) {
        levels[level].local_box = coarsen_forced_link_box(levels[level - 1].local_box);
        for (int d = 0; d < 3; ++d)
            levels[level].extent[d] = (levels[level - 1].extent[d] + 1) >> 1;
    }

    // Each candidate rank is scanned once. Its box advances level by level;
    // ordinary stage links and fine-to-coarse HELP links share one result.
    std::vector<unsigned char> linked(num_proc, 0);
    #pragma omp parallel for schedule(static)
    for (int peer = 0; peer < num_proc; ++peer) {
        if (peer == my_pid) continue;
        const int peer_offset = peer * 3;
        ForcedLinkBox peer_fine = make_forced_link_box(
            glb_begs + peer_offset, glb_ends + peer_offset);
        for (int level = 0; level < num_levels; ++level) {
            const ForcedLinkBox &local_fine = levels[level].local_box;
            if (!forced_box_active(local_fine) || !forced_box_active(peer_fine)) break;

            const bool setup_link = sstructmg_uses_multistage_comm
                ? forced_multistage_link(local_fine, peer_fine, halo_len,
                                         is_period, levels[level].extent)
                : forced_boxes_within_halo(local_fine, peer_fine, halo_len,
                                           is_period, levels[level].extent);
            if (setup_link) {
                linked[peer] = 1;
                break;
            }
            if (level + 1 == num_levels) break;

            const ForcedLinkBox peer_coarse = coarsen_forced_link_box(peer_fine);
            const ForcedLinkBox &local_coarse = levels[level + 1].local_box;
            const bool local_coarse_active = forced_box_active(local_coarse);
            const bool peer_coarse_active = forced_box_active(peer_coarse);
            if (local_coarse_active != peer_coarse_active &&
                forced_boxes_within_halo(local_fine, peer_fine, 1,
                                         is_period, levels[level].extent)) {
                const ForcedLinkBox &active_coarse =
                    local_coarse_active ? local_coarse : peer_coarse;
                const ForcedLinkBox &inactive_fine =
                    local_coarse_active ? peer_fine : local_fine;
                const ForcedLinkBox active_cover = {
                    {active_coarse.beg[0] * 2, active_coarse.beg[1] * 2,
                     active_coarse.beg[2] * 2},
                    {std::min(levels[level].extent[0], active_coarse.end[0] * 2),
                     std::min(levels[level].extent[1], active_coarse.end[1] * 2),
                     std::min(levels[level].extent[2], active_coarse.end[2] * 2)}};
                // coarse_vec.cpp intersects these two unshifted global boxes.
                if (forced_boxes_overlap(active_cover, inactive_fine)) {
                    linked[peer] = 1;
                    break;
                }
            }
            peer_fine = peer_coarse;
        }
    }

    std::vector<int> peers;
    for (int peer = 0; peer < num_proc; ++peer)
        if (linked[peer]) peers.push_back(peer);
    MPI_Barrier(comm);
    const double t0 = wall_time();
    if (my_pid == 0) {
        printf("JZPDEBUG: forced MPI link probe: calc time (rnk0) %.6f s, model=%s, setup_halo=%d, help_halo=1\n",
               t0-calc_beg_time,
               sstructmg_uses_multistage_comm ? "multistage-face" : "full-halo",
               static_cast<int>(halo_len));
        fflush(stdout);
    }
    std::vector<unsigned char> send_bytes(peers.size(), 0xA5), recv_bytes(peers.size(), 0);
    std::vector<MPI_Request> requests(2 * peers.size(), MPI_REQUEST_NULL);
    size_t request_id = 0;
    for (size_t i = 0; i < peers.size(); ++i)
        MPI_Irecv(&recv_bytes[i], 1, MPI_BYTE, peers[i], 0, comm, &requests[request_id++]);
    for (size_t i = 0; i < peers.size(); ++i)
        MPI_Isend(&send_bytes[i], 1, MPI_BYTE, peers[i], 0, comm, &requests[request_id++]);
    if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    MPI_Barrier(comm);
    const double elapsed = wall_time() - t0;
    double max_elapsed = 0.0;
    int max_peer_count = 0, min_peer_count = num_proc;
    long long sum_peer_count = 0;
    const int local_peer_count = static_cast<int>(peers.size());
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_peer_count, &max_peer_count, 1, MPI_INT, MPI_MAX, 0, comm);
    MPI_Reduce(&local_peer_count, &min_peer_count, 1, MPI_INT, MPI_MIN, 0, comm);
    const long long local_peer_count_ll = local_peer_count;
    MPI_Reduce(&local_peer_count_ll, &sum_peer_count, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
    const double avg_peer_count = static_cast<double>(sum_peer_count) / num_proc;
    if (my_pid == 0) {
        printf("JZPDEBUG: forced MPI link probe: calc time (rnk0) %.6f s, Isend/recv (max) %.6f s, min|avg|max peers %d|%.1f|%d, simulated levels %d, coarse limit %lld\n",
               t0-calc_beg_time, max_elapsed, min_peer_count, avg_peer_count, max_peer_count, num_levels, sstructmg_coarsest_size);
        fflush(stdout);
    }
    return my_pid == 0 ? max_elapsed : 0.0;
}

/**
 * 辅助函数：将字符串 vector 格式化为 JSON 数组字符串
 */
std::string to_json_array(const std::vector<std::string>& vec, bool quotes = true) {
    std::string res = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (quotes) res += "\"" + vec[i] + "\"";
        else res += vec[i];
        if (i != vec.size() - 1) res += ", ";
    }
    res += "]";
    return res;
}

/**
 * 辅助函数：将 double vector 格式化为 JSON 数组字符串
 */
std::string to_json_array(const std::vector<double>& vec) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i != vec.size() - 1) oss << ", ";
    }
    oss << "]";
    return oss.str();
}

/**
 * 辅助函数：将 int vector 格式化为 JSON 数组字符串
 */
std::string to_json_array(const std::vector<int>& vec) {
    std::string res = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        res += std::to_string(vec[i]);
        if (i != vec.size() - 1) res += ", ";
    }
    res += "]";
    return res;
}

/**
 * 辅助函数：将 bool vector 格式化为 JSON 数组字符串
 */
std::string to_json_array(const std::vector<bool>& vec) {
    std::string res = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        res += (vec[i] ? "true" : "false");
        if (i != vec.size() - 1) res += ", ";
    }
    res += "]";
    return res;
}

void generate_mg_config(int a, int b, int c, std::string file_name) {
    // 1. 计算 num_levs (k + 1)
    int k = 0;
    while (true) {
        long long la = std::ceil(a / std::pow(2, k));
        long long lb = std::ceil(b / std::pow(2, k));
        long long lc = std::ceil(c / std::pow(2, k));
        if (la * lb * lc <= sstructmg_coarsest_size) break;
        k++;
    }
    int num_levs = k + 1;

    // 打印网格信息 (仿照 Python 打印)
    std::cout << "\n[Grid Coarsening Hierarchy]\n";
    std::cout << std::string(45, '-') << "\n";
    std::cout << std::left << std::setw(8) << "Level" << " | " 
              << std::left << std::setw(18) << "Dimensions" << " | Total Nodes\n";
    std::cout << std::string(45, '-') << "\n";
    for (int i = 0; i < num_levs; ++i) {
        int la = std::ceil(a / std::pow(2, i));
        int lb = std::ceil(b / std::pow(2, i));
        int lc = std::ceil(c / std::pow(2, i));
        std::cout << "lev #" << std::left << std::setw(3) << i << " | "
                  << std::right << std::setw(4) << la << " x " 
                  << std::right << std::setw(4) << lb << " x " 
                  << std::right << std::setw(4) << lc << "  | "
                  << std::right << std::setw(10) << (long long)la * lb * lc << "\n";
    }
    std::cout << std::string(45, '-') << "\n";

    // 2. 构造配置数据
    int shift_levid = num_levs + 6;

#ifdef USE_MG_3d7
    std::string smoother_weak = "PGS";
    std::string smoother_strong = "PGS";
    int lev_switch_to_advanced_smoother = 1;
    int lev_switch_to_smoothed_aggregation = 0;
#else
    std::string smoother_weak = "PJ";
    std::string smoother_strong = "PGS";
    int lev_switch_to_advanced_smoother = 3;  // BILU only supported for stencils that radius = 1.
    int lev_switch_to_smoothed_aggregation = 3;  // SA only supported for stencils that radius = 1.
#endif

    std::vector<std::string> smoother;
    for (int i = 0; i < num_levs - 1; ++i) {
        smoother.push_back(i < lev_switch_to_advanced_smoother ? smoother_weak : smoother_strong);
    }
    // 处理最后一层 smoother
    if (num_levs <= lev_switch_to_advanced_smoother) smoother.push_back(smoother_weak);
    else smoother.push_back(smoother_strong);

    // // 检查 GlbLU 条件
    // long long coarsest_size = std::ceil(a / std::pow(2, k)) * 
    //                           std::ceil(b / std::pow(2, k)) * 
    //                           std::ceil(c / std::pow(2, k));
    // if (coarsest_size <= 400) {
    //     smoother.back() = "GlbLU";
    // }

    std::vector<std::string> galerkin;
    for (int i = 0; i < num_levs - 1; ++i) {
        galerkin.push_back(i < lev_switch_to_smoothed_aggregation ? "RC3d8_PC3d8" : "RC3d8_PC3d8SIp");
    }

    std::vector<double> smwgt_exp = {0.67, 0.73, 0.75, 0.75, 0.75, 0.78, 0.78, 0.8, 0.8, 0.78, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8};
    std::vector<double> current_smwgt;
    for (int i = 0; i < num_levs - 1; ++i) {
        current_smwgt.push_back(smwgt_exp[i]);
    }

    // 3. 写入 JSON 文件
    fs::create_directories("./config");
    std::ofstream out(file_name);

    out << "{\n";
    out << "    \"num_levs\": " << num_levs << ",\n";
    out << "    \"shift_levid\": " << shift_levid << ",\n";
    out << "    \"cycle\": \"V\",\n";
    out << "    \"sweep\": [1, 1],\n";
    out << "    \"smoother\": " << to_json_array(smoother) << ",\n";
    out << "    \"weight\": " << to_json_array(std::vector<double>(num_levs, 1.0)) << ",\n";
    out << "    \"itb_intp\": " << to_json_array(std::vector<bool>(num_levs, true)) << ",\n";
    out << "    \"Block_*\": {\n";
    out << "        \"Coarsen\": " << to_json_array(std::vector<int>(num_levs - 1, 7)) << ",\n";
    out << "        \"restrict\": " << to_json_array(std::vector<std::string>(num_levs - 1, "Cell_3d8")) << ",\n";
    out << "        \"interp\": " << to_json_array(std::vector<std::string>(num_levs - 1, "Cell_3d8")) << ",\n";
    out << "        \"Galerkin\": " << to_json_array(galerkin) << ",\n";
    out << "        \"alpha\": " << to_json_array(std::vector<double>(num_levs - 1, 1.0)) << ",\n";
    out << "        \"smwgt\": " << to_json_array(current_smwgt) << "\n";
    out << "    }\n";
    out << "}\n";

    out.close();
    std::cout << "Config generated successfully.\n" << std::endl;
}

}  // generate config file


void SStructMG_generate_config_file(const int N0, const int N1, const int N2, std::string file_name) {
    generate_mg_config(N0, N1, N2, file_name);
}

template<typename idx_t, typename ksp_t, typename pc_data_t, typename pc_calc_t>
SStructMG_wrapper<idx_t, ksp_t, pc_data_t, pc_calc_t>::~SStructMG_wrapper() {
    destroy();
    SStructMG_Finalize();
}


template<typename idx_t, typename ksp_t, typename pc_data_t, typename pc_calc_t>
void SStructMG_wrapper<idx_t, ksp_t, pc_data_t, pc_calc_t>::init(
    MPI_Comm comm_, const idx_t *glb_dim, const bool *is_period,
    const idx_t *glb_begs, const idx_t *glb_ends,
    const idx_t *my_ilower, const idx_t *my_iupper, const ksp_t *stencil_value,
    const ksp_t *precond_stencil3d7_value, const TEST_CONFIG config,
    const bool as_precond_, const bool fine_grid_all_active
) {
    double time_stamp_0 = wall_time();

    assert(!is_initialized);
    mg_num_levels = generated_num_levels(glb_dim);
    int my_pid; MPI_Comm_rank(comm_, &my_pid);

    const bool PRINT_DEBUG_MESSAGE = false;
    if (PRINT_DEBUG_MESSAGE) {  // print debug message
        double time_print_debug = wall_time();
        int num_proc; MPI_Comm_size(comm_, &num_proc);
        if (my_pid == 0) printf("JZPDEBUG: stencil_value[0,1,2,...]={%.10e, %.10e, %.10e, ...}\n", stencil_value[0], stencil_value[1], stencil_value[2]), fflush(stdout);
        if (my_pid == 0) printf("JZPDEBUG: idx_t %ld, ksp_t %ld, pc_data_t %ld, pc_calc_t %ld\n", sizeof(idx_t), sizeof(ksp_t), sizeof(pc_data_t), sizeof(pc_calc_t)), fflush(stdout);

        int jzp_cnt = 0;
        for (int p = 0; p < num_proc; ++p) {
            if (jzp_cnt == 0 || p == num_proc-1) {
            MPI_Barrier(comm_);
            if (my_pid == p) {
                printf("JZPDEBUG: proc %d/%d, init(): comm_ = %lld, glb_dim = {%d, %d, %d}, is_period = {%d, %d, %d}, my_ilower = {%d, %d, %d}, my_iupper = {%d, %d, %d}\n",
                    my_pid, num_proc,
                    (long long)comm_, (int)glb_dim[0], (int)glb_dim[1], (int)glb_dim[2], (int)is_period[0], (int)is_period[1], (int)is_period[2],
                    (int)my_ilower[0], (int)my_ilower[1], (int)my_ilower[2], (int)my_iupper[0], (int)my_iupper[1], (int)my_iupper[2]
                ), fflush(stdout);
            }
            }
            jzp_cnt = (jzp_cnt + 1) % ((num_proc+9) / 10);
        }
        MPI_Barrier(comm_);
        time_print_debug = wall_time() - time_print_debug;
        if (my_pid == 0) printf("JZPDEBUG: print debug message costs %.6f s\n", time_print_debug), fflush(stdout);
    } else {
        MPI_Barrier(comm_);
    }

    double time_stamp_debug = wall_time();
    {  // split MPI comm
        const bool has_fine_box =
            my_ilower[0] <= my_iupper[0] &&
            my_ilower[1] <= my_iupper[1] &&
            my_ilower[2] <= my_iupper[2];
        if (fine_grid_all_active) {
            comm = comm_;
            owns_comm = false;
        } else {
            const int color = has_fine_box ? 0 : MPI_UNDEFINED;
            MPI_Comm_split(comm_, color, my_pid, &comm);
            if (comm == MPI_COMM_NULL) {
                is_initialized = true;
                return;
            }
            owns_comm = true;
        }
    }

    // Inactive ranks have returned; all collectives below must use active comm.
    double time_stamp_split = wall_time();
    // For MG's 3d7 inexact-preconditioner path, the matrix/coarse-vector halo
    // is 1, while smoothed RAP builds a temporary interpolation halo of 2.
    // Probe the latter so all MG setup links are warmed; HELP remains halo 1.
    // This is independent of as_precond_: GMRES+MG also supplies this stencil.
    const idx_t force_link_halo_len =
        precond_stencil3d7_value ? 2 : radius;
    const double forced_link_time = force_link_neighbors(
        comm, glb_dim, is_period, glb_begs, glb_ends,
        force_link_halo_len);
    const double time_stamp_force_link = wall_time();

    SStructMG_Init();
    box_beg[0] = my_ilower[0]; box_beg[1] = my_ilower[1]; box_beg[2] = my_ilower[2]; 
    box_end[0] = my_iupper[0]+1; box_end[1] = my_iupper[1]+1; box_end[2] = my_iupper[2]+1; 
    ssgrid = new SStructGrid<idx_t>(comm, 3, 1);  // 3 dimensions, 1 block

    if (PRINT_DEBUG_MESSAGE) {
        MPI_Barrier(comm);
        if (my_pid == 0) {
            printf("JZPDEBUG: wrapper_sstructmg ssgrid new ok\n"); fflush(stdout);
            std::time_t start_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::cout << "time: " << std::ctime(&start_time) << std::endl;
        }
    }
    {// 添加本进程所拥有的块和盒子
        ssgrid->addBlock(0, glb_dim, is_period);
        ssgrid->addBox(0, box_beg, box_end);
    }
    if (PRINT_DEBUG_MESSAGE) {
        MPI_Barrier(comm);
        if (my_pid == 0) {
            printf("JZPDEBUG: wrapper_sstructmg add block box ok\n"); fflush(stdout);
            std::time_t start_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::cout << "time: " << std::ctime(&start_time) << std::endl;
        }
    }
    // MG's solve vectors must match the finest 3d7 operator (halo 1).
    // The halo-2 requirement is only for temporary smoothed-RAP setup data
    // and is handled independently by force_link_halo_len above.
    const idx_t halo_len =
        (as_precond_ && precond_stencil3d7_value) ? 1 : radius;
    const idx_t halos[3] = {halo_len, halo_len, halo_len};
    ssgrid->assemble_with_glb_boxs(halos, glb_begs, glb_ends);

    if (PRINT_DEBUG_MESSAGE) {
        MPI_Barrier(comm);
        if (my_pid == 0) {
            printf("JZPDEBUG: wrapper_sstructmg assemble_with_glb_boxs ok\n"); fflush(stdout);
            std::time_t start_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::cout << "time: " << std::ctime(&start_time) << std::endl;
        }
    }

    double time_stamp_ssgrid = wall_time();

    idx_t shalos [my_nblks * 3];
    for (idx_t k = 0; k < my_nblks; k++) {
        shalos[k * 3 + 0] = halos[0];
        shalos[k * 3 + 1] = halos[1];
        shalos[k * 3 + 2] = halos[2];
    }
    vec_b = new par_SstructVector<idx_t, ksp_t, 1>(*ssgrid, shalos, false);
    vec_x = new par_SstructVector<idx_t, ksp_t, 1>(*vec_b);
    if (sizeof(pc_calc_t) != sizeof(ksp_t) && as_precond_)  {
        pc_buf_b = new par_SstructVector<idx_t, pc_calc_t, 1>(*ssgrid, shalos, false);
        pc_buf_x = new par_SstructVector<idx_t, pc_calc_t, 1>(*pc_buf_b);
    }

    if (PRINT_DEBUG_MESSAGE) {
        MPI_Barrier(comm);
        if (my_pid == 0) {
            printf("JZPDEBUG: wrapper_sstructmg Vector construct ok\n"); fflush(stdout);
            std::time_t start_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::cout << "time: " << std::ctime(&start_time) << std::endl;
        }
    }

    double time_stamp_vec = wall_time();

    // 根据(半)结构网格建立矩阵
    idx_t mat_shalos [my_nblks * 3];
    for (idx_t k = 0; k < my_nblks; k++) {  // non-smoothed interp. should set halo = 1
        mat_shalos[k * 3 + 0] = 1;
        mat_shalos[k * 3 + 1] = 1;
        mat_shalos[k * 3 + 2] = 1;
    }
    
    if (stencil_value) {// 填入矩阵数据
        __int128_t masks[my_nblks]; for (idx_t k = 0; k < my_nblks; k++) masks[k] = _mask;
        mat_A = new par_SstructMatrix<idx_t, ksp_t, ksp_t, 1>(*ssgrid, mat_shalos, masks);
        mat_A->init_offd();// 初始化非对角部分
        {
            for (int d = 0; d < num_diag; ++d)
                mat_A->diag_matrixs[0]->set_diag_val(d, stencil_value[d]);

            mat_A->diag_matrixs[0]->set_boundary();  // zero out the edges that are going out of the box
        }

        // 然后处理非对角快，也即非结构部分
        // do nothing

        mat_A->assemble();
    }
    if (PRINT_DEBUG_MESSAGE) {
        MPI_Barrier(comm);
        if (my_pid == 0) {
            printf("JZPDEBUG: wrapper_sstructmg stencil values filled ok\n"); fflush(stdout);
            std::time_t start_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::cout << "time: " << std::ctime(&start_time) << std::endl;
        }
    }

    if (precond_stencil3d7_value) {// 填入矩阵数据
        __int128_t masks[my_nblks]; for (idx_t k = 0; k < my_nblks; k++) masks[k] = _mask3d7;
        precond_mat_A = new par_SstructMatrix<idx_t, ksp_t, ksp_t, 1>(*ssgrid, mat_shalos, masks);
        precond_mat_A->init_offd();// 初始化非对角部分
        {
            for (int d = 0; d < 7; ++d)
                precond_mat_A->diag_matrixs[0]->set_diag_val(d, precond_stencil3d7_value[d]);

            precond_mat_A->diag_matrixs[0]->set_boundary();  // zero out the edges that are going out of the box
        }

        // 然后处理非对角快，也即非结构部分
        // do nothing

        precond_mat_A->assemble();
    }
    if (PRINT_DEBUG_MESSAGE) {
        MPI_Barrier(comm);
        if (my_pid == 0) {
            printf("JZPDEBUG: wrapper_sstructmg precond stencil3d7 values filled ok\n"); fflush(stdout);
            std::time_t start_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::cout << "time: " << std::ctime(&start_time) << std::endl;
        }
    }

    double time_stamp_mat = wall_time();

    as_precond = as_precond_;
    if (as_precond) {
        buildup_MG(config);
    } else {
        std::string its_name = config.its_name;
        // std::string its_name = "CG";
        std::string prc_name = "GMG";
        buildup_solver(its_name, prc_name, config);

        // setup solver
        solver->SetSolverOperator((*mat_A));
    }
    if (PRINT_DEBUG_MESSAGE) {
        MPI_Barrier(comm);
        if (my_pid == 0) {
            printf("JZPDEBUG: wrapper_sstructmg buildup MG & solver ok\n"); fflush(stdout);
            std::time_t start_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::cout << "time: " << std::ctime(&start_time) << std::endl;
        }
    }
    double time_stamp_buildup = wall_time();
    // setup precond
    const idx_t uniform_blk_cnt = 1;
    const idx_t uniform_blk_ids[1] = {0};
    pc_calc_t stencil_value_pc[num_diag];
    if (precond_stencil3d7_value) {
        for (idx_t i = 0; i < 7; ++i) stencil_value_pc[i] = precond_stencil3d7_value[i];  // ksp_t -> pc_calc_t
    } else {
        for (idx_t i = 0; i < num_diag; ++i) stencil_value_pc[i] = stencil_value[i];  // ksp_t -> pc_calc_t
    }
    const pc_calc_t *uniform_stencil_values[1] = {stencil_value_pc};
    using MG = MultiGrid<idx_t, pc_data_t, pc_calc_t, ksp_t, 1>;
    ((MG*)precond)->SetOperatorAndConvertFinestToUniform(precond_stencil3d7_value ? (*precond_mat_A) : (*mat_A), uniform_blk_cnt, uniform_blk_ids, uniform_stencil_values);

    double time_stamp_setup = wall_time();

    if (mat_A) mat_A->convert_to_uniform(0, stencil_value);
    if (precond_mat_A) precond_mat_A->convert_to_uniform(0, precond_stencil3d7_value);

    double time_stamp_convert = wall_time();

    MPI_Barrier(comm);
    double time_stamp_1 = wall_time();

    {  // warming up
        idx_t local_dim[3] = {my_iupper[0]-my_ilower[0]+1, my_iupper[1]-my_ilower[1]+1, my_iupper[2]-my_ilower[2]+1};
        idx_t tot_elem = local_dim[0] * local_dim[1] * local_dim[2];
        std::vector<ksp_t> rhs_data(tot_elem);
        const double PI = 3.1415926535897932384626433832795;
        #pragma omp parallel for collapse(2) schedule(static)
        for (idx_t i0 = my_ilower[0]; i0 <= my_iupper[0]; ++i0)
        for (idx_t i1 = my_ilower[1]; i1 <= my_iupper[1]; ++i1)
            for (idx_t i2 = my_ilower[2]; i2 <= my_iupper[2]; ++i2)
                rhs_data[ ((i0-my_ilower[0]) * local_dim[1] + (i1-my_ilower[1])) * local_dim[2] + (i2-my_ilower[2]) ]
                = -std::sin(2.0 * PI * i0 / glb_dim[0]) * std::sin(2.0 * PI * i1 / glb_dim[1]) * std::sin(2.0 * PI * i2 / glb_dim[2]);
        std::vector<ksp_t> x_data(tot_elem, 0.0);
        solve(rhs_data.data(), x_data.data());
        show_breakdown_and_reset();
    }

    is_initialized = true;

    MPI_Barrier(comm);
    double time_stamp_2 = wall_time();
    MPI_Barrier(comm);
    double time_stamp_3 = wall_time();
    if (my_pid == 0) {
        printf("MG init and warming up are finished.\nSetup took: %.6f s\nWarming up took %.6f s\n (One MPI_Barrier takes %.6f s)\n\n",
            time_stamp_1 - time_stamp_0, time_stamp_2 - time_stamp_1, time_stamp_3 - time_stamp_2);
        printf("Setup: split %.6f s, force_link %.6f s (MPI max %.6f s), ssgrid %.6f s, vec %.6f s, mat %.6f s, buildup %.6f s, setup %.6f s, convert %.6f s, barrier %.6f s.\n",
            time_stamp_split - time_stamp_debug,
            time_stamp_force_link - time_stamp_split,
            forced_link_time,
            time_stamp_ssgrid - time_stamp_force_link,
            time_stamp_vec - time_stamp_ssgrid,
            time_stamp_mat - time_stamp_vec,
            time_stamp_buildup - time_stamp_mat,
            time_stamp_setup - time_stamp_buildup,
            time_stamp_convert - time_stamp_setup,
            time_stamp_1 - time_stamp_convert
        );
        fflush(stdout);
    }
}


template<typename idx_t, typename ksp_t, typename pc_data_t, typename pc_calc_t>
void SStructMG_wrapper<idx_t, ksp_t, pc_data_t, pc_calc_t>::buildup_MG(const TEST_CONFIG & config_file) {
    assert(as_precond);
    assert(precond== nullptr);

    SStructMG_set_print_level(config_file.print_level);

    using MG = MultiGrid<idx_t, pc_data_t, pc_calc_t, ksp_t, 1>;
    MG *mg = nullptr;
    if (config_file.use_in_memory_mg_config) {
        mg = new MG();
        mg->SetDefault3DConfig(*ssgrid, mg_num_levels, precond_mat_A == nullptr ? 6 : 1);  // star3d6r if precond_mat_A not exists
    } else {
        const std::string &path = config_file.config_mg_file;
        assert(std::filesystem::exists(path) && std::filesystem::is_regular_file(path));
        mg = new MG(path);
        mg->ParseConfig(*ssgrid);
    }
    precond = mg;
}


template<typename idx_t, typename ksp_t, typename pc_data_t, typename pc_calc_t>
void SStructMG_wrapper<idx_t, ksp_t, pc_data_t, pc_calc_t>::buildup_solver(std::string its_name, std::string prc_name, const TEST_CONFIG & config_file)
{
    assert(!as_precond);
    assert(solver == nullptr);
    assert(precond== nullptr);

    SStructMG_set_print_level(config_file.print_level);
    
    int my_pid; MPI_Comm_rank(comm, &my_pid);
    const int restart = config_file.restart_len;
    // 设置求解器
    if (its_name == "CG") {
        solver = new CGSolver<idx_t, pc_data_t, pc_calc_t, ksp_t, 1>;
    } else if (its_name == "GMRES") {
        using GMRES = GMRESSolver<idx_t, pc_data_t, pc_calc_t, ksp_t, 1>;
        solver = new GMRES();
        ((GMRES*)solver)->SetRestartlen(restart);
    } else {
        if (my_pid == 0) printf("INVALID iterative solver name of %s\nOnly GCR, CG, GMRES, FGMRES available\n", its_name.c_str());
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    solver->SetMaxIter(config_file.max_iter);
    solver->SetRelTol(config_file.rtol);
    solver->SetAbsTol(config_file.atol);
    if (prc_name == "GMG") {
        using MG = MultiGrid<idx_t, pc_data_t, pc_calc_t, ksp_t, 1>;
        MG *mg = nullptr;
        if (config_file.use_in_memory_mg_config) {
            mg = new MG();
            mg->SetDefault3DConfig(*ssgrid, mg_num_levels, precond_mat_A == nullptr ? 6 : 1);  // star3d6r if precond_mat_A not exists
        } else {
            const std::string &path = config_file.config_mg_file;
            assert(std::filesystem::exists(path) && std::filesystem::is_regular_file(path));
            mg = new MG(path);
            mg->ParseConfig(*ssgrid);
        }
        precond = mg;
    }
    else {
        if (my_pid == 0) printf("No Precond!\n");
    }

    if (precond != nullptr) solver->SetPreconditioner(*precond);
}

// 定义一个 RAII 守卫类
class OmpThreadGuard {
private:
    int old_num_threads;
public:
    // 构造时保存旧值并设置新线程数
    OmpThreadGuard(int num_threads) {
        old_num_threads = omp_get_max_threads(); // 获取当前的默认线程数
        omp_set_num_threads(num_threads);
    }
    
    // 析构时自动恢复旧值
    ~OmpThreadGuard() {
        omp_set_num_threads(old_num_threads);
    }
    
    // 禁止拷贝
    OmpThreadGuard(const OmpThreadGuard&) = delete;
    OmpThreadGuard& operator=(const OmpThreadGuard&) = delete;
};

template<typename idx_t, typename ksp_t, typename pc_data_t, typename pc_calc_t>
idx_t SStructMG_wrapper<idx_t, ksp_t, pc_data_t, pc_calc_t>::solve(const ksp_t *rhs_data, ksp_t *x_data)
{
    if (comm == MPI_COMM_NULL) return -1;
    vec_b->set_box_values(0, box_beg, box_end, rhs_data);
    if (as_precond) {
        if constexpr (sizeof(pc_calc_t) != sizeof(ksp_t)) {  // can be optimized
            IterSolverVectorHelper<idx_t, ksp_t, pc_calc_t, 1>::VecTransPrec(*vec_b, *pc_buf_b);
            {
                OmpThreadGuard guard(solve_num_threads);
                precond->Mult(*pc_buf_b, *pc_buf_x, true);
            }
            IterSolverVectorHelper<idx_t, pc_calc_t, ksp_t, 1>::VecTransPrec(*pc_buf_x, *vec_x);
        } else {
            precond->Mult(*vec_b, *vec_x, true);  // zero_guess == true
        }
        vec_x->get_box_values(0, box_beg, box_end, x_data);
        return 1;  // only Mult once
    } else {
        vec_x->set_box_values(0, box_beg, box_end, x_data);
        {
            OmpThreadGuard guard(solve_num_threads);
            solver->Mult(*vec_b, *vec_x, false);  // zero_guess = false
        }
        vec_x->get_box_values(0, box_beg, box_end, x_data);
        idx_t iter = solver->GetNumIterations();
        return iter;
    }
}

template<typename idx_t, typename ksp_t, typename pc_data_t, typename pc_calc_t>
void SStructMG_wrapper<idx_t, ksp_t, pc_data_t, pc_calc_t>::show_breakdown_and_reset()
{
    if (comm == MPI_COMM_NULL) return ;
    if (solver) {
        double  min_times[NUM_KRYLOV_RECORD], 
                max_times[NUM_KRYLOV_RECORD],
                avg_times[NUM_KRYLOV_RECORD];
        int my_pid; MPI_Comm_rank(comm, &my_pid);
        int num_procs; MPI_Comm_size(comm, &num_procs);
        MPI_Allreduce(solver->part_times, min_times, NUM_KRYLOV_RECORD, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(solver->part_times, max_times, NUM_KRYLOV_RECORD, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(solver->part_times, avg_times, NUM_KRYLOV_RECORD, MPI_DOUBLE, MPI_SUM, comm); 
        for (int i = 0; i < NUM_KRYLOV_RECORD; i++)
            avg_times[i] /= num_procs;
        if (my_pid == 0 && SStructMG_print_level >= 2) {
            printf("prec time min/avg/max %.3e %.3e %.3e\n", min_times[PREC], avg_times[PREC], max_times[PREC]);
            printf("oper time min/avg/max %.3e %.3e %.3e\n", min_times[OPER], avg_times[OPER], max_times[OPER]);
            printf("axpy time min/avg/max %.3e %.3e %.3e\n", min_times[AXPY], avg_times[AXPY], max_times[AXPY]);
            printf("dot  tune min/avg/max %.3e %.3e %.3e\n", min_times[DOT ], avg_times[DOT ], max_times[DOT ]);
        }
        // No need to zero solver->part_times since they are zeroed in every "Mult" of the solver.
    }
    if (precond) {
        using MG = MultiGrid<idx_t, pc_data_t, pc_calc_t, ksp_t, 1>;
        ((MG*)precond)->ShowBreakdownReset();
    }
}

template<typename idx_t, typename ksp_t, typename pc_data_t, typename pc_calc_t>
void SStructMG_wrapper<idx_t, ksp_t, pc_data_t, pc_calc_t>::release_work_vectors()
{
    if (comm == MPI_COMM_NULL) return ;
    using MG = MultiGrid<idx_t, pc_data_t, pc_calc_t, ksp_t, 1>;
    if (precond) ((MG*)precond)->ReleaseWorkVectors();
    if (vec_x) vec_x->release_data();
    if (vec_b) vec_b->release_data();
    if (pc_buf_x) pc_buf_x->release_data();
    if (pc_buf_b) pc_buf_b->release_data();
}

template<typename idx_t, typename ksp_t, typename pc_data_t, typename pc_calc_t>
void SStructMG_wrapper<idx_t, ksp_t, pc_data_t, pc_calc_t>::destroy()
{
    if (precond) { delete precond; precond = nullptr; }
    if (solver ) { delete solver ; solver  = nullptr; }

    if (mat_A) { delete mat_A; mat_A = nullptr; }
    if (precond_mat_A) { delete precond_mat_A; precond_mat_A = nullptr; }
    if (vec_x) {
        assert(vec_b); 
        delete vec_x; vec_x = nullptr;
        delete vec_b; vec_b = nullptr;
    }
    if (pc_buf_x) {
        assert(pc_buf_b); 
        delete pc_buf_x; pc_buf_x = nullptr;
        delete pc_buf_b; pc_buf_b = nullptr;
    }

    if (ssgrid) { delete ssgrid; ssgrid = nullptr; }
    if (comm != MPI_COMM_NULL) {
        if (owns_comm) MPI_Comm_free(&comm);
        comm = MPI_COMM_NULL;
        owns_comm = false;
    }
}

template class SStructMG_wrapper<int, double, double, double>;
template class SStructMG_wrapper<int, double, float, float>;
template class SStructMG_wrapper<int, float, float, float>;
#endif  // USE_SSTRUCTMG
