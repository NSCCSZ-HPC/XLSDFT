#include <cstring>
#include <vector>
#include <fstream>
#include "filesys.h"

Path::Path() : std::string() {}
Path::Path(const char* str) : std::string(this->simplifyPath(str)) {}
Path::Path(const std::string str) : std::string(this->simplifyPath(str)) {}
bool Path::is_absolute() const {
    if (at(0) == '/') {
        return true;
    } else {
        return false;
    }
}
Path& Path::operator/=(const Path& p1)
{
    *this = *this / p1;
    return *this;
}
Path Path::operator/(const Path& p1)
{
    if (p1.is_absolute() || this->size() == 0){
        return p1;
    } else if (back() == '/') {
        Path __result(*this + p1);
        return __result;
    } else {
        Path __result(*this + "/" + p1);
        return __result;
    }
}
/**
 * @brief simplify the Path
 *        for example "/../" to "/"
 *                    "./abc/../def" to "./def"
 *                    "./abc////def//" to "./abc/def/"
 * @param path {std::string, Path, char*} path
 * @return std::string simplified path
 * @ref  modified from LeetCode example:
 * https://leetcode.cn/problems/simplify-path/solutions/1193258/jian-hua-lu-jing-by-leetcode-solution-aucq/
 */
std::string Path::simplifyPath(std::string path) {
    auto split = [](const std::string& s, char delim) -> std::vector<std::string> {
        std::vector<std::string> ans;
        std::string cur;
        for (char ch: s) {
            if (ch == delim) {
                ans.push_back(std::move(cur));
                cur.clear();
            }
            else {
                cur += ch;
            }
        }
        ans.push_back(std::move(cur));
        return ans;
    };
    std::vector<std::string> names = split(path, '/');
    std::vector<std::string> stack;
    for (std::string& name: names) {
        if (name == "..") {
            if (!stack.empty() && stack.back() != "..") {
                stack.pop_back();
            } 
            else {
                stack.push_back(std::move(name));
            }
        }
        else if (!name.empty() && name != ".") {
            stack.push_back(std::move(name));
        }
    }
    std::string ans;
    if (stack.empty()) {
        if (path.empty() || path.at(0) != '/') {
            ans = "./";
        } else {
            ans = "/";
        }
    } else {
        std::vector<std::string>::iterator iter;
        iter = stack.begin();
        if (path.at(0) == '/') {
            ans += '/';
            while (iter != stack.end() && *iter == "..") {
                ++iter;
            }
        }
        if (iter != stack.end()) {
            ans += std::move(*iter);
            iter++;
        }
        for (; iter != stack.end(); ++iter) {
            ans += "/" + std::move(*iter);
        }
        if (path.back() == '/') {
            ans += "/";
        }
    }
    return ans;
}

Input_file::Input_file() {}

Input_file::Input_file(const char* fname) {
    this->read(fname);
}

Input_file::Input_file(const Path& fname) {
    this->read(fname);
}

std::string Input_file::get_value(const std::string& key) const {
    if(this->map.find(key) != this->map.end()) {
        return  this->map.at(key);
    }
    return "No " + key + " object";
}

void Input_file::read(const Path& fname) {
    std::ifstream input_file;
    input_file.open(fname);
    std::string line;
    if (input_file.is_open()) {
        std::string key = "NULL";
        while (input_file.good() ) {
            std::getline (input_file, line);
            line = Filesys::get_effective_string(line.substr(0, line.find('#')));
            if (line.find_first_not_of(" \t\n\v\f\r") != std::string::npos) {
                std::size_t found = line.find(':');
                if (found!=std::string::npos) {
                    key = Filesys::toupper(Filesys::get_effective_string(line.substr(0, found)));
                    this->map[key] = Filesys::get_effective_string(line.substr(found + 1, std::string::npos));
                } else {
                    this->map[key] += " " + Filesys::get_effective_string(line);
                }
            }
        }
        input_file.close();
    } else {
        assert(!"ERROR:: The .inpt file could not be opened!");
    }
    return;
}

void Input_file::show() const {
    this->print(std::cout);
    return;
}

void Input_file::show(const std::string& fname) const {
    std::ofstream outfile(fname);
    this->print(outfile);
    outfile.close();
    return;
}

void Input_file::print(std::ostream& output) const {
    if(this->map.find("CELL") != this->map.end())
        output << "CELL: " << this->get_value("CELL") << std::endl;
    if(this->map.find("LATVEC_SCALE") != this->map.end())
        output << "LATVEC_SCALE: " << this->get_value("LATVEC_SCALE") << std::endl;
    if(this->map.find("BC") != this->map.end())
        output << "BC: " << this->get_value("BC") << std::endl;
    if(this->map.find("EXCHANGE_CORRELATION") != this->map.end())
        output << "EXCHANGE_CORRELATION: " << this->get_value("EXCHANGE_CORRELATION") << std::endl;
    if(this->map.find("ELEC_TEMP_TYPE") != this->map.end())
        output << "ELEC_TEMP_TYPE: " << this->get_value("ELEC_TEMP_TYPE") << std::endl;
    if(this->map.find("SMEARING") != this->map.end())
        output << "SMEARING: " << this->get_value("SMEARING") << std::endl;
    if(this->map.find("NSTATES") != this->map.end())
        output << "NSTATES: " << this->get_value("NSTATES") << std::endl;
    if(this->map.find("MESH_SPACING") != this->map.end())
        output << "MESH_SPACING: " << this->get_value("MESH_SPACING") << std::endl;
    if(this->map.find("FD_ORDER") != this->map.end())
        output << "FD_ORDER: " << this->get_value("FD_ORDER") << std::endl;
    if(this->map.find("TOL_SCF") != this->map.end())
        output << "TOL_SCF: " << this->get_value("TOL_SCF") << std::endl;
    if(this->map.find("PRINT_ATOMS") != this->map.end())
        output << "PRINT_ATOMS: " << this->get_value("PRINT_ATOMS") << std::endl;
    if(this->map.find("PRINT_EIGEN") != this->map.end())
        output << "PRINT_EIGEN: " << this->get_value("PRINT_EIGEN") << std::endl;
    if(this->map.find("PRINT_FORCES") != this->map.end())
        output << "PRINT_FORCES: " << this->get_value("PRINT_FORCES") << std::endl;
    if(this->map.find("PRINT_DENSITY") != this->map.end())
        output << "PRINT_DENSITY: " << this->get_value("PRINT_DENSITY") << std::endl;
    output << std::endl;
    return;
}

Ion_file::Ion_file() {}
Ion_file::Ion_file(const Path& fname) {
    this->read(fname);
}
Ion_file::Ion_file(const char* fname) {
    this->read(fname);
}

void Ion_file::read(const Path& fname) {
    std::ifstream ion_file;
    ion_file.open(fname);
    std::string line;
    if (ion_file.is_open()) {
        std::string key = "NULL";
        int item = 0;
        while (ion_file.good() ) {
            std::getline (ion_file, line);
            line = Filesys::get_effective_string(line.substr(0, line.find('#')));
            if (line.find_first_not_of(" \t\n\v\f\r") != std::string::npos) {
                std::size_t found = line.find(':');
                if (found!=std::string::npos) {
                    key = Filesys::toupper(Filesys::get_effective_string(line.substr(0, found)));
                    if (key == "ATOM_TYPE") {
                        this->atom_types.push_back(Filesys::get_effective_string(line.substr(found + 1, std::string::npos)));
                        this->atom_coords.push_back(std::vector<double>());
                        this->atom_spins.push_back(std::vector<double>());
                        item = 0;
                    } else if (key == "N_TYPE_ATOM") {
                        this->atom_numbers.push_back(std::stoi(Filesys::get_effective_string(line.substr(found + 1, std::string::npos))));
                        this->atom_coords[this->atom_types.size() - 1].reserve(3 * *(this->atom_numbers.rbegin()));
                    } else if (key == "PSEUDO_POT") {
                        this->pseudo_pot_files.push_back(Filesys::get_effective_string(line.substr(found + 1, std::string::npos)));
                    } else if (key == "COORD") {
                        this->is_abs_coord.push_back(true);
                        item = 0;
                    } else if (key == "COORD_FRAC") {
                        this->is_abs_coord.push_back(false);
                        item = 0;
                    } else if (key == "SPIN") {
                        item = 1;
                    }
                } else {
                    std::size_t index1;
                    std::size_t index2;
                    if (item == 0) {
                        this->atom_coords[this->atom_types.size() - 1].push_back(std::stod(line, &index1));
                        this->atom_coords[this->atom_types.size() - 1].push_back(std::stod(line.substr(index1), &index2));
                        this->atom_coords[this->atom_types.size() - 1].push_back(std::stod(line.substr(index1 + index2), nullptr));
                    } else if (item == 1) {
                        std::istringstream iss(line);
                        std::string token;
                        while (iss >> token) {
                            this->atom_spins[this->atom_types.size() - 1].push_back(std::stod(token));
                        }
                    }
                }
            }
        }
        ion_file.close();
    } else {
        assert(!"ERROR:: The .ion file could not be opened!");
    }
    return;
}

void Ion_file::show() const {
    this->print(std::cout);
    return;
}

void Ion_file::show(const std::string& fname) const {
    std::ofstream outfile(fname);
    this->print(outfile);
    outfile.close();
    return;
}

void Ion_file::print(std::ostream& output) const {
    for (uint i = 0; i < this->atom_types.size(); i++)
    {
        output << "ATOM_TYPE: " << this->atom_types[i]
            << "               # atom type " << std::endl;
        output << "N_TYPE_ATOM: " << this->atom_numbers[i]
            << "             # number of atoms of this type" << std::endl;
        output << "PSEUDO_POT: " << this->pseudo_pot_files[i] << std::endl;
        if (this->is_abs_coord[i]) {
            output << "COORD: "
                << "                    #absolute coordinates follows " << std::endl;
        } else {
            output << "COORD_FRAC: "
                << "               #fraction coordinates follows "   << std::endl;
        }
        for (uint j = 0; j < this->atom_numbers[i] * 3; j++)
        {
            output << std::setw(22) << std::setprecision(15) << std::fixed << this->atom_coords[i][j];
            if ((int)j % 3 == 2)
                output << std::endl;
        }
        if (this->atom_spins[i].size() != 0) {
            output << "SPIN:" << std::endl;
            if (this->atom_numbers[i] == this->atom_spins[i].size()) {
                for (uint j = 0; j < this->atom_numbers[i]; j++) {
                    output << std::setw(22) << std::setprecision(15) << std::fixed
                            << this->atom_spins[i][j] << std::endl;
                }
            } else if (this->atom_numbers[i] * 3 == this->atom_spins[i].size()){
                for (uint j = 0; j < this->atom_numbers[i] * 3; j++) {
                    output << std::setw(22) << std::setprecision(15) << std::fixed << this->atom_spins[i][j];
                    if ((int)j % 3 == 2)
                        output << std::endl;
                }
            } else {
                assert(!"ERROR:: SPIN wrong in .ion file");
            }
        }
        output << std::endl;
    }
    return;
}

Atomic_orbital::Atomic_orbital() {}

void Atomic_orbital::set_label(const std::string& label) {
    if (label.empty()) {
        assert(0);
    }
    size_t i = 0;
    while (i < label.size() && std::isdigit(label[i])) {
        i++;
    }
    if (i == 0) {
        assert(0);
    }
    this->n = std::stoi(label.substr(0, i));
    if (i < label.size()) {
        char orbital_char = std::toupper(label[i]);
        switch (orbital_char) {
            case 'S': l = 0; break;
            case 'P': l = 1; break;
            case 'D': l = 2; break;
            case 'F': l = 3; break;
            default: l = -1; assert(0); break;
        }
    }
}

void Atomic_orbital::after_read(const std::vector<double>& r_grid) {
    uint n_grid = r_grid.size();
    this->chi_r.resize(n_grid);
    this->chi_rD.resize(n_grid);
    for (uint i_grid = 1; i_grid < n_grid; i_grid++) {
        this->chi_r[i_grid] = this->chi[i_grid] / r_grid[i_grid];
    }
    this->chi_r[0] = this->chi_r[1];
    Tools::getYD_gen(r_grid.data(), this->chi_r.data(), this->chi_rD.data(), n_grid);
    return;
}

void Atomic_orbital::print(std::ostream& output) const {
    output << "n = " << this->n << std::endl;
    output << "l = " << this->l << std::endl;
    int len = this->chi.size();
    output << "this->chi.size() = " << len << std::endl;
    output << "chi = [" << this->chi[0] << ", ..., " << this->chi[len - 1] << "]" << std::endl;
    output << "chi_r = [" << this->chi_r[0] << ", ..., " << this->chi_r[len - 1] << "]" << std::endl;
    output << "chi_rD = [" << this->chi_rD[0] << ", ..., " << this->chi_rD[len - 1] << "]" << std::endl;
}

Upf_file::Upf_file() {}

void Upf_file::read(const Path& fname) {
    std::ifstream upf_file;
    // std::cout << "reading upf file: " << fname << std::endl;
    upf_file.open(fname);
    if (upf_file.is_open()) {
        // std::cout << fname << " could be opened." << std::endl;
        this->n_chi = this->count_string(upf_file, "<PP_CHI");
        assert(this->n_chi >= 0);
        this->read_PP_R(upf_file);
        assert(Tools::is_uniform(this->r_grid.data(), this->n_grid));
        this->atomic_orbitals.resize(this->n_chi);
        for (int i_chi = 0; i_chi < this->n_chi; i_chi++)
        {
            this->read_PP_CHI(upf_file, i_chi);
            // this->atomic_orbitals[i_chi].print_info();
        }

        upf_file.close();

        this->after_read();
    } else {
        assert(!"ERROR:: The .UPF file could not be opened!");
    }
}

int Upf_file::count_string(std::ifstream &file, const std::string str) {
    if (!file.is_open()) {
        std::cerr << "Error: file not open\n";
        return -1;
    }

    file.clear();
    file.seekg(0);

    int count = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (line.find(str) != std::string::npos) {
            count++;
        }
    }

    file.clear();
    file.seekg(0);

    return count;
}

void Upf_file::read_PP_R(std::ifstream &file) {
    if (!file.is_open()) {
        std::cerr << "Error: file not open\n";
        return;
    }

    file.clear();
    file.seekg(0);

    std::string line;
    bool in_block = false;

    while (std::getline(file, line)) {
        if (line.find("<PP_R ") != std::string::npos) {
            in_block = true;

            // size_t pos_type = line.find("type=");
            size_t pos_size = line.find("size=");
            // size_t pos_cols = line.find("columns=");

            // if (pos_type != std::string::npos) {
            //     size_t start = line.find("\"", pos_type);
            //     size_t end = line.find("\"", start + 1);
            //     result.type = line.substr(start + 1, end - start - 1);
            // }

            if (pos_size != std::string::npos) {
                size_t start = line.find("\"", pos_size);
                size_t end = line.find("\"", start + 1);
                this->n_grid = std::stoi(line.substr(start + 1, end - start - 1));
                this->r_grid.reserve(this->n_grid);
            }

            // if (pos_cols != std::string::npos) {
            //     size_t start = line.find("\"", pos_cols);
            //     size_t end = line.find("\"", start + 1);
            //     result.columns = std::stoi(line.substr(start + 1, end - start - 1));
            // }

            continue;
        }

        if (in_block && line.find("</PP_R>") != std::string::npos) {
            break;
        }

        if (in_block) {
            std::istringstream iss(line);
            double val;
            while (iss >> val) {
                this->r_grid.emplace_back(val);
            }
        }
    }

    file.clear();
    file.seekg(0);
    return;
}

void Upf_file::read_PP_CHI(std::ifstream &file, const int& i_chi) {
    if (!file.is_open()) {
        std::cerr << "File not open.\n";
        return;
    }

    file.clear();
    file.seekg(0);

    std::string line;
    bool in_block = false;

    while (std::getline(file, line)) {

        line = Filesys::get_effective_string(line.substr(0, line.find('#')));

        if (!in_block && line.find("<PP_CHI.") == 0) {
            size_t pos_dot = line.find('.');
            int idx = std::stoi(line.substr(pos_dot + 1, std::string::npos));

            if (idx != i_chi + 1) continue;

            in_block = true;

            while (std::getline(file, line)) {
                line = Filesys::get_effective_string(line.substr(0, line.find('#')));
                if (line.find("size") == 0) {
                    size_t pos_size = line.find("size=");
                    size_t start = line.find("\"", pos_size);
                    size_t end = line.find("\"", start + 1);
                    int n_grid = std::stoi(line.substr(start + 1, end - start - 1));
                    assert(n_grid == this->n_grid);
                    this->atomic_orbitals[i_chi].chi.reserve(n_grid);
                } else if (line.find("label") == 0) {
                    size_t pos_label = line.find("label=");
                    size_t start = line.find("\"", pos_label);
                    size_t end = line.find("\"", start + 1);
                    std::string label = line.substr(start + 1, end - start - 1);
                    label = Filesys::toupper(label);
                    this->atomic_orbitals[i_chi].set_label(label);
                } else {
                    //pass
                }
                if (line.back() == '>') break;
            }


            continue;
        }

        if (in_block && line.find("</PP_CHI.") != std::string::npos) {
            break;
        }

        if (in_block) {
            std::istringstream iss(line);
            double val;
            while (iss >> val) {
                this->atomic_orbitals[i_chi].chi.emplace_back(val);
            }
        }
    }

    file.clear();
    file.seekg(0);
    return;
}

void Upf_file::after_read() {
    for (int i_chi = 0; i_chi < this->n_chi; i_chi++) {
        this->atomic_orbitals[i_chi].after_read(this->r_grid);
    }
}

void Upf_file::print(std::ostream& output) const {
    output << std::scientific << std::setw(12) << std::setprecision(5);
    output << "n_chi = " << this->n_chi << std::endl;
    output << "n_grid = " << this->n_grid << std::endl;
    output << "r_grid = [" << this->r_grid[0] 
            << ", ..., " << this->r_grid[this->n_grid - 1] 
            << "]" << std::endl;
    for (int i_chi = 0; i_chi < this->n_chi; i_chi++) {
        this->atomic_orbitals[i_chi].print(output);
    }
}

MPI_Datatype Upf_file::register_mpi_type_1st() const {
    constexpr std::size_t num_members = 2;
    int lengths[num_members] = { 1, 1};
    MPI_Aint offsets[num_members] = {offsetof(Upf_file, n_chi),
                                     offsetof(Upf_file, n_grid)};
    MPI_Datatype types[num_members] = { MPI_INT,
                                        MPI_INT};
    MPI_Datatype type;
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    return type;
}

void Upf_file::recv_preparation() {
    this->r_grid.resize(this->n_grid);
    this->atomic_orbitals.resize(this->n_chi);
    for (int i_chi = 0; i_chi < this->n_chi; i_chi++) {
        this->atomic_orbitals[i_chi].chi.resize(this->n_grid);
        this->atomic_orbitals[i_chi].chi_r.resize(this->n_grid);
        this->atomic_orbitals[i_chi].chi_rD.resize(this->n_grid);
    }
    return;
}

MPI_Datatype Upf_file::register_mpi_type_2nd() const {
    std::size_t num_members = 1 + 5 * this->n_chi;

    MPI_Aint upf_file_base;
    MPI_Get_address(this, &upf_file_base);
    std::vector<MPI_Aint> address(num_members);
    std::vector<int> lengths(0);
    std::vector<MPI_Datatype> types(0);
    lengths.reserve(num_members);
    types.reserve(num_members);
    uint i = 0;
    MPI_Get_address(this->r_grid.data(), address.data() + i++);
    lengths.emplace_back(this->n_grid);
    types.emplace_back(MPI_DOUBLE);
    for (int i_chi = 0; i_chi < this->n_chi; i_chi++) {
        MPI_Get_address(&(this->atomic_orbitals[i_chi].n), address.data() + i++);
        lengths.emplace_back(1);
        types.emplace_back(MPI_INT);
        MPI_Get_address(&(this->atomic_orbitals[i_chi].l), address.data() + i++);
        lengths.emplace_back(1);
        types.emplace_back(MPI_INT);
        MPI_Get_address(this->atomic_orbitals[i_chi].chi.data(), address.data() + i++);
        lengths.emplace_back(this->n_grid);
        types.emplace_back(MPI_DOUBLE);
        MPI_Get_address(this->atomic_orbitals[i_chi].chi_r.data(), address.data() + i++);
        lengths.emplace_back(this->n_grid);
        types.emplace_back(MPI_DOUBLE);
        MPI_Get_address(this->atomic_orbitals[i_chi].chi_rD.data(), address.data() + i++);
        lengths.emplace_back(this->n_grid);
        types.emplace_back(MPI_DOUBLE);
    }

    std::vector<MPI_Aint> offsets(num_members);
    for (uint i = 0; i < num_members; i++) {
        offsets[i] = address[i] - upf_file_base;
    }

    MPI_Datatype type;
    MPI_Type_create_struct(num_members, lengths.data(), offsets.data(), types.data(), &type);
    MPI_Type_commit(&type);
    return type;
}

void Upf_file::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Upf_file::bcast(const MPI_Comm& comm, const int& root) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Datatype upf_file_type_1st = this->register_mpi_type_1st();
    MPI_Bcast(this, 1, upf_file_type_1st, root, comm);
    this->deregister_mpi_type(upf_file_type_1st);
    if (rank != root) {
        this->recv_preparation();
    }
    MPI_Datatype upf_file_type_2nd = this->register_mpi_type_2nd();
    MPI_Bcast(this, 1, upf_file_type_2nd, root, comm);
    this->deregister_mpi_type(upf_file_type_2nd);
    return;
}

Psp8_file::Psp8_file() : ekb(5) , rgrid(5), kbk_projector(5) {}

Psp8_file::Psp8_file(const Path& fname) : ekb(5) , rgrid(5), kbk_projector(5) {
    this->read(fname);
}

Psp8_file::Psp8_file(const char* fname) : ekb(5) , rgrid(5), kbk_projector(5) {
    this->read(fname);
}

void Psp8_file::read(const Path& fname) {
    std::ifstream psp8_file;
    psp8_file.open(fname);
    if (psp8_file.is_open()) {
        // first 6 lines
        this->read_basic(psp8_file);
        // data block proj and pot
        for (int l = 0; l < 5; l++)
        {   
            int temp;
            if (l != this->lloc && this->nproj[l]) {
                psp8_file >> temp;
                this->read_nonlocal(psp8_file, l);
            } else if (l == this->lloc) {
                psp8_file >> temp;
                this->read_local(psp8_file);
            }
        }

        if (this->lloc >= 5) {
            int temp;
            psp8_file >> temp;
            this->read_local(psp8_file);
        }

        // data block for model core charge for NLCC
        if (this->fchrg > 1e-12) {
            this->rgrid_charge = std::vector<double>(this->mmax);
            this->r_charge = std::vector<std::vector<double>>(5, std::vector<double>(this->mmax));
            this->read_charge(psp8_file);
        }

        // data block for isolated atom electron density
        this->rgrid_density = std::vector<double>(this->mmax);
        this->r_density = std::vector<std::vector<double>>(3, std::vector<double>(this->mmax));
        this->read_density(psp8_file);

        psp8_file.close();

        this->after_read();

        // for PDOS calculation
        std::string suffix = "psp8";
        if (fname.size() >= suffix.size() &&
            fname.compare(fname.size() - suffix.size(), suffix.size(), suffix) == 0) {
            std::string upf_file_name = fname;
            upf_file_name.replace(upf_file_name.size() - suffix.size(), suffix.size(), "upf");
            std::ifstream f(upf_file_name);
            if (f.good()) {
                this->upf_file.read(upf_file_name);
                this->if_has_upf_file = true;
            } else {
                upf_file_name = fname;
                upf_file_name.replace(upf_file_name.size() - suffix.size(), suffix.size(), "UPF");
                std::ifstream f2(upf_file_name);
                if (f2.good()) {
                    this->upf_file.read(upf_file_name);
                    this->if_has_upf_file = true;
                }
            }
        } else {
            std::cout << fname << "is not ended with psp8, can not read upf.file" << std::endl;
        }

    } else {
        assert(!"ERROR:: The .psp8 file could not be opened!");
    }
    return;
}

void Psp8_file::read_basic(std::ifstream& input_file) {
    std::string line;
    // line 1
    getline(input_file, this->title);

    // line 2
    input_file >> this->zatom >> this->zion >> this->pspd;
    std::getline(input_file, line);

    // line 3
    input_file >> this->pspcod >> this->pspxc >> this->lmax >> this->lloc >> this->mmax >> this->r2well;
    std::getline(input_file, line);
    
    //line1
    this->r_core.resize(this->lmax + 1);
    std::stringstream ss(this->title);
    std::string temp;
    std::string bench("r_core=");
    while (true) {
        ss >> temp;
        if (temp.compare(bench) == 0) break;
    }
    for (int l = 0; l <= this->lmax; l++) {
        ss >> this->r_core[l];
    }

    // line 4
    input_file >> this->rchrg >> this->fchrg >> this->qchrg;
    std::getline(input_file, line);

    // line 5
    for (uint l = 0; l < 5; l++)
    {
        input_file >> this->nproj[l];
        this->ekb[l] = std::vector<double>(this->nproj[l]);
        if(this->nproj[l]) {
            this->rgrid[l] = std::vector<double>(this->mmax);
            this->kbk_projector[l] = std::vector<std::vector<double>>(this->nproj[l], std::vector<double>(this->mmax));
        }
    }
    this->local_potential.resize(this->mmax);
    this->rgrid_local_potential.resize(this->mmax);
    std::getline(input_file, line);

    // line 6
    input_file >> this->extension_switch;
    std::getline(input_file, line);
    return;
}

void Psp8_file::read_nonlocal(std::ifstream& input_file, const int& l) {
    for (uint il = 0; il < this->nproj[l]; il++)
    {
        input_file >> this->ekb[l][il];
    }
    int temp;
    for (int immax = 0; immax < this->mmax; ++immax) {
        input_file >> temp >> this->rgrid[l][immax];
        for (uint il = 0; il < this->nproj[l]; il++)
        {
            input_file >> this->kbk_projector[l][il][immax];
        }
    }
    return;
}

void Psp8_file::read_local(std::ifstream& input_file) {
    int temp;
    for (int immax = 0; immax < this->mmax; ++immax) {
        input_file >> temp >> this->rgrid_local_potential[immax] >> this->local_potential[immax];
    }
    return;
}

void Psp8_file::read_charge(std::ifstream& input_file) {
    int temp;
    for (int immax = 0; immax < this->mmax; ++immax) {
        input_file >> temp
            >> this->rgrid_charge[immax]
            >> this->r_charge[0][immax]
            >> this->r_charge[1][immax]
            >> this->r_charge[2][immax]
            >> this->r_charge[3][immax]
            >> this->r_charge[4][immax];
    }
    return;
}

void Psp8_file::read_density(std::ifstream& input_file) {
    int temp;
    for (int immax = 0; immax < this->mmax; ++immax) {
        input_file >> temp
            >> this->rgrid_density[immax]
            >> this->r_density[0][immax]
            >> this->r_density[1][immax]
            >> this->r_density[2][immax];
    }
    return;
}

void Psp8_file::after_read() {
    this->local_potential_R.resize(this->mmax);
    this->local_potential_RD.resize(this->mmax);
    this->r_density_4pi.resize(this->mmax);
    this->r_density_4piD.resize(this->mmax);
    for (int i = 0; i < this->mmax; i++) {
        this->local_potential_R[i] =  this->local_potential[i] * this->rgrid_local_potential[i];
        // this->r_density_4pi[i] = this->r_density[0][i] * (0.25 * M_1_PI);
        this->r_density_4pi[i] = this->r_density[0][i] / (4.0 * M_PI);  // be same with sparc
    }
    Tools::getYD_gen(this->rgrid_local_potential.data(), this->local_potential_R.data(), this->local_potential_RD.data(), this->mmax);
    Tools::getYD_gen(this->rgrid_density.data(), this->r_density_4pi.data(), this->r_density_4piD.data(), this->mmax);

    this->kbk_projector_R.resize(5);
    this->kbk_projector_RD.resize(5);
    for (uint l = 0; l < 5; l++) {
        if(this->nproj[l]) {
            this->kbk_projector_R[l].resize(this->nproj[l]);
            this->kbk_projector_RD[l].resize(this->nproj[l]);
            double const* const rgrid_ptr = this->rgrid[l].data();
            for (uint i = 0; i < this->nproj[l]; i++) {
                this->kbk_projector_R[l][i].resize(this->mmax);
                this->kbk_projector_RD[l][i].resize(this->mmax);
                double* const kbk_projector_R_ptr = this->kbk_projector_R[l][i].data();
                double* const kbk_projector_RD_ptr = this->kbk_projector_RD[l][i].data();
                double const* const kbk_projector_ptr = this->kbk_projector[l][i].data();
                for (int j = 1; j < this->mmax; j++) {
                    kbk_projector_R_ptr[j] = kbk_projector_ptr[j]/rgrid_ptr[j];
                }
                kbk_projector_R_ptr[0] = kbk_projector_R_ptr[1];
                Tools::getYD_gen(rgrid_ptr, kbk_projector_R_ptr, kbk_projector_RD_ptr, this->mmax);
            }
        }
    }

    for (uint l = 0; l < 5; l++) {
        if(this->nproj[l]) this->is_rgrid_uniform[l] = Tools::is_uniform(this->rgrid[l].data(), this->mmax);
    }
    is_rgrid_local_potential_uniform = Tools::is_uniform(this->rgrid_local_potential.data(), this->mmax);
    if (this->fchrg > 1e-12) {
        this->is_rgrid_charge_uniform = Tools::is_uniform(this->rgrid_charge.data(), this->mmax);
        this->rho_c_table.resize(this->mmax);
        this->rho_c_tableD.resize(this->mmax);
        for (int i = 0; i < this->mmax; i++) {
            this->rho_c_table[i] =  this->r_charge[0][i] * (0.25 * M_1_PI);
        }
        Tools::getYD_gen(this->rgrid_charge.data(), this->rho_c_table.data(), this->rho_c_tableD.data(), this->mmax);
    }
    is_rgrid_density_uniform = Tools::is_uniform(this->rgrid_density.data(), this->mmax);

    // modify c_core to make sure c_core is big enough
    // this could change the original data in .psp8 file
    for (int l = 0; l <= this->lmax; l++) {
        double r_core_read = this->r_core[l];
        double* const& rgrid = this->rgrid[l].data();
        for (uint i = 0; i < this->nproj[l]; i++) {
            double* const& kbk_projector_R = this->kbk_projector_R[l][i].data();
            for (int k = 0; k < this->mmax; k++) {
                if (rgrid[k] < r_core_read) continue;
                if (rgrid[k] > this->r_core[l] && kbk_projector_R[k] > 1E-8) {
                    this->r_core[l] = rgrid[k];
                }
            }
        }
    }

    return;
}

void Psp8_file::show() const {
    // std::cout << "These psp8 information is from: "<< *this << std::endl;
    this->print(std::cout);
    return;
}

void Psp8_file::show(const std::string& fname) const {
    std::ofstream outfile(fname);
    this->print(outfile);
    outfile.close();
    return;
}

void Psp8_file::print(std::ostream& output) const {
    // first 6 lines
    this->print_basic(output);
    assert(this->extension_switch == 1 && "ERROR: SPIN ORBIT COUPLING IS NOT SUPPORTED!");

    // data block
    for (int l = 0; l < 5; l++)
    {
        if (l != this->lloc && this->nproj[l] != 0) {
            output << std::setw(4) <<l;
            this->print_nonlocal(output, l);
        } else if (l == this->lloc) {
            output << std::setw(4) <<l;
            this->print_local(output);
        }
    }

    // data block for model core charge for NLCC
    if (this->fchrg > 1e-12) {
        this->print_charge(output);
    }

    // data block for isolated atom electron density
    this->print_density(output);

    if (this->if_has_upf_file) {
        this->upf_file.print();
    }
    return;
}

void Psp8_file::print_basic(std::ostream& output) const {
    // line 1
    output << this->title << std::endl;
    output << "   r_core=";
    for (int l = 0; l <= this->lmax; l++) {
        output << "  " << this->r_core[l];
    }
    output << std::endl;

    //line 2
    output << std::setw(14) << std::setprecision(8) << std::right << std::fixed << this->zatom
        << std::setw(14) << std::setprecision(8) << std::right << std::fixed << this->zion
        << std::setw(14) << std::setprecision(8) << std::right << std::fixed << this->pspd
        << "    zatom, zion, pspd" << std::endl;

    //line 3
    output << std::setw(6) << std::setprecision(8) << std::right << std::fixed << this->pspcod
        << std::setw(6) << std::setprecision(8) << std::right << std::fixed << this->pspxc
        << std::setw(6) << std::setprecision(8) << std::right << std::fixed << this->lmax
        << std::setw(6) << std::setprecision(8) << std::right << std::fixed << this->lloc
        << std::setw(6) << std::setprecision(8) << std::right << std::fixed << this->mmax
        << std::setw(6) << std::setprecision(8) << std::right << std::fixed << this->r2well
        << "    pspcod, pspxc, lmax, lloc, mmax, r2well" << std::endl;

    // line 4
    output << std::setw(14) << std::setprecision(8) << std::right << std::fixed << this->rchrg
        << std::setw(14) << std::setprecision(8) << std::right << std::fixed << this->fchrg
        << std::setw(14) << std::setprecision(8) << std::right << std::fixed << this->qchrg
        << "    rchrg fchrg qchrg" << std::endl;
    
    // line 5
    for (int l = 0; l < 5; l++)
    {
        output << std::setw(6) << std::setprecision(8) << std::right << std::fixed << this->nproj[l];
    }
    output << "    nproj" << std::endl;

    // line 6
    output << std::setw(14) << std::setprecision(8) << std::right << std::fixed << this->extension_switch
        << "    extension_switch" << std::endl;
    return;
}

void Psp8_file::print_nonlocal(std::ostream& output, const int& l) const {
    output << std::setw(27) << "";
    for (uint il = 0; il < this->nproj[l]; il++)
    {
        output << std::setw(25) << std::setprecision(13) << std::right << std::scientific << std::uppercase << this->ekb[l][il];
    }
    output<<std::endl;

    for (int immax = 0; immax < this->mmax; ++immax) {
        output << std::setw(6) << immax+1 
            << std::setw(25) << std::setprecision(13) << std::right << std::scientific << std::uppercase << this->rgrid[l][immax];
        for (uint il = 0; il < this->nproj[l]; il++)
        {
            output << std::setw(25) << std::setprecision(13) << std::right << std::scientific << std::uppercase << this->kbk_projector[l][il][immax];
        }
        output << std::endl;
    }
    return;
}

void Psp8_file::print_local(std::ostream& output) const {
    output<<std::endl;
    for (int immax = 0; immax < this->mmax; ++immax) {
        output << std::setw(6) << immax+1
            << std::setw(25) << std::setprecision(13) << std::right << std::scientific << std::uppercase << this->rgrid_local_potential[immax]
            << std::setw(25) << std::setprecision(13) << std::right << std::scientific << std::uppercase << this->local_potential[immax]
            << std::endl;
    }
    return;
}

void Psp8_file::print_charge(std::ostream& output) const {
    for (int immax = 0; immax < this->mmax; ++immax) {
        output << std::setw(6) << immax+1
            << std::setw(25) << std::setprecision(13) << std::right << std::scientific << std::uppercase << this->rgrid_charge[immax]
            << std::setw(25) << std::setprecision(13) << std::right << std::scientific << std::uppercase << this->r_charge[0][immax]
            << std::setw(25) << std::setprecision(13) << std::right << std::scientific << std::uppercase << this->r_charge[1][immax]
            << std::setw(25) << std::setprecision(13) << std::right << std::scientific << std::uppercase << this->r_charge[2][immax]
            << std::setw(25) << std::setprecision(13) << std::right << std::scientific << std::uppercase << this->r_charge[3][immax]
            << std::setw(25) << std::setprecision(13) << std::right << std::scientific << std::uppercase << this->r_charge[4][immax]
            << std::endl;
    }
    return;
}

void Psp8_file::print_density(std::ostream& output) const {
    for (int immax = 0; immax < this->mmax; ++immax) {
        output << std::setw(6) << immax+1
            << std::setw(25) << std::setprecision(13) << std::right << std::scientific << std::uppercase << this->rgrid_density[immax]
            << std::setw(25) << std::setprecision(13) << std::right << std::scientific << std::uppercase << this->r_density[0][immax]
            << std::setw(25) << std::setprecision(13) << std::right << std::scientific << std::uppercase << this->r_density[1][immax]
            << std::setw(25) << std::setprecision(13) << std::right << std::scientific << std::uppercase << this->r_density[2][immax]
            << std::endl;
    }
    return;
}

MPI_Datatype Psp8_file::register_mpi_type_1st() const {
    constexpr std::size_t num_members = 13;
    int lengths[num_members] = { 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 1, 1, 1};
    // MPI_Aint offsets[num_members] = {offsetof(Psp8_file, zion),
    //                                  offsetof(Psp8_file, lmax),
    //                                  offsetof(Psp8_file, lloc),
    //                                  offsetof(Psp8_file, mmax),
    //                                  offsetof(Psp8_file, rchrg),
    //                                  offsetof(Psp8_file, fchrg),
    //                                  offsetof(Psp8_file, nproj),
    //                                  offsetof(Psp8_file, is_rgrid_uniform),
    //                                  offsetof(Psp8_file, is_rgrid_local_potential_uniform),
    //                                  offsetof(Psp8_file, is_rgrid_charge_uniform),
    //                                  offsetof(Psp8_file, is_rgrid_density_uniform)};
    MPI_Aint psp8_file_base;
    MPI_Get_address(this, &psp8_file_base);
    MPI_Aint offsets[num_members];
    int count = 0;
    MPI_Aint address;
    MPI_Get_address(&(this->zatom), &address);
    offsets[count++] = address - psp8_file_base;
    MPI_Get_address(&(this->zion), &address);
    offsets[count++] = address - psp8_file_base;
    MPI_Get_address(&(this->lmax), &address);
    offsets[count++] = address - psp8_file_base;
    MPI_Get_address(&(this->lloc), &address);
    offsets[count++] = address - psp8_file_base;
    MPI_Get_address(&(this->mmax), &address);
    offsets[count++] = address - psp8_file_base;
    MPI_Get_address(&(this->rchrg), &address);
    offsets[count++] = address - psp8_file_base;
    MPI_Get_address(&(this->fchrg), &address);
    offsets[count++] = address - psp8_file_base;
    MPI_Get_address(&(this->nproj), &address);
    offsets[count++] = address - psp8_file_base;
    MPI_Get_address(&(this->is_rgrid_uniform), &address);
    offsets[count++] = address - psp8_file_base;
    MPI_Get_address(&(this->is_rgrid_local_potential_uniform), &address);
    offsets[count++] = address - psp8_file_base;
    MPI_Get_address(&(this->is_rgrid_charge_uniform), &address);
    offsets[count++] = address - psp8_file_base;
    MPI_Get_address(&(this->is_rgrid_density_uniform), &address);
    offsets[count++] = address - psp8_file_base;
    MPI_Get_address(&(this->if_has_upf_file), &address);
    offsets[count++] = address - psp8_file_base;
    assert(count == num_members);
    MPI_Datatype types[num_members] = {MPI_DOUBLE,
                                       MPI_DOUBLE,
                                       MPI_INT,
                                       MPI_INT,
                                       MPI_INT,
                                       MPI_DOUBLE,
                                       MPI_DOUBLE,
                                       MPI_UNSIGNED,
                                       MPI_C_BOOL, 
                                       MPI_C_BOOL,
                                       MPI_C_BOOL,
                                       MPI_C_BOOL,
                                       MPI_C_BOOL};
    MPI_Datatype type;
    MPI_Type_create_struct(num_members, lengths, offsets, types, &type);
    MPI_Type_commit(&type);
    return type;
}

void Psp8_file::recv_preparation() {
    this->r_core.resize(this->lmax + 1);

    this->rgrid_local_potential.resize(this->mmax);
    this->local_potential.resize(1);
    this->local_potential_R.resize(this->mmax);
    this->local_potential_RD.resize(this->mmax);

    this->rgrid_density.resize(this->mmax);
    this->r_density_4pi.resize(this->mmax);
    this->r_density_4piD.resize(this->mmax);

    this->kbk_projector_R.resize(5);
    this->kbk_projector_RD.resize(5);
    for (uint l = 0; l < 5; l++) {
        if(this->nproj[l]) {
            this->ekb[l].resize(this->nproj[l]);
            this->rgrid[l].resize(this->mmax);
            this->kbk_projector_R[l].resize(this->nproj[l]);
            this->kbk_projector_RD[l].resize(this->nproj[l]);
            for (uint i = 0; i < this->nproj[l]; i++) {
                this->kbk_projector_R[l][i].resize(this->mmax);
                this->kbk_projector_RD[l][i].resize(this->mmax);
            }
        }
    }
    if (this->fchrg > 1e-12) {
        this->rgrid_charge.resize(this->mmax);
        this->rho_c_table.resize(this->mmax);
        this->rho_c_tableD.resize(this->mmax);
    }
    return;
}

int Psp8_file::generate_ncol() const {
    int count = 0;
    for (uint l = 0; l < 5; l++) {
        for (uint i = 0; i < this->nproj[l]; i++) {
            count += 2 * l + 1;
        }
    }
    return count;
}

MPI_Datatype Psp8_file::register_mpi_type_2nd() const {
    uint nproj_sum = 0;
    uint l_count = 0;
    for (uint l = 0; l < 5; ++l) {
        if (this->nproj[l]) {
            nproj_sum += this->nproj[l];
            l_count += 1;
        }
    }
    std::size_t num_members = 8 + 3 + nproj_sum * 2 + l_count * 2;

    MPI_Aint psp8_file_base;
    MPI_Get_address(this, &psp8_file_base);
    // MPI_Aint address[num_members];
    std::vector<MPI_Aint> address(num_members);
    std::vector<int> lengths(0);
    lengths.reserve(num_members);
    uint i = 0;
    MPI_Get_address(this->r_core.data(), address.data() + i++);
    lengths.emplace_back(this->lmax + 1);
    MPI_Get_address(this->rgrid_local_potential.data(), address.data() + i++);
    lengths.emplace_back(this->mmax);
    MPI_Get_address(this->local_potential.data(), address.data() + i++);
    lengths.emplace_back(1);
    MPI_Get_address(this->local_potential_R.data(), address.data() + i++);
    lengths.emplace_back(this->mmax);
    MPI_Get_address(this->local_potential_RD.data(), address.data() + i++);
    lengths.emplace_back(this->mmax);
    MPI_Get_address(this->rgrid_density.data(), address.data() + i++);
    lengths.emplace_back(this->mmax);
    MPI_Get_address(this->r_density_4pi.data(), address.data() + i++);
    lengths.emplace_back(this->mmax);
    MPI_Get_address(this->r_density_4piD.data(), address.data() + i++);
    lengths.emplace_back(this->mmax);
    int rgrid_charge_length = this->fchrg > 1e-12 ? this->mmax : 0;
    MPI_Get_address(this->rgrid_charge.data(), address.data() + i++);
    lengths.emplace_back(rgrid_charge_length);
    MPI_Get_address(this->rho_c_table.data(), address.data() + i++);
    lengths.emplace_back(rgrid_charge_length);
    MPI_Get_address(this->rho_c_tableD.data(), address.data() + i++);
    lengths.emplace_back(rgrid_charge_length);
    for (uint l = 0; l < 5; l++) {
        if (this->nproj[l]) {
            MPI_Get_address(this->ekb[l].data(), address.data() + i++);
            lengths.emplace_back(this->nproj[l]);
            MPI_Get_address(this->rgrid[l].data(), address.data() + i++);
            lengths.emplace_back(this->mmax);
            for (uint j = 0; j < this->nproj[l]; j++) {
                MPI_Get_address(this->kbk_projector_R[l][j].data(), address.data() + i++);
                lengths.emplace_back(this->mmax);
                MPI_Get_address(this->kbk_projector_RD[l][j].data(), address.data() + i++);
                lengths.emplace_back(this->mmax);
            }
        }
    }

    // MPI_Aint offsets[num_members];
    std::vector<MPI_Aint> offsets(num_members);
    // MPI_Datatype types[num_members];
    std::vector<MPI_Datatype> types(num_members, MPI_DOUBLE);
    // int lengths[num_members];
    // std::vector<int> lengths(num_members);
    for (uint i = 0; i < num_members; i++) {
        offsets[i] = address[i] - psp8_file_base;
        // types[i] = MPI_DOUBLE;
        // lengths[i] = this->mmax;
    }
    // lengths[0] = this->lmax + 1;
    // lengths[2] = 1;

    MPI_Datatype type;
    MPI_Type_create_struct(num_members, lengths.data(), offsets.data(), types.data(), &type);
    MPI_Type_commit(&type);
    return type;
}

void Psp8_file::deregister_mpi_type(MPI_Datatype type) const {
    MPI_Type_free(&type);
    return;
}

void Psp8_file::bcast(const MPI_Comm& comm, const int& root) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Datatype Psp8_file_type_1st = this->register_mpi_type_1st();
    MPI_Bcast(this, 1, Psp8_file_type_1st, root, comm);
    this->deregister_mpi_type(Psp8_file_type_1st);
    if (rank != root) {
        this->recv_preparation();
    }
    MPI_Datatype Psp8_file_type_2nd = this->register_mpi_type_2nd();
    MPI_Bcast(this, 1, Psp8_file_type_2nd, root, comm);
    this->deregister_mpi_type(Psp8_file_type_2nd);

    // for PDOS calculation
    if (this->if_has_upf_file) {
        this->upf_file.bcast(comm, root);
    }
    return;
}

#ifdef BCAST_CHECK
void Psp8_file::bcast_check(const MPI_Comm& comm) const {
    int rank;
    MPI_Comm_rank(comm, &rank);
    std::this_thread::sleep_for(std::chrono::milliseconds(rank * 2000));
    this->bcast_print();
    return;
}

void Psp8_file::bcast_print() const {
    std::cout << "r_core = [";
    for (int l = 0; l <= this->lmax; l++) {
        std::cout << this->r_core[l] << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "zion = "
              << std::setw(14) << std::setprecision(8) << std::right << std::fixed << this->zion
              << std::endl;
    std::cout << "lmax = "
              << std::setw(6) << std::setprecision(8) << std::right << std::fixed << this->lmax
              << std::endl;
    std::cout << "lloc = "
              << std::setw(6) << std::setprecision(8) << std::right << std::fixed << this->lloc
              << std::endl;
    std::cout << "mmax = "
              << std::setw(6) << std::setprecision(8) << std::right << std::fixed << this->mmax
              << std::endl;
    std::cout << "rchrg = "
              << std::setw(14) << std::setprecision(8) << std::right << std::fixed << this->rchrg
              << std::endl;
    std::cout << "fchrg = "
              << std::setw(14) << std::setprecision(8) << std::right << std::fixed << this->fchrg
              << std::endl;
    std::cout << "nproj = [";
    for (int l = 0; l < 5; l++)
    {
        std::cout << std::setw(6) << std::setprecision(8) << std::right << std::fixed << this->nproj[l];
    }
    std::cout << " ]" << std::endl;
    std::cout << "is_rgrid_uniform = [";
    for (int l = 0; l < 5; l++)
    {
        std::cout << std::setw(6) << std::setprecision(8) << std::right << std::fixed << this->is_rgrid_uniform[l];
    }
    std::cout << " ]" << std::endl;
    std::cout << "is_rgrid_local_potential_uniform = "
              << std::setw(6) << std::setprecision(8) << std::right << std::fixed << this->is_rgrid_local_potential_uniform
              << std::endl;
    std::cout << "is_rgrid_charge_uniform = "
              << std::setw(6) << std::setprecision(8) << std::right << std::fixed << this->is_rgrid_charge_uniform
              << std::endl;
    std::cout << "is_rgrid_density_uniform = "
              << std::setw(6) << std::setprecision(8) << std::right << std::fixed << this->is_rgrid_density_uniform
              << std::endl;
    
    std::cout << "sum(rgrid_local_potential) = "
              << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_sum(this->rgrid_local_potential.data(), (uint)this->mmax)
              << std::endl;
    std::cout << "square_sum(rgrid_local_potential) = "
              << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_norm_square_sum(this->rgrid_local_potential.data(), (uint)this->mmax)
              << std::endl;
    std::cout << "sum(local_potential) = "
              << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_sum(this->local_potential.data(), 1)
              << std::endl;
    std::cout << "square_sum(local_potential) = "
              << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_norm_square_sum(this->local_potential.data(), 1)
              << std::endl;
    std::cout << "sum(local_potential_R) = "
              << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_sum(this->local_potential_R.data(), (uint)this->mmax)
              << std::endl;
    std::cout << "square_sum(local_potential_R) = "
              << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_norm_square_sum(this->local_potential_R.data(), (uint)this->mmax)
              << std::endl;
    std::cout << "sum(local_potential_RD) = "
              << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_sum(this->local_potential_RD.data(), (uint)this->mmax)
              << std::endl;
    std::cout << "square_sum(local_potential_RD) = "
              << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_norm_square_sum(this->local_potential_RD.data(), (uint)this->mmax)
              << std::endl;
    std::cout << "sum(rgrid_density) = "
              << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_sum(this->rgrid_density.data(), (uint)this->mmax)
              << std::endl;
    std::cout << "square_sum(rgrid_density) = "
              << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_norm_square_sum(this->rgrid_density.data(), (uint)this->mmax)
              << std::endl;
    std::cout << "sum(r_density_4pi) = "
              << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_sum(this->r_density_4pi.data(), (uint)this->mmax)
              << std::endl;
    std::cout << "square_sum(r_density_4pi) = "
              << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_norm_square_sum(this->r_density_4pi.data(), (uint)this->mmax)
              << std::endl;
    std::cout << "sum(r_density_4piD) = "
              << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_sum(this->r_density_4piD.data(), (uint)this->mmax)
              << std::endl;
    std::cout << "square_sum(r_density_4piD) = "
              << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_norm_square_sum(this->r_density_4piD.data(), (uint)this->mmax)
              << std::endl;
    for (uint l = 0; l < 5; l++) {
        if (this->nproj[l]) {
            std::cout << "sum(rgrid[" << l << "]) = "
                      << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_sum(this->rgrid[l].data(), (uint)this->mmax)
                      << std::endl;
            std::cout << "square_sum(rgrid[" << l << "]) = "
                      << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_norm_square_sum(this->rgrid[l].data(), (uint)this->mmax)
                      << std::endl;
            for (uint i = 0; i < this->nproj[l]; i++) {
                std::cout << "this->ekb[" << l << "][" << i << "] = "<< this->ekb[l][i] << std::endl;
                std::cout << "sum(kbk_projector_R[" << l << "][" << i << "]) = "
                        << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_sum(this->kbk_projector_R[l][i].data(), (uint)this->mmax)
                        << std::endl;
                std::cout << "square_sum(kbk_projector_R[" << l << "][" << i << "]) = "
                        << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_norm_square_sum(this->kbk_projector_R[l][i].data(), (uint)this->mmax)
                        << std::endl;
                std::cout << "sum(kbk_projector_RD[" << l << "][" << i << "]) = "
                        << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_sum(this->kbk_projector_RD[l][i].data(), (uint)this->mmax)
                        << std::endl;
                std::cout << "square_sum(kbk_projector_RD[" << l << "][" << i << "]) = "
                        << std::setw(14) << std::setprecision(8) << std::right << std::fixed << Linalg::vector_norm_square_sum(this->kbk_projector_RD[l][i].data(), (uint)this->mmax)
                        << std::endl;
            }
        }
    }

    // for PDOS calculation
    if (this->if_has_upf_file) {
        this->upf_file.print();
    }

    return;
}
#endif

std::string Filesys::get_effective_string(const std::string& str_in, const char& C) {
    std::string temp = str_in.substr(0, str_in.find(C));
    if (temp.find_first_not_of(" \t\n\v\f\r") != std::string::npos) {
        return temp.substr(temp.find_first_not_of(" \t\n\v\f\r"),
                           temp.find_last_not_of(" \t\n\v\f\r") + 1);
    } else {
        return "";
    }
}

std::string Filesys::get_effective_string(const std::string& str_in) {
    return Filesys::get_effective_string(str_in, '#');
}

std::string Filesys::toupper(const std::string& str_in) {
    std::string temp = str_in;
    // for (char & c: temp) c = toupper(c);
    for (uint i = 0; i < temp.length(); ++i) {
        temp[i] = std::toupper(temp[i]);
    }
    return temp;
}
