#ifndef _ATOMIC_SOLVER_
#define _ATOMIC_SOLVER_

#include "filesys.h"
#include "spin.h"

template<typename T>
class Atomic_solver
{
public:
    const Psp8_file& psp8_file;
    // const Spin& spin;
    Atomic_solver(const Psp8_file& psp8_file);
    ~Atomic_solver();
    void print_atom_info(std::ostream& output = std::cout);
};


#endif //_ATOMIC_SOLVER_
