#ifndef _PREPARATION_H_
#define _PREPARATION_H_

#include <algorithm>
#include "filesys.h"
#include "control.h"

class Preparation
{
public:
    bool if_input_file2_exists = false;
    Input_file input_file;
    Input_file input_file2;
    Ion_file ion_file;
    Geometry geometry;
    Control control;
    Control control2;
    std::vector<Psp8_file> psp8_files;
    Preparation();
    ~Preparation();
    Preparation(const char* fname);
    Preparation(const Path& fname);
    void init(const Path& fname);
    void set_charge_cut(Psp8_file& psp8_file);
};



#endif //_PREPARATION_H_