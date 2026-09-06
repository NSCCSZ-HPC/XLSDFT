#ifndef _SPIN_H_
#define _SPIN_H_

#include <iostream>
#include "control.h"

class Spin
{
public:
    const Spin_control& spin_control;
    Spin(const Spin_control& spin_control);
    ~Spin();
    double generate_smearing_coef() const;
    uint generate_nspin() const;
    uint get_spin_type() const;
    void init(const Spin& spin);
    void show() const;
};


#endif //_SPIN_H_