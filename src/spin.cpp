#include "spin.h"

Spin::Spin(const Spin_control& spin_control)
         : spin_control(spin_control) {}

Spin::~Spin() {}

double Spin::generate_smearing_coef() const {
    switch (this->spin_control.spin_type)
    {
    case 0:
        return 2.0;
        break;
    default:
        assert(this->spin_control.spin_type<=0);
        return 0.0;
        break;
    }
}

uint Spin::generate_nspin() const {
    switch (this->spin_control.spin_type)
    {
    case 0:
        return 1;
        break;
    case 1:
        return 2;
        break;
    default:
        assert(this->spin_control.spin_type <= 0);
        return 0;
        break;
    }
}

uint Spin::get_spin_type() const {
    return this->spin_control.spin_type;
}

void Spin::init(const Spin&) {
    return;
}

void Spin::show() const {
    this->spin_control.show();
    return;
}
