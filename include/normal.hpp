#pragma once

#include <Eigen/Dense>
#include "distribution.hpp"

using namespace Eigen;


template<typename T>
class Normal : public Distribution<T>{
    public:
        T  D_;
        Matrix<T,Dynamic,1> mu_;
        Normal(const T D);
    private:
};