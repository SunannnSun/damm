#pragma once

#include <Eigen/Dense>

using namespace Eigen;

template<typename T>
class NIW{
    public:
        Matrix<T,Dynamic,Dynamic> Delta_;
        Matrix<T,Dynamic,1> theta_;
        T nu_,kappa_;
        uint32_t D_;
        NIW();
        NIW(const Matrix<T,Dynamic,Dynamic>& Delta, const Matrix<T,Dynamic,Dynamic>& theta, T nu,  T kappa);
        ~NIW();
};

template class NIW<double>;
template class NIW<float>;