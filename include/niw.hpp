#pragma once

#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include "distribution.hpp"


using namespace Eigen;

template<typename T>
class NIW: public Distribution<T>
{
    public:
        Matrix<T,Dynamic,Dynamic> sigma_0;
        Matrix<T,Dynamic,1> mu_0;
        T nu_0,kappa_0;
        uint32_t dim;


        // NIW();
        NIW(const Matrix<T,Dynamic,Dynamic>& sigma, const Matrix<T,Dynamic,Dynamic>& mu, T nu, T kappa, 
            boost::mt19937 *pRndGen);
        ~NIW();
};

template class NIW<double>;
template class NIW<float>;