#pragma once

#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>

#define PI 3.141592653589793

using namespace Eigen;

template<typename T>
class gaussDamm
{
    public:
        gaussDamm(const Matrix<T,Dynamic,1> &meanPos, const Matrix<T,Dynamic, Dynamic> &covPos,
        const Matrix<T,Dynamic,1>& meanDir, T covDir, boost::mt19937 &rndGen);   
        gaussDamm(){};
        ~gaussDamm(){};
        T logProb(const Matrix<T,Dynamic,1> &x_i);


    private:
        boost::mt19937 rndGen_;

        // parameters
        Matrix<T,Dynamic,1> meanPos_;
        Matrix<T,Dynamic,1> meanDir_;
        Matrix<T,Dynamic,Dynamic> covPos_;
        T covDir_;

        Matrix<T,Dynamic,1> meanHat_;
        Matrix<T,Dynamic,Dynamic> covHat_;
        uint32_t dim_;
};
