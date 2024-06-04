#pragma once

#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>

#define PI 3.141592653589793

using namespace Eigen;

template<typename T>
class Gauss
{
    public:
        Gauss(const Matrix<T,Dynamic,1> &mu, const Matrix<T,Dynamic,Dynamic> &sigma, boost::mt19937 &rndGen);
        Gauss(){};
        ~Gauss(){};


        T logProb(const Matrix<T,Dynamic,1> &x_i);
        T prob(const Matrix<T,Dynamic,1> &x_i);

    private:
        boost::mt19937 rndGen_;

        // parameters
        uint32_t dim_;
        Matrix<T,Dynamic,1> mu_;
        Matrix<T,Dynamic,Dynamic> sigma_;
};
