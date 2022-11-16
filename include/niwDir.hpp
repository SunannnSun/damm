#pragma once

#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include "distribution.hpp"
#include "global.hpp"


using namespace Eigen;

template<typename T>
class NIWDIR: public Distribution<T>
{
    public:
        Matrix<T,Dynamic,Dynamic> sigma_;
        Matrix<T,Dynamic,1> mu_;
        T nu_,kappa_;
        uint32_t dim_;


        // NIW();
        NIWDIR(const Matrix<T,Dynamic,Dynamic>& sigma, const Matrix<T,Dynamic,Dynamic>& mu, T nu, T kappa, 
            boost::mt19937 *pRndGen);
        ~NIWDIR();

    //     T logPosteriorProb(const Vector<T,Dynamic>& x_i, const Matrix<T,Dynamic, Dynamic>& x_k);
        T logProb(const Matrix<T,Dynamic,1>& x_i);
    //     NIW<T> posterior(const Matrix<T,Dynamic, Dynamic>& x_k);
        void getSufficientStatistics(const Matrix<T,Dynamic, Dynamic>& x_k);
        Matrix<T, Dynamic, 1> karcherMean(const Matrix<T,Dynamic, Dynamic>& x_k);

    public:
        Matrix<T,Dynamic,Dynamic> scatter_;
        Matrix<T,Dynamic,1> mean_;
        uint16_t count_;
};

template class NIWDIR<double>;
template class NIWDIR<float>;