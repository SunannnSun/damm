#pragma once

#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include "normal.hpp"

#define PI 3.141592653589793

using namespace Eigen;

template<typename T>
class NIW
{
    public:
        NIW(const Matrix<T,Dynamic,Dynamic>& sigma, const Matrix<T,Dynamic,Dynamic>& mu,const T nu,const T kappa, boost::mt19937* pRndGen);
        ~NIW();

        T logPosteriorProb(const Vector<T,Dynamic>& x_i, const Matrix<T,Dynamic, Dynamic>& x_k);
        T logProb(const Matrix<T,Dynamic,1>& x_i);

        Normal<T> samplePosteriorParameter(const Matrix<T,Dynamic, Dynamic>& x_k);
        NIW<T> posterior(const Matrix<T,Dynamic, Dynamic>& x_k);
        void getSufficientStatistics(const Matrix<T,Dynamic, Dynamic>& x_k);
        Normal<T> sampleParameter();

 
    public:
        boost::mt19937* pRndGen_;


        // Hyperparameters
        Matrix<T,Dynamic,Dynamic> sigma_;
        Matrix<T,Dynamic,1> mu_;
        T nu_,kappa_;
        uint32_t dim_;

        
        // Sufficient statistics
        Matrix<T,Dynamic,Dynamic> scatter_;
        Matrix<T,Dynamic,1> mean_;
        uint16_t count_;
};
