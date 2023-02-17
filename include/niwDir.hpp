#pragma once

#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include "normal.hpp"


using namespace Eigen;

template<typename T>
class NIWDIR
{
    public:
        NIWDIR(const Matrix<T,Dynamic,Dynamic>& sigma, const Matrix<T,Dynamic,Dynamic>& mu, T nu, T kappa,
        boost::mt19937 &rndGen);
        ~NIWDIR();

        T logPosteriorProb(const Vector<T,Dynamic>& x_i, const Matrix<T,Dynamic, Dynamic>& x_k);
        T logProb(const Matrix<T,Dynamic,1>& x_i);
        T logProb(const Matrix<T,Dynamic,1>& x_i, const Matrix<T,Dynamic,Dynamic>& x_k);
        T prob(const Matrix<T,Dynamic,1> &x_i);

        Normal<T> samplePosteriorParameter(const Matrix<T,Dynamic, Dynamic> &x_k);
        NIWDIR<T> posterior(const Matrix<T,Dynamic, Dynamic>& x_k);
        void getSufficientStatistics(const Matrix<T,Dynamic, Dynamic>& x_k);
        Normal<T> sampleParameter();
        // Matrix<T, Dynamic, 1> karcherMean(const Matrix<T,Dynamic, Dynamic>& x_k);

    public:
        boost::mt19937 rndGen_;


        // Hyperparameters
        Matrix<T,Dynamic,Dynamic> Sigma_;
        Matrix<T,Dynamic,1> mu_;
        T nu_,kappa_;
        uint32_t dim_;


        // Sufficient statistics
        Matrix<T,Dynamic,Dynamic> scatter_;
        Matrix<T,Dynamic,1> mean_ ;
        uint16_t count_;
};

