#pragma once

#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include "normalDir.hpp"


using namespace Eigen;

template<typename T>
class NIWDIR
{
    public:
        NIWDIR(const Matrix<T,Dynamic,Dynamic>& sigma, const Matrix<T,Dynamic,Dynamic>& mu, T nu, T kappa,
        boost::mt19937 &rndGen);
        NIWDIR(const Matrix<T,Dynamic,1>& muPos, const Matrix<T,Dynamic,Dynamic>& SigmaPos, 
        const Matrix<T,Dynamic,1>& muDir, T SigmaDir, 
        T nu, T kappa, boost::mt19937 &rndGen);        
        ~NIWDIR();

        void getSufficientStatistics(const Matrix<T,Dynamic, Dynamic>& x_k);
        NIWDIR<T> posterior(const Matrix<T,Dynamic, Dynamic>& x_k);
        NormalDir<T> samplePosteriorParameter(const Matrix<T,Dynamic, Dynamic> &x_k);
        NormalDir<T> sampleParameter();

    public:
        boost::mt19937 rndGen_;


        // Hyperparameters
        Matrix<T,Dynamic,Dynamic> SigmaPos_;
        T SigmaDir_;
        Matrix<T,Dynamic,Dynamic> Sigma_;
        Matrix<T,Dynamic,1> muPos_;
        Matrix<T,Dynamic,1> muDir_;
        Matrix<T,Dynamic,1> mu_;
        T nu_,kappa_;
        uint32_t dim_;


        // Sufficient statistics
        Matrix<T,Dynamic,Dynamic> Scatter_;
        Matrix<T,Dynamic,Dynamic> ScatterPos_;
        T ScatterDir_;
        Matrix<T,Dynamic,1> meanPos_;
        Matrix<T,Dynamic,1> meanDir_;
        Matrix<T,Dynamic,1> mean_;
        uint16_t count_;
};

