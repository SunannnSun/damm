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
        NIW(const MatrixXd &Sigma, const VectorXd &mu, T nu, T kappa, boost::mt19937 &rndGen);
        ~NIW();

        // NIWDIR<T> getNIWDIR();

        T logPostPredProb(const Matrix<T,Dynamic,1> &x_i);
        T postPredProb(const Matrix<T,Dynamic,1> &x_i);

        Normal<T> samplePosteriorParameter(const Matrix<T,Dynamic, Dynamic> &x_k);
        NIW<T> posterior(const Matrix<T,Dynamic, Dynamic> &x_k);
        void getSufficientStatistics(const Matrix<T,Dynamic, Dynamic> &x_k);
        Normal<T> sampleParameter();

 
    public:
        boost::mt19937 rndGen_;


        // Hyperparameters
        Matrix<T,Dynamic,Dynamic> Sigma_;
        Matrix<T,Dynamic,1> mu_;
        T nu_,kappa_;
        uint32_t dim_;

        Matrix<T,Dynamic,Dynamic> SigmaPos_;
        T SigmaDir_;
        Matrix<T,Dynamic,1> muPos_;
        Matrix<T,Dynamic,1> muDir_;

        
        // Sufficient statistics
        Matrix<T,Dynamic,Dynamic> Scatter_;
        Matrix<T,Dynamic,1> mean_;
        uint16_t count_;
};
