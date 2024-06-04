/*
* Normal Inverse Wishart distribution class
*/

#pragma once

#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include "gauss.hpp"
#include <memory>


using namespace Eigen;

template<typename T> //cyclic dependency
class NiwDamm;


template<typename T>
class Niw
{
    public:
        Niw(const MatrixXd &sigma, const VectorXd &mu, T nu, T kappa, boost::mt19937 &rndGen, int base);
        Niw(const MatrixXd &sigma, const VectorXd &mu, T nu, T kappa, boost::mt19937 &rndGen);
        Niw(){};
        ~Niw(){};

        void getSufficientStatistics(const Matrix<T,Dynamic, Dynamic> &x_k);
        Niw<T> posterior(const Matrix<T,Dynamic, Dynamic> &x_k);
        Gauss<T> samplePosteriorParameter(const Matrix<T,Dynamic, Dynamic> &x_k);
        Gauss<T> sampleParameter();

 
    private:
        boost::mt19937 rndGen_;
        uint32_t dim_;  // 2 or 3 for base 1; 4 or 6 for base 2

        // hyperparameter
        Matrix<T,Dynamic,Dynamic> sigma_;
        Matrix<T,Dynamic,1> mu_;
        T nu_,kappa_;

        // sufficient statistics
        Matrix<T,Dynamic,Dynamic> scatter_;
        Matrix<T,Dynamic,1> mean_;
        uint16_t count_;
};






/*---------------------------------------------------*/
//-------------------Inactive Methods-----------------
/*---------------------------------------------------*/   
// T logPostPredProb(const Matrix<T,Dynamic,1>& x_i, const Matrix<T,Dynamic, Dynamic>& x_k);
// T logPredProb(const Matrix<T,Dynamic,1> &x_i);
// T predProb(const Matrix<T,Dynamic,1> &x_i);

/*---------------------------------------------------*/
//-------------------Inactive Members-----------------
/*---------------------------------------------------*/
// std::shared_ptr<NiwDamm<T>> NIWDIR_ptr;
// Matrix<T,Dynamic,Dynamic> SigmaPos_;
// T SigmaDir_;
// Matrix<T,Dynamic,1> muPos_;
// Matrix<T,Dynamic,1> muDir_;