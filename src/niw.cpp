#include "niw.hpp"
#include "niwDir.hpp"
#include <cmath>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>


#define PI 3.141592653589793


template<class T>
NIW<T>::NIW(const MatrixXd &sigma, 
  const VectorXd &mu, T nu, T kappa, boost::mt19937 &rndGen, int base)
: nu_(nu), kappa_(kappa), rndGen_(rndGen) 
{
  if (base == 1){
    dim_ = mu.rows()/2;
    mu_  = mu(seq(0, dim_-1), all);
    sigma_ = sigma(seq(0, dim_-1), seq(0, dim_-1));
  }

  else if (base == 2){
    dim_ = mu.rows();
    mu_  = mu;
    sigma_ = sigma;
  }

  
  // if (mu.rows()==4 ||  mu.rows()==6){
  //   dim_ = mu.rows()/2;
  //   muPos_ = mu(seq(0, dim_-1), all);
  //   muDir_ = mu(seq(dim_, last), all);
  //   SigmaPos_ = Sigma(seq(0, dim_-1), seq(0, dim_-1));
  //   SigmaDir_ = Sigma(last, last);

  //   Sigma_ = SigmaPos_;
  //   mu_    = muPos_;
  //   NIWDIR_ptr = std::make_shared<NIWDIR<T>>(muPos_, SigmaPos_, muDir_, SigmaDir_, nu_, kappa_, 0, rndGen_);
  // }
};


template<class T>
NIW<T>::NIW(const MatrixXd &sigma, 
  const VectorXd &mu, T nu, T kappa, boost::mt19937 &rndGen)
:sigma_(sigma), mu_(mu), nu_(nu), kappa_(kappa), dim_(mu.rows()), rndGen_(rndGen) 
{};




template<class T>
NIW<T>::~NIW()
{};



template<class T>
NIW<T> NIW<T>::posterior(const Matrix<T,Dynamic, Dynamic> &x_k)
{
  getSufficientStatistics(x_k);
  return NIW<T>(
    sigma_+scatter_ + ((kappa_*count_)/(kappa_+count_))*(mean_-mu_)*(mean_-mu_).adjoint(), 
    (kappa_*mu_+ count_*mean_)/(kappa_+count_),
    nu_+count_,
    kappa_+count_, rndGen_);
};


template<class T>
void NIW<T>::getSufficientStatistics(const Matrix<T,Dynamic, Dynamic> &x_k)
{
	mean_ = x_k.colwise().mean();
  MatrixXd x_k_mean = x_k.rowwise() - mean_.transpose();
  scatter_ = x_k_mean.adjoint() * x_k_mean;
	count_ = x_k.rows();
};


template<class T>
T NIW<T>::logPredProb(const Matrix<T,Dynamic,1>& x_i)
{
  // Multivariate student-t distribution
  // https://en.wikipedia.org/wiki/Multivariate_t-distribution
  // https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf pg.21
  T doF = nu_ - dim_ + 1.;
  Matrix<T,Dynamic,Dynamic> scaledSigma = sigma_*(kappa_+1.)/(kappa_*(nu_-dim_+1.));   
  LLT<Matrix<T,Dynamic,Dynamic>> lltObj(scaledSigma);
  
  T logPredProb = boost::math::lgamma(0.5*(doF + dim_));
  logPredProb -= boost::math::lgamma(0.5*(doF));
  logPredProb -= 0.5*dim_*log(doF);
  logPredProb -= 0.5*dim_*log(PI);
  logPredProb -= 0.5*2*log(lltObj.matrixL().determinant());
  logPredProb -= 0.5*(doF + dim_)*log(1.+1./doF*(lltObj.matrixL().solve(x_i-mu_)).squaredNorm());

  return logPredProb;
};


template<class T>
T NIW<T>::predProb(const Matrix<T,Dynamic,1>& x_i)
{ 
  T logPredProb = this ->logPredProb(x_i);
  return exp(logPredProb);
};


template<typename T>
T NIW<T>::logPostPredProb(const Matrix<T,Dynamic,1>& x_i, const Matrix<T,Dynamic, Dynamic>& x_k)
{
  NIW<T> posterior = this ->posterior(x_k);
  return posterior.logPredProb(x_i);
};


template<class T>
Normal<T> NIW<T>::samplePosteriorParameter(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  NIW<T> posterior = this ->posterior(x_k);
  return posterior.sampleParameter();
}



template<class T>
Normal<T> NIW<T>::sampleParameter()
{
  Matrix<T,Dynamic,Dynamic> sampledCov(dim_,dim_);
  Matrix<T,Dynamic,Dynamic> sampledInvCov(dim_,dim_);
  Matrix<T,Dynamic,1> sampledMean(dim_);

  MatrixXd inv_scale_matrix = sigma_.inverse();
  LLT<Matrix<T,Dynamic,Dynamic> > lltObj(inv_scale_matrix);
  Matrix<T,Dynamic,Dynamic> cholFacotor = lltObj.matrixL();

  Matrix<T,Dynamic,Dynamic> matrixA(dim_,dim_);
  matrixA.setZero();
  boost::random::normal_distribution<double> gauss_(0.0, 1.0);
  for (int i=0; i<dim_; ++i){
      for (int j=i; j<dim_; ++j){
        if (i==j){
          boost::random::chi_squared_distribution<> chiSq_(nu_-i);
          matrixA(i, i) =  sqrt(chiSq_(rndGen_));
        }
        else 
          matrixA(j, i) = gauss_(rndGen_);
      }
  }
  sampledInvCov = cholFacotor * matrixA * matrixA.transpose() * cholFacotor.transpose();
  sampledCov = sampledInvCov.inverse();

  // LLT<Matrix<T,Dynamic,Dynamic> > lltObj(Sigma_);
  // Matrix<T,Dynamic,Dynamic> cholFacotor = lltObj.matrixL();
  // for (uint32_t i=0; i<dim_; ++i)  {
  //   boost::random::chi_squared_distribution<> chiSq_(nu_-i);
  //   matrixA(i,i) = sqrt(chiSq_(rndGen_)); 
  //   for (uint32_t j=i+1; j<dim_; ++j)
  //     matrixA(j, i) = gauss_(rndGen_);
  // }
  // sampledCov = matrixA.inverse()*cholFacotor;
  // sampledCov = sampledCov.transpose()*sampledCov;

  // lltObj.compute(sampledCov);
  // cholFacotor = lltObj.matrixL();


  MatrixXd lowerMatrix = (sampledCov/kappa_).llt().matrixL();

  for (uint32_t i=0; i<dim_; ++i)
    sampledMean[i] = gauss_(rndGen_);
  sampledMean =  lowerMatrix * sampledMean + mu_;

  // std::cout << sampledMean << std::endl;
  // std::cout << sampledCov  << std::endl;

  return Normal<T>(sampledMean, sampledCov, rndGen_);
};






template class NIW<double>;