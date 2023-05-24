#include "niw.hpp"
#include "niwDir.hpp"
#include <cmath>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>


#define PI 3.141592653589793


template<class T>
NIW<T>::NIW(const MatrixXd &Sigma, 
  const VectorXd &mu, T nu, T kappa, boost::mt19937 &rndGen)
: nu_(nu), kappa_(kappa), rndGen_(rndGen) 
{
  if (Sigma.rows()==4 ||  Sigma.rows()==6){
    dim_ = mu.rows()/2;
    muPos_ = mu(seq(0, dim_-1), all);
    muDir_ = mu(seq(dim_, last), all);
    SigmaPos_ = Sigma(seq(0, dim_-1), seq(0, dim_-1));
    SigmaDir_ = Sigma(last, last);

    Sigma_ = SigmaPos_;
    mu_    = muPos_;
    NIWDIR_ptr = std::make_shared<NIWDIR<T>>(muPos_, SigmaPos_, muDir_, SigmaDir_, nu_, kappa_, 0, rndGen_);
  }
  else {
    dim_ = mu.rows();
    Sigma_ = Sigma;
    mu_    = mu;
  }
};



template<class T>
NIW<T>::~NIW()
{};




// template<class T>
// T NIW<T>::logPosteriorProb(const Vector<T,Dynamic> &x_i, const Matrix<T,Dynamic, Dynamic> &x_k)
// {
//   NIW<T> posterior = this->posterior(x_k);
//   return posterior.logProb(x_i);
// };


// template<class T>
// T NIW<T>::logPosteriorProb(const Matrix<T,Dynamic, Dynamic>& x_i, const Matrix<T,Dynamic, Dynamic>& x_j, )
// {
//   NIW<T> posterior = this ->posterior(x_k);

//   return posterior.logProb(x_i);
// };


template<class T>
NIW<T> NIW<T>::posterior(const Matrix<T,Dynamic, Dynamic> &x_k)
{
  getSufficientStatistics(x_k);
  return NIW<T>(
    Sigma_+Scatter_ + ((kappa_*count_)/(kappa_+count_))
      *(mean_-mu_)*(mean_-mu_).transpose(), 
    (kappa_*mu_+ count_*mean_)/(kappa_+count_),
    nu_+count_,
    kappa_+count_, rndGen_);
};


template<class T>
void NIW<T>::getSufficientStatistics(const Matrix<T,Dynamic, Dynamic> &x_k)
{
	mean_ = x_k.colwise().mean();
  Matrix<T,Dynamic, Dynamic> x_k_mean;
  x_k_mean = x_k.rowwise() - mean_.transpose();
  Scatter_ = x_k_mean.adjoint() * x_k_mean;
	count_ = x_k.rows();
};


template<class T>
T NIW<T>::logPredProb(const Matrix<T,Dynamic,1>& x_i)
{
  // Multivariate student-t distribution
  // https://en.wikipedia.org/wiki/Multivariate_t-distribution
  // https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf pg.21
  T doF = nu_ - dim_ + 1.;
  Matrix<T,Dynamic,Dynamic> scaledSigma = Sigma_*(kappa_+1.)/(kappa_*(nu_-dim_+1.));   
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
  Matrix<T,Dynamic,1> sampledMean(dim_);

  LLT<Matrix<T,Dynamic,Dynamic> > lltObj(Sigma_);
  Matrix<T,Dynamic,Dynamic> cholFacotor = lltObj.matrixL();

  Matrix<T,Dynamic,Dynamic> matrixA(dim_,dim_);
  matrixA.setZero();
  boost::random::normal_distribution<> gauss_;
  for (uint32_t i=0; i<dim_; ++i)  {
    boost::random::chi_squared_distribution<> chiSq_(nu_-i);
    matrixA(i,i) = sqrt(chiSq_(rndGen_)); 
    for (uint32_t j=i+1; j<dim_; ++j)
      matrixA(j, i) = gauss_(rndGen_);
  }
  sampledCov = matrixA.inverse()*cholFacotor;
  sampledCov = sampledCov.transpose()*sampledCov;


  lltObj.compute(sampledCov);
  cholFacotor = lltObj.matrixL();

  for (uint32_t i=0; i<dim_; ++i)
    sampledMean[i] = gauss_(rndGen_);
  sampledMean = cholFacotor * sampledMean / sqrt(kappa_) + mu_;
  
  return Normal<T>(sampledMean, sampledCov, rndGen_);
};




// template<class T>
// T NIW<T>::logPosteriorProb(const Matrix<T,Dynamic,Dynamic>& x, VectorXu& z, uint32_t k, uint32_t i)
// {
//   uint32_t z_i = z[i];
//   z[i] = k+1; // so that we definitely not use x_i in posterior computation 
//   // (since the posterior is only computed form x_{z==k})
//   NIW posterior = this->posterior(x,z,k);
//   z[i] = z_i; // reset to old value
//   return posterior.logProb(x.col(i));
// };



template class NIW<double>;