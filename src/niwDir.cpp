#include "niwDir.hpp"
#include "karcher.hpp"
#include <cmath>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>


template<typename T>
NIWDIR<T>::NIWDIR(const Matrix<T,Dynamic,Dynamic>& sigma, 
  const Matrix<T,Dynamic,Dynamic>& mu, T nu,  T kappa, boost::mt19937 &rndGen):
  Sigma_(sigma), mu_(mu), nu_(nu), kappa_(kappa), dim_(mu.size()), rndGen_(rndGen)
{
  
  assert(Sigma_.rows()==mu_.size()); 
  assert(Sigma_.cols()==mu_.size());
};


template<typename T>
NIWDIR<T>::~NIWDIR()
{};


template<typename T>
T NIWDIR<T>::logPosteriorProb(const Vector<T,Dynamic>& x_i, const Matrix<T,Dynamic, Dynamic>& x_k)
{
  NIWDIR<T> posterior = this ->posterior(x_k);
  return posterior.logProb(x_i, x_k);
  // return x_i.rows() + x_k.cols();
};

template<typename T>
NIWDIR<T> NIWDIR<T>::posterior(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  getSufficientStatistics(x_k);
  return NIWDIR<T>(
    Sigma_+scatter_ + ((kappa_*count_)/(kappa_+count_))
      *(mean_-mu_)*(mean_-mu_).transpose(), 
    (kappa_*mu_+ count_*mean_)/(kappa_+count_),
    nu_+count_,
    kappa_+count_, rndGen_);
};

template<typename T>
void NIWDIR<T>::getSufficientStatistics(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  mean_.setZero(dim_);
	mean_(seq(0, dim_-2)) = x_k(all, (seq(0, dim_-2))).colwise().mean().transpose();
  Matrix<T,Dynamic, Dynamic> x_k_mean;
  // // x_k_mean = x_k.rowwise() - mean.transpose();
  x_k_mean = x_k.rowwise() - x_k.colwise().mean();

  scatter_.setZero(dim_, dim_);
  scatter_(seq(0, dim_-2), seq(0, dim_-2)) = (x_k_mean.adjoint() * x_k_mean)(seq(0, dim_-2), seq(0, dim_-2));
  count_ = x_k.rows();
};


template<class T>
Normal<T> NIWDIR<T>::samplePosteriorParameter(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  NIWDIR<T> posterior = this ->posterior(x_k);
  return posterior.sampleParameter();
}


template<class T>
Normal<T> NIWDIR<T>::sampleParameter()
{
  Matrix<T,Dynamic,Dynamic> sampledCov(dim_,dim_);
  Matrix<T,Dynamic,1> sampledMean(dim_);

  LLT<Matrix<T,Dynamic,Dynamic> > lltObj(Sigma_);
  Matrix<T,Dynamic,Dynamic> cholFacotor = lltObj.matrixL();

  Matrix<T,Dynamic,Dynamic> matrixA(dim_,dim_);
  matrixA.setZero();
  boost::random::normal_distribution<> gauss_;
  for (uint32_t i=0; i<dim_; ++i)
  {
    boost::random::chi_squared_distribution<> chiSq_(nu_-i);
    matrixA(i,i) = sqrt(chiSq_(rndGen_)); 
    for (uint32_t j=i+1; j<dim_; ++j)
    {
      matrixA(j, i) = gauss_(rndGen_);
    }
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








template<typename T>
T NIWDIR<T>::logProb(const Matrix<T,Dynamic,1>& x_i)
{
  Matrix<T,Dynamic,1>  x_i_dir(dim_);
  x_i_dir.setZero();
  x_i_dir(seq(0,last-1)) = x_i(seq(0,last-1));

  // std::cout << x_i << std::endl;
  // std::cout << x_i_dir << std::endl;
  // std::cout << x_i << std::endl;
  // using multivariate student-t distribution; missing terms?
  // https://en.wikipedia.org/wiki/Multivariate_t-distribution
  // https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf pg.21
  T doF = nu_ - dim_ + 1.;
  Matrix<T,Dynamic,Dynamic> scaledSigma = Sigma_*(kappa_+1.)/(kappa_*(nu_-dim_+1));                       
  // T logProb = 0;
  T logProb = boost::math::lgamma(0.5*(doF + dim_));  
  logProb -= boost::math::lgamma(0.5*(doF));
  logProb -= 0.5*dim_*log(doF);
  logProb -= 0.5*dim_*log(PI);
  logProb -= 0.5*log(scaledSigma.determinant());
  // logProb -= 0.5*((scaledSigma.eigenvalues()).array().log().sum()).real();
  logProb -= (0.5*(doF + dim_))
    *log(1.+ 1/doF*((x_i_dir-mu_).transpose()*scaledSigma.inverse()*(x_i_dir-mu_)).sum());
  // approximate using moment-matched Gaussian; Erik Sudderth PhD essay
  return logProb;
};


template<typename T>
T NIWDIR<T>::logProb(const Matrix<T,Dynamic,1>& x_i, const Matrix<T,Dynamic,Dynamic>& x_k)
{
  Matrix<T,Dynamic,1>  x_i_dir(dim_);
  x_i_dir.setZero();
  x_i_dir(seq(0,last-1)) = x_i(seq(0,last-1));
  x_i_dir(-1) = (rie_log(x_i, karcherMean(x_k))).norm();

  // std::cout << x_i << std::endl;
  // std::cout << x_i_dir << std::endl;
  // std::cout << x_i << std::endl;
  // using multivariate student-t distribution; missing terms?
  // https://en.wikipedia.org/wiki/Multivariate_t-distribution
  // https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf pg.21
  T doF = nu_ - dim_ + 1.;
  Matrix<T,Dynamic,Dynamic> scaledSigma = Sigma_*(kappa_+1.)/(kappa_*(nu_-dim_+1));                       
  // T logProb = 0;
  T logProb = boost::math::lgamma(0.5*(doF + dim_));  
  logProb -= boost::math::lgamma(0.5*(doF));
  logProb -= 0.5*dim_*log(doF);
  logProb -= 0.5*dim_*log(PI);
  logProb -= 0.5*log(scaledSigma.determinant());
  // logProb -= 0.5*((scaledSigma.eigenvalues()).array().log().sum()).real();
  logProb -= (0.5*(doF + dim_))
    *log(1.+ 1/doF*((x_i_dir-mu_).transpose()*scaledSigma.inverse()*(x_i_dir-mu_)).sum());
  // approximate using moment-matched Gaussian; Erik Sudderth PhD essay
  return logProb;
};


template<class T>
T NIWDIR<T>::prob(const Matrix<T,Dynamic,1>& x_i)
{ 
  T logProb = this ->logProb(x_i);
  return exp(logProb);
};

// template<typename T>
// Matrix<T, Dynamic, 1> NIWDIR<T>::karcherMean(const Matrix<T,Dynamic, Dynamic>& x_k)
// {
//   float tolerance = 0.01;
//   T angle;
//   Matrix<T, Dynamic, 1> angle_sum(x_k.cols()/2);
//   Matrix<T, Dynamic, 1> x_tp(x_k.cols()/2);   // x in tangent plane
//   Matrix<T, Dynamic, 1> x(x_k.cols()/2);
//   Matrix<T, Dynamic, 1> p(x_k.cols()/2);
  
//   p = x_k(0, seq(x_k.cols()/2, last)).transpose();
//   if (x_k.rows() == 1) return p;

//   while (1)
//   { 
//     angle_sum.setZero();
//     for (int i=0; i<x_k.rows(); ++i)
//     {
//       x = x_k(i, seq(x_k.cols()/2, last)).transpose();
//       angle_sum = angle_sum + rie_log(p, x);
//     }
//     x_tp = 1. / x_k.rows() * angle_sum;
//     // cout << x_tp.norm() << endl;
//     if (x_tp.norm() < tolerance) 
//     {
//       return p;
//     }
//     p = rie_exp(p, x_tp);
//   }
// };


template class NIWDIR<double>;