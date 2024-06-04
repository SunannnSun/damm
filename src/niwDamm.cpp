#include "niwDamm.hpp"
#include "riem.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>


template<typename T>
NiwDamm<T>::NiwDamm(const Matrix<T,Dynamic,Dynamic>& sigma, 
  const Matrix<T,Dynamic,Dynamic>& mu, T nu,  T kappa, T sigmaDir, boost::mt19937 &rndGen):
  nu_(nu), kappa_(kappa), sigmaDir_(sigmaDir), rndGen_(rndGen) 
{
  /**
   * This is the NIWDir distribution constructor where the hyperparameters are sparsed from the standard inputs
   * called for IW Gibbs sampler
   * 
   * @param dim is the dimension d of the state variable; i.e. \xi_{pos} in paper
   * @param rndGen the random number generator
   * @param NIW_ptr is created as the pointer to a Niw base distribution
   * 
   */

  dim_ = mu.rows()/2;
  muPos_  = mu(seq(0, dim_-1), all);
  sigmaPos_ = sigma(seq(0, dim_-1), seq(0, dim_-1));

  NIW_ptr = std::make_shared<Niw<T>> (sigmaPos_, muPos_, nu_, kappa_, rndGen_);
};



template<typename T>
NiwDamm<T>::NiwDamm(const Matrix<T,Dynamic,Dynamic>& sigmaPos, const Matrix<T,Dynamic,1>& muPos, T nu, T kappa, T sigmaDir, const Matrix<T,Dynamic,1>& muDir, 
  T count, boost::mt19937 &rndGen):
  sigmaPos_(sigmaPos), muPos_(muPos), nu_(nu), kappa_(kappa), sigmaDir_(sigmaDir), muDir_(muDir), count_(count), dim_(muPos_.rows()), rndGen_(rndGen)
{
  /**
   * This NiwDamm distribution constructor is only called when:
   * 1. called from NiwDamm posterior
   * 
   */
};




template<typename T>
void NiwDamm<T>::getSufficientStatistics(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  const MatrixXd xPos_k = x_k(all, seq(0, dim_-1));
  const MatrixXd xDir_k = x_k(all, seq(dim_, last));

  meanPos_ = xPos_k.colwise().mean().transpose();  
  Matrix<T,Dynamic, Dynamic> x_k_mean; 
  x_k_mean = xPos_k.rowwise() - meanPos_.transpose(); 
  scatterPos_ = (x_k_mean.adjoint() * x_k_mean); 


  meanDir_ = karcherMean(xDir_k);
  scatterDir_ = riemScatter(xDir_k, meanDir_); 

  count_ = x_k.rows();
};


template<typename T>
NiwDamm<T> NiwDamm<T>::posterior(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  getSufficientStatistics(x_k);
  return NiwDamm<T>(
    sigmaPos_+scatterPos_ + ((kappa_*count_)/(kappa_+count_))*(meanPos_-muPos_)*(meanPos_-muPos_).transpose(),
    (kappa_*muPos_+ count_*meanPos_)/(kappa_+count_),
    nu_+count_,
    kappa_+count_,
    (nu_ * sigmaDir_ + scatterDir_)/(nu_+count_),
    meanDir_,

    count_, 
    rndGen_);
};


template<class T>
gaussDamm<T> NiwDamm<T>::samplePosteriorParameter(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  NiwDamm<T> posterior = this ->posterior(x_k);
  return posterior.sampleParameter();
}


template<class T>
gaussDamm<T> NiwDamm<T>::sampleParameter()
{
  Matrix<T,Dynamic,1> meanPos(dim_);
  Matrix<T,Dynamic,Dynamic> covPos(dim_, dim_);
  Matrix<T,Dynamic,1> meanDir(dim_);
  T covDir;


  LLT<Matrix<T,Dynamic,Dynamic> > lltObj(sigmaPos_);
  Matrix<T,Dynamic,Dynamic> cholFacotor = lltObj.matrixL();

  Matrix<T,Dynamic,Dynamic> matrixA(dim_, dim_);
  matrixA.setZero();
  boost::random::normal_distribution<> gauss_(0.0, 1.0);
  for (uint32_t i=0; i<dim_; ++i)  {
    boost::random::chi_squared_distribution<> chiSq_(nu_-i);
    matrixA(i,i) = sqrt(chiSq_(rndGen_)); 
    for (uint32_t j=i+1; j<dim_; ++j)
      matrixA(j, i) = gauss_(rndGen_);
  }
  covPos = matrixA.inverse()*cholFacotor;
  covPos = covPos.transpose()*covPos;


  lltObj.compute(covPos/kappa_);
  cholFacotor = lltObj.matrixL();

  for (uint32_t i=0; i<dim_; ++i)
    meanPos[i] = gauss_(rndGen_);
  meanPos = muPos_ + cholFacotor * meanPos;

  /**
   * Below implements the sampling of a variance from a posterior inverse chi-squared distribution
   * 
   * @note first sample an instance from @param chiSq_ and take the reciprocal; hence @param inv_chi_sqrd
   * as if comes from an inverse chi-squared distribution
   * 
   * @note then convert the instance from chi-squared distribution to scaled chi-squared distribution by
   * multiplying it with the scale and dof, https://en.wikipedia.org/wiki/Scaled_inverse_chi-squared_distribution
   */

  boost::random::chi_squared_distribution<> chiSq_(nu_);
  T inv_chi_sqrd = 1 / chiSq_(rndGen_);
  covDir = inv_chi_sqrd * sigmaDir_ * nu_;

  meanDir = muDir_;


  return gaussDamm<T>(meanPos, covPos, meanDir, covDir, rndGen_);
};



template class NiwDamm<double>;






/*---------------------------------------------------*/
//-------------------Inactive Methods-----------------
/*---------------------------------------------------*/   


/*



template<class T>
T NiwDamm<T>::logPredProb(const Matrix<T,Dynamic,1>& x_i)
{
  // Multivariate student-t distribution
  // https://en.wikipedia.org/wiki/Multivariate_t-distribution
  // https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf pg.21
  int dim = sigmaPos_.cols() + 1;

  Matrix<T,Dynamic,1> x_i_new(dim);
  x_i_new.setZero();
  x_i_new(seq(0, dim-2)) = x_i(seq(0, dim-2));


  Matrix<T,Dynamic,1> x_i_dir(dim-1);
  x_i_dir.setZero();
  x_i_dir(seq(0, dim-2)) = x_i(seq(dim-1, last));
  x_i_new(dim-1) = (rie_log(muDir_, x_i_dir)).norm();

  T doF = nu_ - dim + 1.;
  Matrix<T,Dynamic,Dynamic> scaledSigma = sigmaPos_*(kappa_+1.)/(kappa_*(nu_-dim+1));   

  // Matrix<T,Dynamic,Dynamic> scaledSigma = sigma_*(kappa_+1.)/(kappa_*(nu_-dim+1));   
  LLT<Matrix<T,Dynamic,Dynamic>> lltObj(scaledSigma);

  T logPredProb = boost::math::lgamma(0.5*(doF + dim));
  logPredProb -= boost::math::lgamma(0.5*(doF));
  logPredProb -= 0.5*dim*log(doF);
  logPredProb -= 0.5*dim*log(PI);
  logPredProb -= 0.5*log(lltObj.matrixL().determinant());
  // logPredProb -= (0.5*(doF + dim))*log(1.+ 1/doF*(lltObj.matrixL().solve(x_i_new-mu_)).squaredNorm());
  logPredProb -= (0.5*(doF + dim))*log(1.+ 1/doF*(lltObj.matrixL().solve(x_i_new)).squaredNorm());

  return logPredProb;
};


template<class T>
T NiwDamm<T>::predProb(const Matrix<T,Dynamic,1>& x_i)
{ 
  T logPredProb = this ->logPredProb(x_i);
  return exp(logPredProb);
};


template<typename T>
T NiwDamm<T>::logPostPredProb(const Vector<T,Dynamic>& x_i, const Matrix<T,Dynamic, Dynamic>& x_k)
{
  NiwDamm<T> posterior = this ->posterior(x_k);
  return posterior.logPredProb(x_i);
};



*/