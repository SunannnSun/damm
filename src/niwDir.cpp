#include "niwDir.hpp"
#include "karcher.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>


template<typename T>
NIWDIR<T>::NIWDIR(const Matrix<T,Dynamic,Dynamic>& sigma, 
  const Matrix<T,Dynamic,Dynamic>& mu, T nu,  T kappa, T sigmaDir, boost::mt19937 &rndGen):
  nu_(nu), kappa_(kappa), sigmaDir_(sigmaDir), rndGen_(rndGen) //dim_ is dimParam defined in main.cpp
{
  dim_ = mu.rows()/2;
  muPos_  = mu(seq(0, dim_-1), all);
  sigmaPos_ = sigma(seq(0, dim_-1), seq(0, dim_-1));


  // NIW_ptr = std::make_shared<NIW<T>>(sigmaPos_, muPos_, nu_, kappa_, rndGen_);
};



template<typename T>
NIWDIR<T>::NIWDIR(const Matrix<T,Dynamic,Dynamic>& sigmaPos, const Matrix<T,Dynamic,1>& muPos, T nu, T kappa, T sigmaDir, const Matrix<T,Dynamic,1>& muDir, 
  T count, boost::mt19937 &rndGen):
  sigmaPos_(sigmaPos), muPos_(muPos), nu_(nu), kappa_(kappa), sigmaDir_(sigmaDir), muDir_(muDir), count_(count), dim_(muPos_.rows()), rndGen_(rndGen) //dim_ is dimParam defined in main.cpp
{};



template<typename T>
NIWDIR<T>::~NIWDIR()
{};



template<typename T>
void NIWDIR<T>::getSufficientStatistics(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  const MatrixXd xPos_k = x_k(all, seq(0, dim_-1));
  const MatrixXd xDir_k = x_k(all, seq(dim_, last));

  meanPos_ = xPos_k.colwise().mean().transpose();  
  Matrix<T,Dynamic, Dynamic> x_k_mean; 
  x_k_mean = xPos_k.rowwise() - meanPos_.transpose(); 
  scatterPos_ = (x_k_mean.adjoint() * x_k_mean); 


  meanDir_ = karcherMean(xDir_k);
  scatterDir_ = karcherScatter(xDir_k, meanDir_); 

  count_ = x_k.rows();
};


template<typename T>
NIWDIR<T> NIWDIR<T>::posterior(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  getSufficientStatistics(x_k);
  return NIWDIR<T>(
    sigmaPos_+scatterPos_ + ((kappa_*count_)/(kappa_+count_))*(meanPos_-muPos_)*(meanPos_-muPos_).transpose(),
    (kappa_*muPos_+ count_*meanPos_)/(kappa_+count_),
    nu_+count_,
    kappa_+count_,
    // NOTE ON Posterior SIGMA DIRRECTION
    // sigmaDir_+scatterDir_ + ((kappa_*count_)/(kappa_+count_))*pow(rie_log(meanDir_, muDir_).norm(), 2),
    sigmaDir_+scatterDir_ ,
    meanDir_,

    count_, 
    rndGen_);
};


template<class T>
NormalDir<T> NIWDIR<T>::samplePosteriorParameter(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  NIWDIR<T> posterior = this ->posterior(x_k);
  return posterior.sampleParameter();
}


template<class T>
NormalDir<T> NIWDIR<T>::sampleParameter()
{
  Matrix<T,Dynamic,1> meanPos(dim_);
  Matrix<T,Dynamic,Dynamic> covPos(dim_, dim_);
  Matrix<T,Dynamic,1> meanDir(dim_);
  T covDir;


  LLT<Matrix<T,Dynamic,Dynamic> > lltObj(sigmaPos_);
  Matrix<T,Dynamic,Dynamic> cholFacotor = lltObj.matrixL();

  Matrix<T,Dynamic,Dynamic> matrixA(dim_, dim_);
  matrixA.setZero();
  boost::random::normal_distribution<> gauss_;
  for (uint32_t i=0; i<dim_; ++i)  {
    boost::random::chi_squared_distribution<> chiSq_(nu_-i);
    matrixA(i,i) = sqrt(chiSq_(rndGen_)); 
    for (uint32_t j=i+1; j<dim_; ++j)
      matrixA(j, i) = gauss_(rndGen_);
  }
  covPos = matrixA.inverse()*cholFacotor;
  covPos = covPos.transpose()*covPos;


  lltObj.compute(covPos);
  cholFacotor = lltObj.matrixL();

  for (uint32_t i=0; i<dim_; ++i)
    meanPos[i] = gauss_(rndGen_);
  meanPos = cholFacotor * meanPos / sqrt(kappa_) + muPos_;


  // covDir = SigmaDir_;
  // EVERYTHING BELOW NEEDS TO BE RECTIFIED
  boost::random::chi_squared_distribution<> chiSq_(nu_);
  T inv_chi_sqrd = 1 / chiSq_(rndGen_);
  covDir = inv_chi_sqrd * SigmaDir_ / count_ * nu_;
  // if (covDir > 0.2) 
  //   covDir = 0.07;
  // covDir = std::min(covDir, 0.1);  
  covDir = 0.5;

  // if (dim==2)
  // {
  //   boost::random::normal_distribution<> normal_(0, covDir/kappa_);
  //   T angDiff = normal_(rndGen_);
  //   Matrix<T,Dynamic,Dynamic> rotationMatrix(2, 2); // change the rotation matrix dimension later on to accomodate for 3D data
  //   rotationMatrix << cos(angDiff), -sin(angDiff), sin(angDiff), cos(angDiff);
  //   meanDir = (muDir_.transpose() * rotationMatrix).transpose();
  // }
  // else 
  
  meanDir = muDir_;

  // std::cout << meanDir << std::endl;

  return NormalDir<T>(meanPos, covPos, meanDir, covDir, rndGen_);
};



template<class T>
T NIWDIR<T>::logPredProb(const Matrix<T,Dynamic,1>& x_i)
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
T NIWDIR<T>::predProb(const Matrix<T,Dynamic,1>& x_i)
{ 
  T logPredProb = this ->logPredProb(x_i);
  return exp(logPredProb);
};


template<typename T>
T NIWDIR<T>::logPostPredProb(const Vector<T,Dynamic>& x_i, const Matrix<T,Dynamic, Dynamic>& x_k)
{
  NIWDIR<T> posterior = this ->posterior(x_k);
  return posterior.logPredProb(x_i);
};

template class NIWDIR<double>;
