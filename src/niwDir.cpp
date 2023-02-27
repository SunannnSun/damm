#include "niwDir.hpp"
#include "karcher.hpp"
#include <cmath>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/normal_distribution.hpp>


template<typename T>
NIWDIR<T>::NIWDIR(const Matrix<T,Dynamic,Dynamic>& sigma, 
  const Matrix<T,Dynamic,Dynamic>& mu, T nu,  T kappa, boost::mt19937 &rndGen):
  Sigma_(sigma), mu_(mu), nu_(nu), kappa_(kappa), dim_(mu.size()), rndGen_(rndGen) //dim_ is dimParam defined in main.cpp
{
  muPos_ = mu_(seq(0, 1));
  SigmaPos_ = Sigma_(seq(0, 1), seq(0, 1));
  muDir_ = mu_(seq(2, 3));
  SigmaDir_ = Sigma_(2, 2);
  // assert(Sigma_.cols()==mu_.size());
};



template<typename T>
NIWDIR<T>::NIWDIR(const Matrix<T,Dynamic,1>& muPos, const Matrix<T,Dynamic,Dynamic>& SigmaPos,  
  const Matrix<T,Dynamic,1>& muDir, T SigmaDir,
  T nu, T kappa, boost::mt19937 &rndGen):
  SigmaPos_(SigmaPos), SigmaDir_(SigmaDir), muPos_(muPos), muDir_(muDir), nu_(nu), kappa_(kappa), rndGen_(rndGen) //dim_ is dimParam defined in main.cpp
{};



template<typename T>
NIWDIR<T>::~NIWDIR()
{};


template<typename T>
void NIWDIR<T>::getSufficientStatistics(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  meanDir_ = karcherMean(x_k);
  // std::cout << meanDir_ << std::endl;
  meanPos_ = x_k(all, (seq(0, 1))).colwise().mean().transpose();  //!!!!later change the number 1 to accomodate for 3D data
  // Matrix<T,Dynamic, 1> mean(meanPos_.size() + meanDir_.size());
  // mean << meanPos_, meanDir_;
  
 
  Matrix<T,Dynamic, Dynamic> x_k_mean;  //this is the value of each data point subtracted from the mean value calculated from the previous procedure
  x_k_mean = x_k.rowwise() - x_k.colwise().mean(); //Though formally it should only be the first two dimension that contains the position, but it's ok to do the full dimension and only use the first two dimension later
  ScatterPos_ = (x_k_mean.adjoint() * x_k_mean)(seq(0, 1), seq(0, 1)); //!!!!later change the number 1 to accomodate for 3D data
  ScatterDir_ = SigmaDir_;  //!!! This is Riemannian covariance mentioned in the paper but can be approximated as zero now and complete its form later 
  // Matrix<T,Dynamic, Dynamic> Scatter(3, 3); //!!!!later change the number 3 to accomodate for 3D data
  // Scatter.setZero();
  // Scatter(seq(0, 1), seq(0, 1)) = ScatterPos_; //!!!!later change the number 1 to accomodate for 3D data
  // Scatter(2, 2) = ScatterDir_;   //!!!!later change the number 2 to accomodate for 3D data
  

  // mean_ = mean;
  // Scatter_ = Scatter;
  count_ = x_k.rows();
  

  // std::cout << mean_ << std::endl;
  // std::cout << Scatter_ <<std::endl;
  // std::cout << mean_ <<std::endl;
};


template<typename T>
NIWDIR<T> NIWDIR<T>::posterior(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  getSufficientStatistics(x_k);
  // std::cout << meanDir_ << std::endl;
  // std::cout << "here0" << std::endl;
  return NIWDIR<T>(
    (kappa_*muPos_+ count_*meanPos_)/(kappa_+count_),
    SigmaPos_+ScatterPos_ + ((kappa_*count_)/(kappa_+count_))*(meanPos_-muPos_)*(meanPos_-muPos_).transpose(),
    meanDir_,
    SigmaDir_,
    nu_+count_,
    kappa_+count_, 
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
  // std::cout << muDir_ << std::endl;
  int dim = 2;

  Matrix<T,Dynamic,1> meanPos(dim);
  Matrix<T,Dynamic,Dynamic> covPos(dim, dim);
  Matrix<T,Dynamic,1> meanDir(dim);
  T covDir;


  LLT<Matrix<T,Dynamic,Dynamic> > lltObj(SigmaPos_);
  Matrix<T,Dynamic,Dynamic> cholFacotor = lltObj.matrixL();

  Matrix<T,Dynamic,Dynamic> matrixA(dim, dim);
  matrixA.setZero();
  boost::random::normal_distribution<> gauss_;
  for (uint32_t i=0; i<dim; ++i)
  {
    boost::random::chi_squared_distribution<> chiSq_(nu_-i);
    matrixA(i,i) = sqrt(chiSq_(rndGen_)); 
    for (uint32_t j=i+1; j<dim; ++j)
    {
      matrixA(j, i) = gauss_(rndGen_);
    }
  }
  covPos = matrixA.inverse()*cholFacotor;
  covPos = covPos.transpose()*covPos;


  lltObj.compute(covPos);
  cholFacotor = lltObj.matrixL();

  for (uint32_t i=0; i<dim; ++i)
    meanPos[i] = gauss_(rndGen_);
  meanPos = cholFacotor * meanPos / sqrt(kappa_) + muPos_;

  // std::cout << "here2" << std::endl;
  covDir = SigmaDir_;


  meanDir = muDir_; //the mean location of the normal distribution is sampled from the posterior mu of niw which is the data mean of the data points in cluster
  // std::cout << muDir_<<std::endl;

  return NormalDir<T>(meanPos, covPos, meanDir, covDir, rndGen_);
};




template class NIWDIR<double>;
