#include "niwDir.hpp"

// template<typename T>
// NIW<T>::NIW(){};


template<typename T>
NIWDIR<T>::NIWDIR(const Matrix<T,Dynamic,Dynamic>& sigma, 
  const Matrix<T,Dynamic,Dynamic>& mu, T nu,  T kappa, boost::mt19937 *pRndGen)
: Distribution<T>(pRndGen), sigma_(sigma), mu_(mu), nu_(nu), kappa_(kappa), dim_(mu.size())
{
  assert(sigma_.rows()==mu_.size()); 
  assert(sigma_.cols()==mu_.size());
};


template<typename T>
NIWDIR<T>::~NIWDIR()
{};


// template<typename T>
// T NIW<T>::logPosteriorProb(const Vector<T,Dynamic>& x_i, const Matrix<T,Dynamic, Dynamic>& x_k)
// {
//   NIW<T> posterior = this ->posterior(x_k);
//   return posterior.logProb(x_i);
//   // return x_i.rows() + x_k.cols();
// };

// template<typename T>
// NIW<T> NIW<T>::posterior(const Matrix<T,Dynamic, Dynamic>& x_k)
// {
//   getSufficientStatistics(x_k);
//   return NIW<T>(
//     sigma_+scatter_ + ((kappa_*count_)/(kappa_+count_))
//       *(mean_-mu_)*(mean_-mu_).transpose(), 
//     (kappa_*mu_+ count_*mean_)/(kappa_+count_),
//     nu_+count_,
//     kappa_+count_,
//     this->pRndGen_);
// };

template<typename T>
void NIWDIR<T>::getSufficientStatistics(const Matrix<T,Dynamic, Dynamic>& x_k)
{

	mean_ = x_k.colwise().mean();
  mean_(dim_-1) = 0; //change the last directional element to zero;

  // std::cout << mean_ << std::endl;
  // Matrix<T,Dynamic, Dynamic> x_k_mean;
  // MatrixXd x_k
  // x_k_mean = x_k.rowwise() - mean_.transpose();
  // x_k_mean = x_k.rowwise() - x_k.colwise().mean();

  // scatter_ = x_k_mean.adjoint() * x_k_mean;
	// count_ = x_k.rows();
};

template<typename T>
T NIWDIR<T>::logProb(const Matrix<T,Dynamic,1>& x_i)
{
  Matrix<T,Dynamic,1>  x_i_dir(dim_);
  x_i_dir.setZero();
  x_i_dir(seq(0,last-1), all) = x_i(seq(0,last-1), all);


  // std::cout << x_i << std::endl;
  // std::cout << x_i_dir << std::endl;
  // std::cout << x_i << std::endl;
  // using multivariate student-t distribution; missing terms?
  // https://en.wikipedia.org/wiki/Multivariate_t-distribution
  // https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf pg.21
  T doF = nu_ - dim_ + 1.;
  Matrix<T,Dynamic,Dynamic> scaledSigma = sigma_*(kappa_+1.)/(kappa_*(nu_-dim_+1));                       
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
Matrix<T, Dynamic, 1> NIWDIR<T>::karcherMean(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  float tolerance = 0.01;
  T angle;
  Matrix<T, Dynamic, 1> angle_sum;
  Matrix<T, Dynamic, 1> x_tp; // x in tangent plane
  Matrix<T, Dynamic, 1> x;
  Matrix<T, Dynamic, 1> p;
  
  if (x_k.rows() == 1)
  {
    p = x_k(0, seq(x_k.cols()/2, last)).transpose();
    return p;
  }

  angle_sum.setZero(x_k.cols()/2);
  x_tp.setZero(x_k.cols()/2);  x_tp = (x_tp.array() + 1).matrix();
  p.setZero(x_k.cols()/2); p(0) = 1;

  while (x_tp.norm() > tolerance)
  {
    for (int i=0; i<x_k.rows(); ++i)
    {
      x = x_k(i, seq(x_k.cols()/2, last)).transpose();
      // std::cout << x << std::endl;
      angle = std::acos(p.dot(x));
      if (angle > 0.01)
      {
        angle_sum = angle_sum + (x-p*std::cos(angle))*angle/std::sin(angle);
      }
      x_tp = 1 / x_k.rows() * angle_sum;
      p = p * std::cos(x_tp.norm()) + x_tp / x_tp.norm() * std::sin(x_tp.norm());
    }
  }
  return p;
};

