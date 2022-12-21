#include "normal.hpp"
#include <boost/random/mersenne_twister.hpp>


#define PI 3.141592653589793


template<class T>
Normal<T>::Normal(const Matrix<T,Dynamic,Dynamic>& sigma, 
  const Matrix<T,Dynamic,1>& mu, boost::mt19937* pRndGen)
: sigma_(sigma), mu_(mu), dim_(mu.size()), pRndGen_(pRndGen) 
{
  assert(sigma_.rows()==mu_.size()); 
  assert(sigma_.cols()==mu_.size());
};


template<class T>
Normal<T>::~Normal()
{};

template<class T>
T Normal<T>::logProb(const Matrix<T,Dynamic,1>& x_i)
{ 
  T logProb =  dim_ * log(2*PI);
  logProb += log(sigma_.determinant());
  logProb += ((x_i-mu_).transpose()*sigma_.inverse()*(x_i-mu_)).sum();

  return -0.5 * logProb;
};


template class Normal<double>;
template class Normal<float>;