#include "normal.hpp"
#include <boost/random/mersenne_twister.hpp>


#define PI 3.141592653589793


template<class T>
Normal<T>::Normal(const Matrix<T,Dynamic,1>& mu, 
  const Matrix<T,Dynamic,Dynamic>& Sigma, boost::mt19937* pRndGen)
:mu_(mu), Sigma_(Sigma), dim_(mu.size()), pRndGen_(pRndGen) 
{
  assert(Sigma_.rows()==mu_.size()); 
  assert(Sigma_.cols()==mu_.size());
};


template<class T>
Normal<T>::~Normal()
{};

template<class T>
T Normal<T>::logProb(const Matrix<T,Dynamic,1>& x_i)
{ 
  T logProb =  dim_ * log(2*PI);
  logProb += log(Sigma_.determinant());
  logProb += ((x_i-mu_).transpose()*Sigma_.inverse()*(x_i-mu_)).sum();

  return -0.5 * logProb;
};


template class Normal<double>;
template class Normal<float>;