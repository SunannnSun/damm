#include "gauss.hpp"

#define PI 3.141592653589793


template<class T>
Gauss<T>::Gauss(const Matrix<T,Dynamic,1> &mu, const Matrix<T,Dynamic,Dynamic> &sigma, boost::mt19937 &rndGen)
:mu_(mu), sigma_(sigma), dim_(mu.size()), rndGen_(rndGen) 
{
  assert(sigma_.rows()==mu_.size()); 
  assert(sigma_.cols()==mu_.size());
};



template<class T>
T Gauss<T>::logProb(const Matrix<T,Dynamic,1> &x_i)
{ 
  LLT<Matrix<T,Dynamic,Dynamic>> lltObj(sigma_);
  T logProb =  dim_ * log(2*PI);
  if (sigma_.rows()==2)
    logProb += log(sigma_(0, 0) * sigma_(1, 1) - sigma_(0, 1) * sigma_(1, 0));
  else
    logProb += 2 * log(lltObj.matrixL().determinant());
  logProb += (lltObj.matrixL().solve(x_i-mu_)).squaredNorm();

  return -0.5 * logProb;
};


template<class T>
T Gauss<T>::prob(const Matrix<T,Dynamic,1> &x_i)
{ 
  T logProb = this ->logProb(x_i);
  return exp(logProb);
};


template class Gauss<double>;