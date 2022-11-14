#include "niw.hpp"


// template<typename T>
// NIW<T>::NIW(){};


template<typename T>
NIW<T>::NIW(const Matrix<T,Dynamic,Dynamic>& sigma, 
  const Matrix<T,Dynamic,Dynamic>& mu, T nu,  T kappa, boost::mt19937 *pRndGen)
: Distribution<T>(pRndGen), sigma_0(sigma), mu_0(mu), nu_0(nu), kappa_0(kappa)
{
  assert(sigma_0.rows()==mu_0.size()); 
  assert(sigma_0.cols()==mu_0.size());
};


template<typename T>
NIW<T>::~NIW()
{};