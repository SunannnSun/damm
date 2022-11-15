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


template<typename T>
double NIW<T>::logPosteriorProb()
{
  return 12;
};

// template<typename T>
// T NIW<T>::logPosteriorProb(const Matrix<T,Dynamic,Dynamic>& x, VectorXu& z, uint32_t k, uint32_t i)
// {
//   uint32_t z_i = z[i];
//   z[i] = k+1; // so that we definitely not use x_i in posterior computation 
//   // (since the posterior is only computed form x_{z==k})
//   NIW posterior = this->posterior(x,z,k);
//   z[i] = z_i; // reset to old value
//   return posterior.logProb(x.col(i));
// };