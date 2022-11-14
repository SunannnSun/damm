#include "niw.hpp"


template<typename T>
NIW<T>::NIW(){};


template<typename T>
NIW<T>::NIW(const Matrix<T,Dynamic,Dynamic>& Delta, 
  const Matrix<T,Dynamic,Dynamic>& theta, T nu,  T kappa)
: Delta_(Delta), theta_(theta), nu_(nu), kappa_(kappa)
{
  assert(Delta_.rows()==theta_.size()); 
  assert(Delta_.cols()==theta_.size());
};


template<typename T>
NIW<T>::~NIW()
{};