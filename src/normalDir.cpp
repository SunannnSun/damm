#include "normalDir.hpp"
#include "karcher.hpp"


#define PI 3.141592653589793


template<class T>
NormalDir<T>::NormalDir(const Matrix<T,Dynamic,1> &meanPos, const Matrix<T,Dynamic, Dynamic> &covPos,
const Matrix<T,Dynamic,1>& meanDir, T covDir, boost::mt19937 &rndGen) 
:meanPos_(meanPos), covPos_(covPos), meanDir_(meanDir), covDir_(covDir), rndGen_(rndGen)
{
  cov_.setZero(3, 3);
  cov_(seq(0,1), seq(0,1)) = covPos_;
  cov_(2, 2) = covDir_;
  mean_.setZero(3);
  mean_(seq(0,1)) = meanPos_;
};


template<class T>
T NormalDir<T>::logProb(const Matrix<T,Dynamic,1> &x_i)
{ 
  int dim = 3;
  Matrix<T,Dynamic,1> x_i_new(dim);
  x_i_new.setZero();
  x_i_new(seq(0, dim-2)) = x_i(seq(0, dim-2));
  Matrix<T,Dynamic,1> x_i_dir(2);
  x_i_dir << x_i[dim-1] , x_i[dim];
  x_i_new(dim-1) = (rie_log(meanDir_, x_i_dir)).norm();
 
  LLT<Matrix<T,Dynamic,Dynamic>> lltObj(cov_);
  T logProb =  dim_ * log(2*PI);
  logProb += 2 * log(lltObj.matrixL().determinant());
  logProb += (lltObj.matrixL().solve(x_i_new-mean_)).squaredNorm();

  return -0.5 * logProb;
};


template class NormalDir<double>;