#include "normalDir.hpp"
#include "karcher.hpp"


#define PI 3.141592653589793


template<class T>
NormalDir<T>::NormalDir(const Matrix<T,Dynamic,1> &meanPos, const Matrix<T,Dynamic, Dynamic> &covPos,
const Matrix<T,Dynamic,1>& meanDir, T covDir, boost::mt19937 &rndGen) 
:meanPos_(meanPos), covPos_(covPos), meanDir_(meanDir), covDir_(covDir), rndGen_(rndGen)
{
  // std::cout << meanDir << std::endl;
  cov_.setZero(3, 3);
  cov_(seq(0,1), seq(0,1)) = covPos_;
  cov_(2, 2) = covDir_;
  // std::cout << cov_ << std::endl;
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
  // std::cout << x_i_new<< std::endl;
  // std::cout << x_i<< std::endl;
  Matrix<T,Dynamic,1> x_i_dir(2);
  x_i_dir << x_i[dim-1] , x_i[dim];
  // std::cout << x_i_dir << std::endl;
  // std::cout << (rie_log(x_i_dir, muDir_)).norm()<< std::endl;
  // std::cout << x_i_dir << std::endl;
  // std::cout << meanDir_ << std::endl;

  x_i_new(dim-1) = (rie_log(x_i_dir, meanDir_)).norm();
  // std::cout << "here3"<< std::endl;

  // std::cout << x_i_new<< std::endl;
  // std::cout << cov_<< std::endl;
  // std::cout << mean_<< std::endl;

  LLT<Matrix<T,Dynamic,Dynamic>> lltObj(cov_);
  T logProb =  dim_ * log(2*PI);

  if (cov_.rows()==2)
    logProb += log(cov_(0, 0) * cov_(1, 1) - cov_(0, 1) * cov_(1, 0));
  else
    logProb += 2 * log(lltObj.matrixL().determinant());
  logProb += (lltObj.matrixL().solve(x_i_new-mean_)).squaredNorm();

  return -0.5 * logProb;
};


template class NormalDir<double>;