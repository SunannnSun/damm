#include "gaussDamm.hpp"
#include "riem.hpp"


#define PI 3.141592653589793


template<class T>
gaussDamm<T>::gaussDamm(const Matrix<T,Dynamic,1> &meanPos, const Matrix<T,Dynamic, Dynamic> &covPos,
const Matrix<T,Dynamic,1> &meanDir, T covDir, boost::mt19937 &rndGen) 
:meanPos_(meanPos), covPos_(covPos), meanDir_(meanDir), covDir_(covDir), rndGen_(rndGen)
{
  dim_ = meanPos.rows();
  meanHat_.setZero(dim_+1);
  meanHat_(seq(0,dim_-1)) = meanPos_;

  covHat_.setZero(dim_+1, dim_+1);
  covHat_(seq(0,dim_-1), seq(0,dim_-1)) = covPos_;
  covHat_(dim_, dim_) = covDir_;
};



template<class T>
T gaussDamm<T>::logProb(const Matrix<T,Dynamic,1> &x_i)
{ 
  Matrix<T,Dynamic,1> xDir_i = x_i(seq(dim_, last));
  Matrix<T,Dynamic,1> xHat(dim_+1);
  xHat(seq(0, dim_-1)) = x_i(seq(0, dim_-1));
  xHat(dim_) = (rie_log(meanDir_, xDir_i)).norm();
  
  // xHat(dim_) = unsigned_angle(xDir, meanDir_);
  // xHat(dim_) = 0;

  LLT<Matrix<T,Dynamic,Dynamic>> lltObj(covHat_);
  T logProb =  dim_ * log(2*PI);
  logProb += 2 * log(lltObj.matrixL().determinant());
  logProb += (lltObj.matrixL().solve(xHat-meanHat_)).squaredNorm();
  return -0.5 * logProb;
}



  // if (x_i.rows()==2) //3D data only pos 
  // {
  //   int dim = 2;
  //   Matrix<T,Dynamic,Dynamic> cov(2, 2);
  //   cov = cov_(seq(0, 1), seq(0, 1));
  //   Matrix<T,Dynamic,1> mean(2);
  //   mean = mean_(seq(0, 1));
  
  //   LLT<Matrix<T,Dynamic,Dynamic>> lltObj(cov);
  //   T logProb =  dim * log(2*PI);
  //   logProb += 2 * log(lltObj.matrixL().determinant());
  //   logProb += (lltObj.matrixL().solve(x_i-mean)).squaredNorm();
  //   return -0.5 * logProb;
  // }
  // else if (x_i.rows()==4) //2D data full pos and dir
  // {
  //   int dim = 3;
  //   Matrix<T,Dynamic,1> x_i_new(dim);
  //   x_i_new.setZero();
  //   x_i_new(seq(0, dim-2)) = x_i(seq(0, dim-2));
  //   Matrix<T,Dynamic,1> x_i_dir(2);
  //   x_i_dir << x_i[dim-1] , x_i[dim];
  //   x_i_new(dim-1) = (rie_log(meanDir_, x_i_dir)).norm();
  
  //   LLT<Matrix<T,Dynamic,Dynamic>> lltObj(cov_);
  //   T logProb =  dim * log(2*PI);
  //   logProb += 2 * log(lltObj.matrixL().determinant());
  //   logProb += (lltObj.matrixL().solve(x_i_new-mean_)).squaredNorm();
  //   return -0.5 * logProb;
  // }
  // else if (x_i.rows()==6)  //3D data full pos and dir
  // {
  //   int dim = 4;
  //   Matrix<T,Dynamic,1> x_i_new(4);
  //   x_i_new.setZero();
  //   x_i_new(seq(0, 2)) = x_i(seq(0, 2));
  //   Matrix<T,Dynamic,1> x_i_dir(3);
  //   x_i_dir << x_i[3] , x_i[4], x_i[5];
  //   x_i_new(3) = (rie_log(meanDir_, x_i_dir)).norm();
  
  //   LLT<Matrix<T,Dynamic,Dynamic>> lltObj(cov_);
  //   T logProb =  dim * log(2*PI);
  //   logProb += 2 * log(lltObj.matrixL().determinant());
  //   logProb += (lltObj.matrixL().solve(x_i_new-mean_)).squaredNorm();
  //   return -0.5 * logProb;
  // }
  // else return 0;
  // return -0.5 * logProb;
// };


template class gaussDamm<double>;