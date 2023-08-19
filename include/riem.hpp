#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

template <typename T>
double unsigned_angle(const Matrix<T,Dynamic, 1>&x, const Matrix<T,Dynamic, 1>&y)
{
  /**
   * This function computes the (unsigned) angle between two points in unit sphere
   *
   * @param x point x 
   * @param y point y
   * 
   */
  
  double dotProduct = x.dot(y);
  double cosAngle = dotProduct / (x.norm() * y.norm());
  double angle = std::acos(std::min(std::max(cosAngle, -1.0), 1.0));

  if (std::isnan(angle)) {
      throw std::runtime_error("NaN angle value");
  }

  return angle; 
}


template <typename T>
Matrix<T,Dynamic, 1> rie_log(const Matrix<T,Dynamic, 1>&x, const Matrix<T,Dynamic, 1>&y)
{   
  /**
   * This function maps a point y to the tangent space defined by x
   *
   * @param x is the point of tangency 
   * @param y point y
   * @param tanDir non-unit length tangent direction at point x 
   * @param v is the point y mapped to tangent space defined by x as the point of tangency; i.e. log_x(y)
   * 
   * @note v gives the corrdinates starting at point x
   * @note when x and y are in opposite direction (rarely happens), log map returns zero; hence we manually add perturbation
   * @note when x and y are equal (tanDir.norm()=0), return tanDir = (0, 0, 0)
   */

  double angle;
  try {
      angle = unsigned_angle(x, y);
  } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      exit(0);
  }
  
  VectorXd tanDir = y - x.dot(y) * x;
  if (tanDir.norm() == 0)
    return tanDir;

  if (angle == M_PI)
    angle = M_PI-0.0001;
  VectorXd v = angle * tanDir / tanDir.norm();

  return v;
}




template <typename T>
Matrix<T,Dynamic, 1> rie_exp(const Matrix<T,Dynamic, 1>&x, const Matrix<T,Dynamic, 1>&v)
{
  /**
   * This function maps a point y to the tangent space defined by x
   *
   * @param x is the point of tangency
   * @param v is the point y mapped to tangent space defined by x as the point of tangency; i.e. log_x(y)
   * @param y is the point of v mapped back to unit sphere
   * 
   */

  Matrix<T,Dynamic, 1> y = x * std::cos(v.norm()) + v / v.norm() * std::sin(v.norm());
  return y;
}


template<typename T>
Matrix<T, Dynamic, 1> karcherMean(const Matrix<T,Dynamic, Dynamic>& xDir_k)
{
  /**
   * This function computes the Fréchet mean in unit sphere
   * 
   * @param xDir_k denotes all the directional vectors belong to k_th group
   * @param xTan is the point of tangency that is randomly initalized, updated, and eventually converged to the mean
   * @param xDir is the directional part of xDir_k
   * @param sumDir is the summation of the logrithmic map of all xDir_k w.r.t. xTan
   * @param meanDir is the point that awaits be mapped back to sphere as the new xTan
   * 
   */

  int dim = xDir_k.cols();
  int num = xDir_k.rows();

  float tolerance = 0.01;

  Matrix<T, Dynamic, 1> xTan = xDir_k(0, all).transpose();
  Matrix<T, Dynamic, 1> xDir(dim);
  Matrix<T, Dynamic, 1> sumDir(dim);
  Matrix<T, Dynamic, 1> meanDir(dim);


  while (1)  { 
    sumDir.setZero();

    for (int i=0; i<num; ++i){
      xDir = xDir_k(i, all).transpose();
      sumDir = sumDir + rie_log(xTan, xDir);
    }

    meanDir = sumDir / num;
    if (meanDir.norm() < tolerance)
      return xTan;

    xTan = rie_exp(xTan, meanDir);
  }
};



template<typename T>
T riemScatter(const Matrix<T,Dynamic, Dynamic>& xDir_k)
{
  return riemScatter(xDir_k, karcherMean(xDir_k));
}


template<typename T>
T riemScatter(const Matrix<T,Dynamic, Dynamic>& xDir_k, const Matrix<T, Dynamic, 1>& mean)
{
  /**
   * This function computes the empirical scatter on the Riemannian manifold
   * 
   * @note karcherScatter is a wrong terminology, rather it is the empirical scatter on the tangent space
   * 
   * @note this is SCATTER and NOT variance, this has not been divided by number of component
   */

  int dim = xDir_k.cols();
  int num = xDir_k.rows();

  T scatter = 0;
  for (int i = 0; i < num; ++i) {
    Matrix<T, Dynamic, 1> xDir_i = xDir_k(i, all).transpose();
    scatter = scatter + pow(rie_log(mean, xDir_i).norm(), 2); 
  }
  return scatter;
}