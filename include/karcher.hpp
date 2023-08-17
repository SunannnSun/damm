#include <iostream>
// #include <cmath>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

#define PI 3.141592653589793


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
      // std::cout << x << std::endl << y  << std::endl << angle << std::endl;
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
   * when x and y are in opposite direction, log map returns zero; hence we manually add perturbation
   * 
   */

  double angle;
  try {
      angle = unsigned_angle(x, y);
  } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      exit(0);
  }
  
  if (angle == M_PI)
    angle = M_PI-0.0001;
  VectorXd tanDir = y - x.dot(y) * x;

  if (tanDir.norm() == 0)
    return tanDir;

  VectorXd v = angle * tanDir / tanDir.norm();

  return v;
}




template <typename T>
Matrix<T,Dynamic, 1> rie_exp(const Matrix<T,Dynamic, 1>&x, Matrix<T,Dynamic, 1>&v)
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
Matrix<T, Dynamic, 1> karcherMean(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  /**
   * This function maps a point y to the tangent space defined by x
   *
   * @param xTan is the point of tangency that is initalized randomly, updated, and eventually converged to Karcher mean
   * @param xDir is the directional part of x_k
   * @param sumDir is the summation of the logrithmic map of all x_k w.r.t. xTan
   * @param meanDir is the point that awaits be mapped back to sphere as the new xTan
   * 
   */


  // std::cout << "HIIIIIIIIIIIIIIIIIIII" << x_k.rows() << std::endl;

  int dim = x_k.cols()/2;
  int num = x_k.rows();

  float tolerance = 0.01;

  // T angle;
  Matrix<T, Dynamic, 1> xTan = x_k(0, seq(dim, last)).transpose();
  Matrix<T, Dynamic, 1> xDir(dim);
  Matrix<T, Dynamic, 1> sumDir(dim);
  Matrix<T, Dynamic, 1> meanDir(dim);


  while (1)  { 
    sumDir.setZero();
    // std::cout <<sumDir << std::endl;

    for (int i=0; i<num; ++i){
      xDir = x_k(i, seq(dim, last)).transpose();
      // std::cout <<sumDir << std::endl;
      // std::cout << rie_log(xTan, xDir) << "sumDIR" << std::endl;

      sumDir = sumDir + rie_log(xTan, xDir);
      // std::cout <<sumDir << std::endl;

      // std::cout << sumDir + rie_log(xTan, xDir) << "sumDIR" << std::endl;
    }


    meanDir = sumDir / num;

    if (meanDir.norm() < tolerance)
      return xTan;


    xTan = rie_exp(xTan, meanDir);

  }
};



template<typename T>
T karcherScatter(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  return karcherScatter(x_k, karcherMean(x_k));
}


template<typename T>
T karcherScatter(const Matrix<T,Dynamic, Dynamic>& x_k, Matrix<T, Dynamic, 1> mean)
{
  int dim = x_k.cols()/2;
  int num = x_k.rows();
  T scatter = 0;
  Matrix<T, Dynamic, 1> x_i_dir(dim);

  for (int i = 0; i < num; ++i) {
    x_i_dir = x_k(i, seq(dim, last)).transpose();
    scatter = scatter + pow(rie_log(mean, x_i_dir).norm(), 2); //squared
  }
  return scatter;
}