

#include <iostream>
// #include <stdint.h>
// #include <vector>
#include <Eigen/Dense>
#include "niw.hpp"

using namespace Eigen;
using namespace std;
typedef Matrix<uint32_t, Dynamic,1> VectorXu;

/*
 * DP mixture model
 * following Neal [[http://www.stat.purdue.edu/~rdutta/24.PDF]]
 * Algo 3
 */



template <class Dist_t>
class DPMM
{
public:
  DPMM(double alpha, const Dist_t& H): alpha_(alpha), H_(H) {};
  ~DPMM(){};

  void initialize(const MatrixXd& x);

private:

  double alpha_; 
  Dist_t H_; 
  MatrixXd x_;
  VectorXu z_; 
  std::vector<Dist_t> components_;
};

// ---------------- impl -----------------------------------------------------
template <class Dist_t> 
void DPMM<Dist_t>::initialize(const MatrixXd& x)
{
  uint32_t K0=1;
  x_ = x;
  z_.setZero(x.rows(),1);
  for (uint32_t k=0; k<K0; ++k)
    // components_.push_back(H_);
    components_.push_back(Dist_t(H_));
  cout<<components_.size()<<endl;
  //cout<<x_<<endl;
  cout<<alpha_<<endl;
  //random number of initial components; and sample each assignment from a symmetric categorical distribution
  // if (K0>1)
  // {
  //   VectorXd pi(K0);
  //   pi.setOnes();
  //   pi /= static_cast<double>(K0);
  //   Catd cat(pi,H_.pRndGen_);
  //   for (uint32_t i=0; i<z_.size(); ++i)
  //     z_(i) = cat.sample();
  // }
};
