

#include <iostream>
// #include <stdint.h>
// #include <vector>
// #include <Eigen/Dense>
	
#include <Eigen/Eigen>
#include "niw.hpp"
#include "global.hpp"

using namespace Eigen;
using namespace std;


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
  void sampleLabels();


private:
  double alpha_; 
  Dist_t H_; 
  MatrixXd x_;
  VectorXu z_; 
  uint16_t N_;
  uint16_t K_;
  std::vector<Dist_t> components_;
};

// ---------------- impl -----------------------------------------------------
template <class Dist_t> 
void DPMM<Dist_t>::initialize(const MatrixXd& x)
{
  uint32_t K0=1;
  x_ = x;
  z_.setZero(x.rows());
  N_ = z_.size();
  K_ = z_.maxCoeff() + 1;
  for (uint32_t k=0; k<K0; ++k)
    components_.push_back(H_);
    // components_.push_back(Dist_t(H_));
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

template <class Dist_t> 
void DPMM<Dist_t>::sampleLabels()
{
  // cout << x_.size() << endl;
  // cout << x_.rows() << endl;
  // cout << x_.cols() << endl;
  // cout << K_ << endl;
  for(uint32_t i=0; i<N_; ++i)
  {
    // cout << i << endl;
    // compute clustercounts 
    VectorXu Nk(K_);
    Nk.setZero(K_);
    for(uint32_t ii=0; ii<N_; ++ii)
      Nk(z_(ii))++;

    VectorXd pi(K_+1); 
    for (uint32_t k=0; k<K_; ++k)
    { cout << "Data Number: " << i << endl;

      MatrixXd x_i = x_(i, all);
      uint32_t z_i = z_[i];
      z_[i] = -1;


      // cout << x_(i, all) << endl;
      // x_i = x(i, );
      // cout << x_(i, all) << endl;
      // cout << seq(2,5) << endl;
      // pi(k) = log(Nk(k))-log(N_+alpha_) + components_[k].logPosteriorProb(x_, z_, k, i);
      // cout << pi(k) << endl;
      // cout << pi(N_) << endl;
    }
    pi(K_) = log(alpha_)-log(N_+alpha_);
  }
};

