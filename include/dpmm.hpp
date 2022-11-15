

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
  // MatrixXu z_; 

  VectorXi z_; 
  uint16_t N_;
  uint16_t K_;
  vector<Dist_t> components_;
};

// ---------------- impl -----------------------------------------------------
template <class Dist_t> 
void DPMM<Dist_t>::initialize(const MatrixXd& x)
{
  uint32_t K0=1;
  x_ = x;
  VectorXi z(x.rows());
  z.setZero();
  z_ = z;
  // z_.setZero(x_.rows());
  N_ = x_.rows();
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
    { 
      // int x[] = {1, 2 , 3};
      // vector<int> x;
      // x.push_back(1);
      // x.push_back(2);
      // cout << x_(x, all) << endl;
      // cout << z_(x) << endl;
      cout << "Data Number: " << i << endl;

      VectorXd x_i = x_(i, all);
      // uint32_t z_i = z_(i);
      // x_(1, 1) = -1;
      cout << Nk << endl;
      // z_[5] = 33; 
      // cout << z_ << endl;
      vector<int> xk_index;
      for (uint32_t ii = 0; ii<N_; ++ii)
      {
        // cout << ii;
        // cout << z_[ii];
        if (ii!= i && z_[ii] == k) xk_index.push_back(ii); 
      }
      cout << xk_index.size() << endl;

      // cout << xk_index << endl;
      Nk(0) = 1;


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

