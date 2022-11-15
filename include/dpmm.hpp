

#include <iostream>
// #include <stdint.h>
// #include <vector>
// #include <Eigen/Dense>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>

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
    VectorXd x_i;
    x_i = x_(i, all);
    for (uint32_t k=0; k<K_; ++k)
    { 
      // int x[] = {1, 2 , 3};
      // vector<int> x;
      // x.push_back(1);
      // x.push_back(2);
      // cout << x_(x, all) << endl;
      // cout << z_(x) << endl;
      cout << "Data Number: " << i << endl;


      // uint32_t z_i = z_(i);
      // x_(1, 1) = -1;
      // cout << Nk << endl;
      // z_[5] = 33; 
      // cout << z_ << endl;
      vector<int> x_k_index;
      for (uint32_t ii = 0; ii<N_; ++ii)
      {
        // cout << ii;
        // cout << z_[ii];
        if (ii!= i && z_[ii] == k) x_k_index.push_back(ii); 
      }
      // cout << x_k_index.size() << endl;
      MatrixXd x_k(x_k_index.size(), x_.cols()); 
      x_k = x_(x_k_index, all);
      // cout << xk_index << endl;
      // Nk(0) = 1;


      // cout << x_(i, all) << endl;
      // x_i = x(i, );
      // cout << x_(i, all) << endl;
      // cout << seq(2,5) << endl;
      // pi(k) = log(Nk(k))-log(N_+alpha_) + components_[k].logPosteriorProb(x_i, x_k);
      pi(k) = log(Nk(k))-log(N_+alpha_) + components_[k].logPosteriorProb(x_i, x_k);

      // cout << pi(k) << endl;
      // cout << x_k.rowwise() - x_k.colwise().mean() << endl;

      // cout << x_k.colwise().mean() << endl;
      // cout << pi(N_) << endl;
    }
    // cout << H_.nu_- H_.dim_+1 << endl;
    pi(K_) = log(alpha_)-log(N_+alpha_) + H_.logProb(x_i);
    // cout << pi << endl;
    // normalize pi and exponentiate it; redo it later to comply with new eigen library
    // https://dev.to/seanpgallivan/solution-running-sum-of-1d-array-34na#c-code
    double pi_max = pi.maxCoeff();
    // cout << pi.size() <<endl;
    // pi = (pi.array()-(pi_max + log((pi.array() - pi_max).exp().sum()))).exp().matrix();
    pi = pi / pi.sum();
    for (uint32_t i = 1; i < pi.size(); ++i){
      pi[i] = pi[i-1]+ pi[i];
    }
    cout << pi[0] << endl;
    cout << pi[1] << endl;
  


    // Generate a uniform number from [0, 1]
    boost::random::uniform_01<> uni_;
    // boost::random::mt19937 gen = ;
    // https://www.boost.org/doc/libs/1_80_0/doc/html/boost_random/tutorial.html
    // note: Distinguish boost::math::uniform vs. boost::random::uniform
    
    boost::random::variate_generator<boost::random::mt19937&, 
                           boost::random::uniform_01<> > var_nor(*H_.pRndGen_, uni_);
    double uni_draw = var_nor();
    cout << uni_draw << endl;
    uint32_t k = 0;
    // uni_draw = 0.999;
    // cout << (pi[0] < uni_draw) << endl;
    while (pi[k] < uni_draw) k++;
    cout << k << endl;
    // while (k < pi.size()){
    //   cout << pi[k] << endl;
    //   k++;
    // }
    // for (int k = 0; k< pi.size(); k++)
    // {
    //   cout << k << endl;;
    // }
    // z_[i] = k;

    // cout << uni_(gen) << endl;
  }
};

