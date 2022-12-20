#pragma once

#include <boost/random/mersenne_twister.hpp>

#include <Eigen/Dense>
#include "niw.hpp"

using namespace Eigen;
using namespace std;


template <class Dist_t>
class DPMM
{
public:
  DPMM(const double alpha, const Dist_t& H, boost::mt19937* pRndGen);
  ~DPMM(){};

  void initialize(const MatrixXd& x, const int init_cluster);
//   void sampleLabels();
//   void reorderAssignments();
  const VectorXi & getLabels(){return z_;};

public:
  //class constructor(indepedent of data)
  double alpha_; 
  Dist_t H_; 
  boost::mt19937* pRndGen_;

  //class initializer(dependent on data)
  MatrixXd x_;
  VectorXi z_; //membership vector
  uint16_t N_;
  uint16_t K_;
  vector<Dist_t> components_;
};