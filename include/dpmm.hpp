#pragma once

#include <boost/random/mersenne_twister.hpp>
#include <Eigen/Dense>

#include "niw.hpp"
#include "normal.hpp"

using namespace Eigen;
using namespace std;


template <class Dist_t>
class DPMM
{
public:
  DPMM(const MatrixXd& x, const int init_cluster, const double alpha, const Dist_t& H, boost::mt19937* pRndGen);
  DPMM(const MatrixXd& x, const VectorXi& z, const vector<int> indexList, const double alpha, const Dist_t& H, boost::mt19937* pRndGen);
  ~DPMM(){};

  void splitProposal();
  // void mergeProposal();
  void sampleCoefficients();
  void sampleCoefficients(const uint32_t index_i, const uint32_t index_j);
  void sampleParameters();
  void sampleParameters(const uint32_t index_i, const uint32_t index_j);
  void sampleCoefficientsParameters(const uint32_t index_i, const uint32_t index_j);
  void sampleLabels();
  void sampleLabels(const uint32_t index_i, const uint32_t index_j);

  double transitionProb(const uint32_t index_i, const uint32_t index_j);
  double posteriorRatio(const uint32_t index_i, const uint32_t index_j, const double prevPi, Normal<double>& prevParameter);
  void reorderAssignments();
  const VectorXi & getLabels(){return z_;};

  void splitProposal(const uint32_t index_i, const uint32_t index_j);

public:
  //class constructor(indepedent of data)
  double alpha_; 
  Dist_t H_; 
  boost::mt19937* pRndGen_;

  //class initializer(dependent on data)
  MatrixXd x_;
  VectorXi z_;  //membership vector
  VectorXd Pi_; //coefficient vector
  VectorXi index_; //index vector
  uint16_t N_;
  uint16_t K_;

  //sampled parameters
  vector<Dist_t> components_; //
  vector<Normal<double>> parameters_; //parameter vector

  //spilt/merge proposal
  vector<int> indexList_;
};