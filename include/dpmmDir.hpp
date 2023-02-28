#pragma once

#include <boost/random/mersenne_twister.hpp>
#include <Eigen/Dense>
#include "normalDir.hpp"

using namespace Eigen;
using namespace std;


template <class dist_t>
class DPMMDIR
{
public:
  DPMMDIR(const MatrixXd& x, int init_cluster, double alpha, const dist_t& H, const boost::mt19937& rndGen);
  DPMMDIR(const MatrixXd& x, const VectorXi& z, const vector<int> indexList, const double alpha, const dist_t& H, boost::mt19937& rndGen);
  ~DPMMDIR(){};


  void sampleCoefficients();
  void sampleParameters();
  void sampleCoefficientsParameters();
  void sampleLabels();


  void reorderAssignments();
  void updateIndexLists();
  vector<vector<int>> getIndexLists();
  const VectorXi & getLabels(){return z_;};

  //split proposal
  int splitProposal(vector<int> indexList);
  void sampleCoefficientsParameters(vector<int> indexList);
  void sampleLabels(vector<int> indexList);

public:
  //class constructor(indepedent of data)
  double alpha_; 
  dist_t H_; 
  boost::mt19937 rndGen_;

  //class initializer(dependent on data)
  MatrixXd x_;
  VectorXi z_;  //membership vector
  VectorXd Pi_; //coefficient vector
  VectorXi index_; //index vector
  uint16_t N_;
  uint16_t K_;

  //sampled parameters
  vector<dist_t> components_; // NIW
  vector<NormalDir<double>> parameters_; //Normal

  //spilt/merge proposal
  vector<int> indexList_;
  vector<vector<int>> indexLists_;
};