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
    // constructor & desctructor
    DPMMDIR(const MatrixXd &x, int init_cluster, double alpha, const dist_t &H, const boost::mt19937 &rndGen);
    DPMMDIR(const MatrixXd& x, const VectorXi& z, const vector<int> indexList, const double alpha, const dist_t& H, boost::mt19937& rndGen);
    ~DPMMDIR(){};

    // parallel sampling
    void sampleCoefficients();
    void sampleParameters();
    void sampleCoefficientsParameters();
    void sampleLabels();


    //split/merge proposal
    int splitProposal(const vector<int> &indexList);
    int mergeProposal(const vector<int> &indexList_i, const vector<int> &indexList_j);

    
    void sampleCoefficientsParameters(vector<int> indexList);
    void sampleLabels(vector<int> indexList);
    double logProposalRatio(vector<int> indexList_i, vector<int> indexList_j);
    double logTargetRatio(vector<int> indexList_i, vector<int> indexList_j);


    // utilies
    void reorderAssignments();
    void updateIndexLists();
    vector<vector<int>> getIndexLists();
    const VectorXi & getLabels(){return z_;};
    vector<vector<vector<int>>> computeSimilarity(int num);

    private:
      double KL_div(const MatrixXd& Sigma_p, const MatrixXd& Sigma_q, const MatrixXd& mu_p, const MatrixXd& mu_q);
    /*---------------------------------------------------*/
    //Following class methods are currently not being used 
    /*---------------------------------------------------*/
    // void sampleCoefficientsParameters(vector<int> indexList);



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
    vector<dist_t>            parameters_ ;     // Normal Inverse-Wishart 
    vector<NormalDir<double>> components_;      // Multi-variate Normal

    //spilt/merge proposal
    vector<int> indexList_;
    vector<vector<int>> indexLists_;


    //log in number of components, joint likelihood every iteration
    vector<int> logNum_;
    vector<double> logLogLik_; //https://stats.stackexchange.com/questions/398780/understanding-the-log-likelihood-score-in-scikit-learn-gmm
};