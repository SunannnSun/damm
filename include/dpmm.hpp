#pragma once

#include <boost/random/mersenne_twister.hpp>
#include <Eigen/Dense>
#include "gauss.hpp"

using namespace Eigen;
using namespace std;


template <class dist_t>
class Dpmm 
{
  public:
    /*---------------------------------------------------*/
    //-------------Constructor & Desctructor--------------
    /*---------------------------------------------------*/
    Dpmm(const MatrixXd& x, int init_cluster, double alpha, const dist_t& H, const boost::mt19937& rndGen, int base);
    Dpmm(const MatrixXd& x, const VectorXi& z, const vector<int>& indexList, const double alpha, const dist_t& H, boost::mt19937& rndGen);
    Dpmm(){};
    ~Dpmm(){};


    /*---------------------------------------------------*/
    //------------------Parallel Sampling-----------------
    /*---------------------------------------------------*/
    void sampleCoefficientsParameters();
    void sampleLabels();


    /*---------------------------------------------------*/
    //----------------Split/Merge Proposal----------------
    /*---------------------------------------------------*/
    void sampleCoefficientsParameters(const vector<int> &indexList);
    void sampleLabels(const vector<int> &indexList);
    double logProposalRatio(const vector<int> & indexList_i,const vector<int> & indexList_j);
    double logTargetRatio(const vector<int> &indexList_i, const vector<int> &indexList_j);


    /*---------------------------------------------------*/
    //---------------------Utilities---------------------
    /*---------------------------------------------------*/    
    void reorderAssignments();
    void updateIndexLists();
    vector<vector<int>> getIndexLists();
    int getK(){return K_;};
    const VectorXi & getLabels(){return z_;};
    

  private:
    //class constructor(indepedent of data)
    uint32_t dim_;
    double alpha_; 
    dist_t H_; 
    boost::mt19937 rndGen_;

    //class initializer(dependent on data)
    MatrixXd x_;
    VectorXi z_;  
    VectorXd Pi_; 
    uint16_t N_;
    uint16_t K_;

    //sampled parameters
    vector<dist_t> parameters_; 
    vector<Gauss<double>> components_; 

    //spilt/merge proposal
    vector<int> indexList_;

    //log in number of components, joint likelihood every iteration
    vector<VectorXi> logZ_;
    vector<int> logNum_;
    vector<double> logLogLik_; //https://stats.stackexchange.com/questions/398780/understanding-the-log-likelihood-score-in-scikit-learn-gmm

public:
    vector<vector<int>> indexLists_;
};







/*---------------------------------------------------*/
//-------------------Inactive Members-----------------
/*---------------------------------------------------*/   
// MatrixXd x_full_;
// VectorXi index_; 
// Eigen::Matrix<std::int32_t, Eigen::Dynamic, 1> z_;


/*---------------------------------------------------*/
//-------------------Inactive Methods-----------------
/*---------------------------------------------------*/
// vector<vector<vector<int>>> computeSimilarity(int num);
// double KL_div(const MatrixXd& Sigma_p, const MatrixXd& Sigma_q, const MatrixXd& mu_p, const MatrixXd& mu_q);
// void sampleCoefficients();
// void sampleParameters();
// int sampleLabelsCollapsed();
// void sampleLabelsCollapsedParallel();
// int splitProposal(const vector<int> &indexList);
// int mergeProposal(const vector<int> &indexList_i, const vector<int> &indexList_j);
// Dpmm(const MatrixXd& x, int init_cluster, double alpha, const dist_t& H, const boost::mt19937& rndGen);
// Dpmm(const MatrixXd& x, const VectorXi& z, const double alpha, const dist_t& H, boost::mt19937 &rndGen);
// void sampleCoefficients(const uint32_t index_i, const uint32_t index_j);
// void sampleParameters(const uint32_t index_i, const uint32_t index_j);
// Gauss<double> sampleParameters(vector<int> indexList);
// void sampleCoefficientsParameters(const uint32_t index_i, const uint32_t index_j);
// void sampleLabels(const uint32_t index_i, const uint32_t index_j);
// void sampleSplit(uint32_t z_i, uint32_t z_j);
// int splitProposal(vector<int> indexList);
// int mergeProposal(vector<int> indexList_i, vector<int> indexList_j);
// double transitionProb(const uint32_t index_i, const uint32_t index_j);
// double transitionProb(const uint32_t index_i, const uint32_t index_j, VectorXi z_original);
// double posteriorRatio(vector<int> indexList_i, vector<int> indexList_j, vector<int> indexList_ij);