#pragma once

#include <boost/random/mersenne_twister.hpp>
#include <Eigen/Dense>
#include "normal.hpp"

using namespace Eigen;
using namespace std;


template <class dist_t>
class DPMM {
  public:
    /*---------------------------------------------------*/
    //-------------Constructor & Desctructor--------------
    /*---------------------------------------------------*/
    DPMM(const MatrixXd& x, int init_cluster, double alpha, const dist_t& H, const boost::mt19937& rndGen);
    DPMM(const MatrixXd& x, const VectorXi& z, const vector<int> indexList, const double alpha, const dist_t& H, boost::mt19937& rndGen);
    ~DPMM(){};



    /*---------------------------------------------------*/
    //------------------Parallel Sampling-----------------
    /*---------------------------------------------------*/
    void sampleCoefficients();
    void sampleParameters();
    void sampleCoefficientsParameters();
    void sampleLabels();



    /*---------------------------------------------------*/
    //----------------Split/Merge Proposal----------------
    /*---------------------------------------------------*/
    void sampleCoefficientsParameters(const vector<int> &indexList);
    void sampleLabels(const vector<int> &indexList);

    double logPosteriorProb(const vector<int> &indexList_i, const vector<int> &indexList_j);
    double logTransitionProb(const vector<int> &indexList_i, const vector<int> &indexList_j);

    // int splitProposal(const vector<int> &indexList);
    // int mergeProposal(const vector<int> &indexList_i, const vector<int> &indexList_j);

    /*---------------------------------------------------*/
    //---------------------Utilities---------------------
    /*---------------------------------------------------*/    
    void reorderAssignments();
    void updateIndexLists();
    vector<vector<int>> getIndexLists();
    const VectorXi & getLabels(){return z_;};
    vector<vector<int>> computeSimilarity();

    /*---------------------------------------------------*/
    //-------------------Inactive Methods-----------------
    /*---------------------------------------------------*/
    // void sampleCoefficients(const uint32_t index_i, const uint32_t index_j);
    // void sampleParameters(const uint32_t index_i, const uint32_t index_j);
    // Normal<double> sampleParameters(vector<int> indexList);
    // void sampleCoefficientsParameters(const uint32_t index_i, const uint32_t index_j);
    // void sampleLabels(const uint32_t index_i, const uint32_t index_j);
    // void sampleSplit(uint32_t z_i, uint32_t z_j);
    // int splitProposal(vector<int> indexList);
    // int mergeProposal(vector<int> indexList_i, vector<int> indexList_j);
    // double transitionProb(const uint32_t index_i, const uint32_t index_j);
    // double transitionProb(const uint32_t index_i, const uint32_t index_j, VectorXi z_original);
    // double posteriorRatio(vector<int> indexList_i, vector<int> indexList_j, vector<int> indexList_ij);
    // void sampleLabelsCollapsed(const vector<int> &indexList);




  public:
    //class constructor(indepedent of data)
    double alpha_; 
    dist_t H_; 
    boost::mt19937 rndGen_;

    //class initializer(dependent on data)
    MatrixXd x_;
    MatrixXd x_full_;
    VectorXi z_;  
    VectorXd Pi_; 
    VectorXi index_; 
    uint16_t N_;
    uint16_t K_;

    //sampled parameters
    vector<dist_t> parameters_; // NIW
    vector<Normal<double>> components_; //Normal

    //spilt/merge proposal
    vector<int> indexList_;
    vector<vector<int>> indexLists_;

    //log in number of components, joint likelihood every iteration
    vector<int> logNum_;
    vector<double> logLogLik_; //https://stats.stackexchange.com/questions/398780/understanding-the-log-likelihood-score-in-scikit-learn-gmm
};