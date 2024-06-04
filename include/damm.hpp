#pragma once

#include <boost/random/mersenne_twister.hpp>
#include <Eigen/Dense>
#include "gaussDamm.hpp"

using namespace Eigen;
using namespace std;


template <class dist_t>
class Damm
{
  public:
    /*---------------------------------------------------*/
    //-------------Constructor & Desctructor--------------
    /*---------------------------------------------------*/
    Damm(const MatrixXd &x, int init_cluster, double alpha, const dist_t &H, const boost::mt19937 &rndGen);
    Damm(const MatrixXd &x, int init_cluster, double alpha, const dist_t &H, const boost::mt19937 &rndGen, VectorXi z);
    Damm(){};
    ~Damm(){};


    /*---------------------------------------------------*/
    //------------------Parallel Sampling-----------------
    /*---------------------------------------------------*/
    void sampleCoefficientsParameters();
    void sampleLabels();


    /*---------------------------------------------------*/
    //----------------Split/Merge Proposal----------------
    /*---------------------------------------------------*/
    int splitProposal(const vector<int> &indexList);
    int mergeProposal(const vector<int> &indexList_i, const vector<int> &indexList_j);

    /*---------------------------------------------------*/
    //----------------Incremental Learning----------------
    /*---------------------------------------------------*/
    void sampleLabels_increm();

    /*---------------------------------------------------*/
    //---------------------Utilities---------------------
    /*---------------------------------------------------*/  
    void reorderAssignments();
    void updateIndexLists();
    vector<vector<int>> getIndexLists();
    const VectorXi & getLabels(){return z_;};
    int getK(){return K_;};
    vector<array<int, 2>>  computeSimilarity(int mergeNum, int mergeIdx);

  private:
    double KL_div(const MatrixXd& Sigma_p, const MatrixXd& Sigma_q, const MatrixXd& mu_p, const MatrixXd& mu_q);


  private:
    //class constructor(indepedent of data)
    uint32_t dim_;
    double alpha_; 
    dist_t H_; 
    boost::mt19937 rndGen_;

    //class initializer(dependent on data)
    MatrixXd x_;
    MatrixXd xPos_;
    MatrixXd xDir_;
    VectorXi z_;   
    VectorXd Pi_;  
    uint16_t N_;
    uint16_t K_;

    //sampled parameters
    vector<dist_t> parameters_ ;     
    vector<gaussDamm<double>> components_;      

    vector<vector<int>> indexLists_;


    //log in number of components, joint likelihood every iteration
    vector<VectorXi> logZ_;
    vector<int> logNum_;
    vector<double> logLogLik_; //https://stats.stackexchange.com/questions/398780/understanding-the-log-likelihood-score-in-scikit-learn-gmm



    // incremental Learning
    vector<int> indexList_new_;
};





/*---------------------------------------------------*/
//-------------------Inactive Members-----------------
/*---------------------------------------------------*/   
// vector<int> indexList_;
// VectorXi index_; 

/*---------------------------------------------------*/
//-------------------Inactive Methods-----------------
/*---------------------------------------------------*/
// Damm(const MatrixXd& x, const VectorXi& z, const vector<int> indexList, const double alpha, const dist_t& H, boost::mt19937& rndGen);
// void sampleCoefficients();
// void sampleParameters();
// void sampleCoefficientsParameters(vector<int> indexList);
// void sampleLabels(vector<int> indexList);
// double logProposalRatio(vector<int> indexList_i, vector<int> indexList_j);
// double logTargetRatio(vector<int> indexList_i, vector<int> indexList_j);

// void sampleCoefficientsParameters(vector<int> indexList);
