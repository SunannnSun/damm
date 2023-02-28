#include <iostream>
#include <limits>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include "dpmmDir.hpp"
#include "niwDir.hpp"


template <class dist_t> 
DPMMDIR<dist_t>::DPMMDIR(const MatrixXd& x, int init_cluster, double alpha, const dist_t& H, const boost::mt19937 &rndGen)
: alpha_(alpha), H_(H), rndGen_(rndGen), x_(x), N_(x.rows())
{
  VectorXi z(x.rows());
  if (init_cluster == 1) 
  {
    z.setZero();
  }
  else if (init_cluster >= 1)
  {
    boost::random::uniform_int_distribution<> uni_(0, init_cluster-1);
    for (int i=0; i<N_; ++i)z[i] = uni_(rndGen_); 
  }
  else
  { 
    cout<< "Number of initial clusters not supported yet" << endl;
    exit(1);
  }
  z_ = z;
  K_ = z_.maxCoeff() + 1; // equivalent to the number of initial clusters
};



template <class dist_t> 
DPMMDIR<dist_t>::DPMMDIR(const MatrixXd& x, const VectorXi& z, const vector<int> indexList, const double alpha, const dist_t& H, boost::mt19937 &rndGen)
: alpha_(alpha), H_(H), rndGen_(rndGen), x_(x), N_(x.rows()), z_(z), K_(z.maxCoeff() + 1), indexList_(indexList)
{
  vector<int> indexList_i;
  vector<int> indexList_j;

  boost::random::uniform_int_distribution<> uni_01(0, 1);
  for (int i = 0; i<indexList_.size(); ++i)
  {
    if (uni_01(rndGen_) == 0) indexList_i.push_back(indexList_[i]);
    else indexList_j.push_back(indexList_[i]);
  }
  indexLists_.push_back(indexList_i);
  indexLists_.push_back(indexList_j);
};


template <class dist_t> 
void DPMMDIR<dist_t>::sampleCoefficients()
{
  VectorXi Nk(K_);
  Nk.setZero();
  for(uint32_t ii=0; ii<N_; ++ii)
  {
    Nk(z_(ii))++;
  }

  VectorXd Pi(K_);
  for (uint32_t k=0; k<K_; ++k)
  {
    assert(Nk(k)!=0);
    boost::random::gamma_distribution<> gamma_(Nk(k), 1);
    Pi(k) = gamma_(rndGen_);
  }
  Pi_ = Pi / Pi.sum();
}


template <class dist_t> 
void DPMMDIR<dist_t>::sampleParameters()
{ 
  components_.clear();
  parameters_.clear();

  for (uint32_t k=0; k<K_; ++k)
  {
    vector<int> indexList_k;
    for (uint32_t ii = 0; ii<N_; ++ii)
    {
      if (z_[ii] == k) indexList_k.push_back(ii); 
    }
    MatrixXd x_k(indexList_k.size(), x_.cols()); 
    x_k = x_(indexList_k, all);

    components_.push_back(H_.posterior(x_k));  //components are NIW
    parameters_.push_back(components_[k].sampleParameter()); //parameters are Normal
  }
}


template <class dist_t> 
void DPMMDIR<dist_t>::sampleCoefficientsParameters()
{ 
  components_.clear();
  parameters_.clear();
  VectorXd Pi(K_);

  vector<vector<int>> indexLists(K_);
  for (uint32_t ii = 0; ii<N_; ++ii)
  {
    indexLists[z_[ii]].push_back(ii); 
  }
  
  for (uint32_t k=0; k<K_; ++k)
  {
    boost::random::gamma_distribution<> gamma_(indexLists[k].size(), 1);
    Pi(k) = gamma_(rndGen_);
    components_.push_back(H_.posterior(x_(indexLists[k], all)));
    parameters_.push_back(components_[k].sampleParameter());
  }
  Pi_ = Pi / Pi.sum();
}



template <class dist_t> 
void DPMMDIR<dist_t>::sampleLabels()
{
  #pragma omp parallel for num_threads(4) schedule(static) private(rndGen_)
  for(uint32_t i=0; i<N_; ++i)
  {
    VectorXd x_i;
    x_i = x_(i, all); //current data point x_i
    VectorXd prob(K_);
    for (uint32_t k=0; k<K_; ++k)
    {
      prob[k] = log(Pi_[k]) + parameters_[k].logProb(x_i);
    }

    double prob_max = prob.maxCoeff();
    prob = (prob.array()-(prob_max + log((prob.array() - prob_max).exp().sum()))).exp().matrix();
    prob = prob / prob.sum();
    for (uint32_t ii = 1; ii < prob.size(); ++ii){
      prob[ii] = prob[ii-1]+ prob[ii];
    }
    boost::random::uniform_01<> uni_;   
    double uni_draw = uni_(rndGen_);
    uint32_t k = 0;
    while (prob[k] < uni_draw) k++;
    z_[i] = k;
  }
}


template <class dist_t>
void DPMMDIR<dist_t>::reorderAssignments()
{ 
  vector<uint8_t> rearrange_list;
  for (uint32_t i=0; i<N_; ++i)
  {
    if (rearrange_list.empty()) rearrange_list.push_back(z_[i]);
    vector<uint8_t>::iterator it;
    it = find (rearrange_list.begin(), rearrange_list.end(), z_[i]);
    if (it == rearrange_list.end())
    {
      rearrange_list.push_back(z_[i]);
      z_[i] = rearrange_list.size() - 1;
    }
    else if (it != rearrange_list.end())
    {
      int index = it - rearrange_list.begin();
      z_[i] = index;
    }
  }
  K_ = z_.maxCoeff() + 1;
}


template <class dist_t>
vector<vector<int>> DPMMDIR<dist_t>::getIndexLists()
{
  this ->updateIndexLists();
  return indexLists_;
}

template <class dist_t>
void DPMMDIR<dist_t>::updateIndexLists()
{
  vector<vector<int>> indexLists(K_);
  for (uint32_t ii = 0; ii<N_; ++ii)
  {
    indexLists[z_[ii]].push_back(ii); 
  }
  indexLists_ = indexLists;
}


template <class dist_t> 
int DPMMDIR<dist_t>::splitProposal(vector<int> indexList)
{
  VectorXi z_launch = z_;
  VectorXi z_split = z_;
  uint32_t z_split_i = z_split.maxCoeff() + 1;
  uint32_t z_split_j = z_split[indexList[0]];

  DPMMDIR<dist_t> dpmm_split(x_, z_launch, indexList, alpha_, H_, rndGen_);
  for (int tt=0; tt<100; ++tt)
  {
    if (dpmm_split.indexLists_[0].size()==0 || dpmm_split.indexLists_[1].size() ==0) return 1;
    dpmm_split.sampleCoefficientsParameters(indexList);
    dpmm_split.sampleLabels(indexList);
  }

  vector<int> indexList_i = dpmm_split.indexLists_[0];
  vector<int> indexList_j = dpmm_split.indexLists_[1];


  for (int i = 0; i < indexList_i.size(); ++i)
  {
    z_split[indexList_i[i]] = z_split_i;
  }
  for (int i = 0; i < indexList_j.size(); ++i)
  {
    z_split[indexList_j[i]] = z_split_j;
  }
  z_ = z_split;
  K_ += 1;
  this -> updateIndexLists();    
  std::cout << "Component " << z_split_j <<": Split proposal Aceepted" << std::endl;
  return 0;
}


template <class dist_t> 
int DPMMDIR<dist_t>::mergeProposal(vector<int> indexList_i, vector<int> indexList_j)
{
  VectorXi z_launch = z_;
  VectorXi z_merge = z_;
  uint32_t z_merge_i = z_merge[indexList_i[0]];
  uint32_t z_merge_j = z_merge[indexList_j[0]];

  vector<int> indexList;
  indexList.reserve(indexList_i.size() + indexList_j.size() ); // preallocate memory
  indexList.insert( indexList.end(), indexList_i.begin(), indexList_i.end() );
  indexList.insert( indexList.end(), indexList_j.begin(), indexList_j.end() );


  DPMMDIR<dist_t> dpmm_merge(x_, z_launch, indexList, alpha_, H_, rndGen_);
  for (int tt=0; tt<100; ++tt)
  {    
    if (dpmm_merge.indexLists_[0].size()==0 || dpmm_merge.indexLists_[1].size() ==0)
    {
      for (int i = 0; i < indexList_i.size(); ++i) z_merge[indexList_i[i]] = z_merge_j;
      z_ = z_merge;
      this -> reorderAssignments();
      std::cout << "Component " << z_merge_j << "and" << z_merge_i <<": Merge proposal Aceepted" << std::endl;
      return 0;
    };
    dpmm_merge.sampleCoefficientsParameters(indexList);
    dpmm_merge.sampleLabels(indexList);
  }
  std::cout << "Component " << z_merge_j << "and" << z_merge_i <<": Merge proposal Rejected" << std::endl;
  return 1;
}



template <class dist_t> 
void DPMMDIR<dist_t>::sampleCoefficientsParameters(vector<int> indexList)
{
  vector<int> indexList_i = indexLists_[0];
  vector<int> indexList_j = indexLists_[1];

  MatrixXd x_i(indexList_i.size(), x_.cols()); 
  MatrixXd x_j(indexList_j.size(), x_.cols()); 
  x_i = x_(indexList_i, all);
  x_j = x_(indexList_j, all);
  
  components_.clear();
  parameters_.clear();
  components_.push_back(H_.posterior(x_i));
  components_.push_back(H_.posterior(x_j));
  parameters_.push_back(components_[0].sampleParameter());
  parameters_.push_back(components_[1].sampleParameter());
  

  VectorXi Nk(2);
  Nk(0) = indexList_i.size();
  Nk(1) = indexList_j.size();


  VectorXd Pi(2);
  for (uint32_t k=0; k<2; ++k)
  {
    boost::random::gamma_distribution<> gamma_(Nk(k), 1);
    Pi(k) = gamma_(rndGen_);
  }
  Pi_ = Pi / Pi.sum();
}


template <class dist_t> 
void DPMMDIR<dist_t>::sampleLabels(vector<int> indexList)
{
  vector<int> indexList_i;
  vector<int> indexList_j;

  boost::random::uniform_01<> uni_;    
  // #pragma omp parallel for num_threads(4) schedule(static) private(rndGen_)
  for(uint32_t i=0; i<indexList.size(); ++i)
  {
    VectorXd x_i;
    x_i = x_(indexList[i], all); //current data point x_i from the index_list
    VectorXd prob(2);
    for (uint32_t k=0; k<2; ++k)
    {
      prob[k] = log(Pi_[k]) + parameters_[k].logProb(x_i); //first component is always the set of x_i (different notion from x_i here)
    }

    double prob_max = prob.maxCoeff();
    prob = (prob.array()-(prob_max + log((prob.array() - prob_max).exp().sum()))).exp().matrix();
    prob = prob / prob.sum();
    for (uint32_t ii = 1; ii < prob.size(); ++ii){
      prob[ii] = prob[ii-1]+ prob[ii];
    }
    double uni_draw = uni_(rndGen_);
    if (uni_draw < prob[0]) indexList_i.push_back(indexList_[i]);
    else indexList_j.push_back(indexList_[i]);
  }

  // std::cout << indexList_i.size() << std::endl;
  // std::cout << indexList_j.size() << std::endl;

  indexLists_.clear();
  indexLists_.push_back(indexList_i);
  indexLists_.push_back(indexList_j);
}


template class DPMMDIR<NIWDIR<double>>;

