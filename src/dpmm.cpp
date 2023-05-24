#include <iostream>
#include <limits>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include "dpmmDir.hpp"
#include "dpmm.hpp"
#include "niw.hpp"
#include "niwDir.hpp"


template <class dist_t> 
DPMM<dist_t>::DPMM(const MatrixXd& x, int init_cluster, double alpha, const dist_t& H, const boost::mt19937 &rndGen)
: alpha_(alpha), H_(H), rndGen_(rndGen), N_(x.rows())
{
  // Store both the full Data and only Pos Data
  x_full_ = x;
  x_      = x_full_(all, seq(0, x_full_.cols()/2-1));

  VectorXi z(x_.rows());

  if (init_cluster == 1) 
    z.setZero();
  else if (init_cluster >= 1) {
    boost::random::uniform_int_distribution<> uni_(0, init_cluster-1);
    for (int i=0; i<N_; ++i) z[i] = uni_(rndGen_); 
  }
  else { 
    cout<< "Invalid Number of Initial Components" << endl;
    exit(1);
  }
  z_ = z;

  K_ = z_.maxCoeff() + 1; 
  this ->updateIndexLists();
};


template <class dist_t> 
DPMM<dist_t>::DPMM(const MatrixXd& x, const VectorXi& z, const vector<int> indexList, const double alpha, const dist_t& H, boost::mt19937 &rndGen)
: alpha_(alpha), H_(H), rndGen_(rndGen), N_(x.rows()), z_(z), K_(z.maxCoeff() + 1), indexList_(indexList)
{
  // Slice the data if containing directional info
  if (x.cols()==4 || x.cols()==6)  
    x_ = x(all, seq(0, x.cols()/2-1));


  //Initialize the data points of given indexList by randomly assigning them into one of the two clusters
  vector<int> indexList_i;
  vector<int> indexList_j;
  int z_i = z_.maxCoeff() + 1; 
  int z_j = z_[indexList_[0]];


  boost::random::uniform_int_distribution<> uni_01(0, 1);
  for (int ii = 0; ii<indexList_.size(); ++ii)  {
    if (uni_01(rndGen_) == 0) {
        indexList_i.push_back(indexList_[ii]);
        z_[indexList_[ii]] = z_i;
      }
    else  {
        indexList_j.push_back(indexList_[ii]);
        z_[indexList_[ii]] = z_j;
      }
  }
  indexLists_.push_back(indexList_i);
  indexLists_.push_back(indexList_j);
};



template <class dist_t> 
void DPMM<dist_t>::sampleCoefficients()
{
  VectorXd Pi(K_);
  for (uint32_t kk=0; kk<K_; ++kk)  {
    boost::random::gamma_distribution<> gamma_(indexLists_[kk].size(), 1);
    Pi(kk) = gamma_(rndGen_);
  }
  Pi_ = Pi / Pi.sum();
}



template <class dist_t> 
void DPMM<dist_t>::sampleParameters()
{ 
  components_.clear();
  parameters_.clear();

  for (uint32_t kk=0; kk<K_; ++kk)
  {
    parameters_.push_back(H_.posterior(x_(indexLists_[kk], all)));
    components_.push_back(parameters_[kk].sampleParameter());
  }
}



template <class dist_t> 
void DPMM<dist_t>::sampleCoefficientsParameters()
{ 
  components_.clear();
  parameters_.clear();
  VectorXd Pi(K_);

  for (uint32_t kk=0; kk<K_; ++kk)
  {
    boost::random::gamma_distribution<> gamma_(indexLists_[kk].size(), 1);
    Pi(kk) = gamma_(rndGen_);
    parameters_.push_back(H_.posterior(x_(indexLists_[kk], all)));
    components_.push_back(parameters_[kk].sampleParameter());
  }

  Pi_ = Pi / Pi.sum();
}




template <class dist_t> 
void DPMM<dist_t>::sampleLabels()
{
  #pragma omp parallel for num_threads(4) schedule(static) private(rndGen_)
  for(uint32_t ii=0; ii<N_; ++ii)
  { 
    VectorXd prob(K_);
    for (uint32_t kk=0; kk<K_; ++kk) {
      prob[kk] = log(Pi_[kk]) + components_[kk].logProb(x_(ii, all));
    }

    prob = (prob.array()-(prob.maxCoeff() + log((prob.array() - prob.maxCoeff()).exp().sum()))).exp().matrix();
    prob = prob / prob.sum();
    for (uint32_t kk = 1; kk < prob.size(); ++kk)  prob[kk] = prob[kk-1]+ prob[kk];

    boost::random::uniform_01<> uni_;   
    double uni_draw = uni_(rndGen_);
    uint32_t kk = 0;
    while (prob[kk] < uni_draw) kk++;
    z_[ii] = kk;
  }
}



template <class dist_t> 
void DPMM<dist_t>::sampleCoefficientsParameters(const vector<int> &indexList)
{
  parameters_.clear();
  components_.clear();

  parameters_.push_back(H_.posterior(x_(indexLists_[0], all)));
  parameters_.push_back(H_.posterior(x_(indexLists_[1], all)));
  components_.push_back(parameters_[0].sampleParameter());
  components_.push_back(parameters_[1].sampleParameter());
  

  VectorXd Pi(2);
  for (uint32_t kk=0; kk<2; ++kk)
  {
    boost::random::gamma_distribution<> gamma_(indexLists_[kk].size(), 1);
    Pi(kk) = gamma_(rndGen_);
  }
  Pi_ = Pi / Pi.sum();
}


template <class dist_t> 
void DPMM<dist_t>::sampleLabels(const vector<int> &indexList)
{
  indexLists_.clear();
  vector<int> indexList_i;
  vector<int> indexList_j;

  boost::random::uniform_01<> uni_;    
  #pragma omp parallel for num_threads(6) schedule(static) private(rndGen_)    
  for(uint32_t ii=0; ii<indexList.size(); ++ii)  {
    VectorXd prob(2);
    for (uint32_t kk=0; kk<2; ++kk)
      prob[kk] = log(Pi_[kk]) + components_[kk].logProb(x_(indexList[ii], all)); 

    prob = (prob.array()-(prob.maxCoeff() + log((prob.array() - prob.maxCoeff()).exp().sum()))).exp().matrix();
    prob = prob / prob.sum();
    
    double uni_draw = uni_(rndGen_);

    #pragma omp critical
    if (uni_draw < prob[0]) 
      indexList_i.push_back(indexList[ii]);
    else 
      indexList_j.push_back(indexList[ii]);
  }

  indexLists_.push_back(indexList_i);
  indexLists_.push_back(indexList_j);
}



template <class dist_t> 
int DPMM<dist_t>::splitProposal(const vector<int> &indexList)
{
  VectorXi z_launch = z_;
  VectorXi z_split = z_;
  uint32_t z_split_i = z_split.maxCoeff() + 1;
  uint32_t z_split_j = z_split[indexList[0]];

  NIWDIR<double> H_NIWDIR = * H_.NIWDIR_ptr;
  DPMMDIR<NIWDIR<double>> dpmm_split(x_full_, z_launch, indexList, alpha_, H_NIWDIR, rndGen_);

  for (int tt=0; tt<50; ++tt)  {
    if (dpmm_split.indexLists_[0].empty()==true || dpmm_split.indexLists_[1].empty()==true)
      return 1;
    dpmm_split.sampleCoefficientsParameters(indexList);
    dpmm_split.sampleLabels(indexList);
  }

  vector<int> indexList_i = dpmm_split.indexLists_[0];
  vector<int> indexList_j = dpmm_split.indexLists_[1];


  double logAcceptanceRatio = 0;
  logAcceptanceRatio -= dpmm_split.logProposalRatio(indexList_i, indexList_j);
  logAcceptanceRatio += dpmm_split.logTargetRatio(indexList_i, indexList_j);

  if (logAcceptanceRatio > 0) {
    for (int i = 0; i < indexList_i.size(); ++i)
      z_split[indexList_i[i]] = z_split_i;
    for (int i = 0; i < indexList_j.size(); ++i)
      z_split[indexList_j[i]] = z_split_j;

    z_ = z_split;
    K_ += 1;
    logNum_.push_back(K_);
    std::cout << "Component " << z_split_j <<": Split proposal Aceepted with Log Acceptance Ratio " << logAcceptanceRatio << std::endl;
    return 0;
  }
  else
    return 1;
}



template <class dist_t> 
int DPMM<dist_t>::mergeProposal(const vector<int> &indexList_i, const vector<int> &indexList_j)
{
  double logAcceptanceRatio = 0;
  VectorXi z_launch = z_;
  VectorXi z_merge = z_;
  uint32_t z_merge_i = z_merge[indexList_i[0]];
  uint32_t z_merge_j = z_merge[indexList_j[0]];

  vector<int> indexList;
  indexList.reserve(indexList_i.size() + indexList_j.size() ); // preallocate memory
  indexList.insert( indexList.end(), indexList_i.begin(), indexList_i.end() );
  indexList.insert( indexList.end(), indexList_j.begin(), indexList_j.end() );

  NIWDIR<double> H_NIWDIR = * H_.NIWDIR_ptr;
  DPMMDIR<NIWDIR<double>> dpmm_merge(x_full_, z_launch, indexList, alpha_, H_NIWDIR, rndGen_);
  // DPMM<NIW<double>> dpmm_merge(x_full_, z_launch, indexList, alpha_, H_, rndGen_);
  
  for (int tt=0; tt<50; ++tt)  {    
    if (dpmm_merge.indexLists_[0].empty()==true || dpmm_merge.indexLists_[1].empty()==true)
    {
      // double logAcceptanceRatio = 0;
      // logAcceptanceRatio += log(dpmm_merge.transitionProb(indexList_i, indexList_j));
      // logAcceptanceRatio -= dpmm_merge.logPosteriorProb(indexList_i, indexList_j);;

      // std::cout << logAcceptanceRatio << std::endl;
      for (int i = 0; i < indexList_i.size(); ++i) 
        z_merge[indexList_i[i]] = z_merge_j;
      z_ = z_merge;
      this -> reorderAssignments();
      std::cout << "Component " << z_merge_j << "and" << z_merge_i <<": Merge proposal Aceepted with Log Acceptance Ratio " << logAcceptanceRatio << std::endl;
      return 0;
    };
    dpmm_merge.sampleCoefficientsParameters(indexList);
    dpmm_merge.sampleLabels(indexList);
  }


  logAcceptanceRatio += dpmm_merge.logProposalRatio(indexList_i, indexList_j);
  logAcceptanceRatio -= dpmm_merge.logTargetRatio(indexList_i, indexList_j);;

  if (logAcceptanceRatio > 0)  {
    for (int i = 0; i < indexList_i.size(); ++i) 
      z_merge[indexList_i[i]] = z_merge_j;
    z_ = z_merge;
    this -> reorderAssignments();
    std::cout << "Component " << z_merge_j << " and " << z_merge_i <<": Merge proposal Aceepted with Log Acceptance Ratio " << logAcceptanceRatio << std::endl;
    return 0;
  }
  std::cout << "Component " << z_merge_j << " and " << z_merge_i <<": Merge proposal Rejected with Log Acceptance Ratio " << logAcceptanceRatio << std::endl;
  return 1;
}


template <class dist_t> 
double DPMM<dist_t>::logProposalRatio(vector<int> indexList_i, vector<int> indexList_j)
{
  double logProposalRatio = 0;

  for (uint32_t ii=0; ii < indexList_i.size(); ++ii)  {
    logProposalRatio += log(Pi_(0) * parameters_[0].predProb(x_(indexList_i[ii], all))) -
    log(Pi_(0) * parameters_[0].predProb(x_(indexList_i[ii], all)) + Pi_(1) *  parameters_[1].predProb(x_(indexList_i[ii], all)));
  }

  for (uint32_t ii=0; ii < indexList_j.size(); ++ii)  {
    logProposalRatio += log(Pi_(1) * parameters_[1].predProb(x_(indexList_j[ii], all))) -
    log(Pi_(0) * parameters_[0].predProb(x_(indexList_j[ii], all)) + Pi_(1) *  parameters_[1].predProb(x_(indexList_j[ii], all)));
  }

  return logProposalRatio;
}


template <class dist_t>
double DPMM<dist_t>::logTargetRatio(vector<int> indexList_i, vector<int> indexList_j)
{
  vector<int> indexList_ij;
  indexList_ij.reserve(indexList_i.size() + indexList_j.size() ); // preallocate memory
  indexList_ij.insert( indexList_ij.end(), indexList_i.begin(), indexList_i.end() );
  indexList_ij.insert( indexList_ij.end(), indexList_j.begin(), indexList_j.end() );

  NIW<double> parameter_ij = H_.posterior(x_(indexList_ij, all));
  NIW<double> parameter_i  = H_.posterior(x_(indexList_i, all));
  NIW<double> parameter_j  = H_.posterior(x_(indexList_j, all));

  double logTargetRatio = 0;
  for (uint32_t ii=0; ii < indexList_i.size(); ++ii) {
    logTargetRatio += parameter_i.logPredProb(x_(indexList_i[ii], all)) ;
    logTargetRatio -= parameter_ij.logPredProb(x_(indexList_i[ii], all));
  }
  for (uint32_t jj=0; jj < indexList_j.size(); ++jj)  {
    logTargetRatio += parameter_j.logPredProb(x_(indexList_j[jj], all)) ;
    logTargetRatio -= parameter_ij.logPredProb(x_(indexList_j[jj], all));
  }

  double logPrior = indexList_i.size() * log(indexList_i.size()) + 
                    indexList_j.size() * log(indexList_j.size()) - 
                    indexList_ij.size() * log(indexList_ij.size());
  logTargetRatio += logPrior;

  return logTargetRatio;
}


template <class dist_t>
void DPMM<dist_t>::reorderAssignments()
{ 
  vector<uint8_t> rearrange_list;
  for (uint32_t ii=0; ii<N_; ++ii)
  {
    if (rearrange_list.empty()) rearrange_list.push_back(z_[ii]);
    vector<uint8_t>::iterator it;
    it = find (rearrange_list.begin(), rearrange_list.end(), z_[ii]);
    if (it == rearrange_list.end())
    {
      rearrange_list.push_back(z_[ii]);
      z_[ii] = rearrange_list.size() - 1;
    }
    else if (it != rearrange_list.end())
    {
      int index = it - rearrange_list.begin();
      z_[ii] = index;
    }
  }
  K_ = z_.maxCoeff() + 1;
}


template <class dist_t>
vector<vector<int>> DPMM<dist_t>::getIndexLists()
{
  this ->updateIndexLists();
  return indexLists_;
}


template <class dist_t>
void DPMM<dist_t>::updateIndexLists()
{
  vector<vector<int>> indexLists(K_);
  for (uint32_t ii = 0; ii<N_; ++ii)  
    indexLists[z_[ii]].push_back(ii); 
  indexLists_ = indexLists;
}




template <class dist_t> 
vector<vector<int>> DPMM<dist_t>::computeSimilarity()
{
  int num_comp = K_;
  vector<vector<int>> indexLists = this-> getIndexLists();
  vector<MatrixXd>       muLists;

  for (int kk=0; kk< num_comp; ++kk)
  {
    MatrixXd x_k = x_(indexLists[kk],  seq(0, (x_.cols()/2)-1));
    muLists.push_back(x_k.colwise().mean().transpose());
  }

  MatrixXd similarityMatrix = MatrixXd::Constant(num_comp, num_comp, numeric_limits<float>::infinity());
  for (int ii=0; ii<num_comp; ++ii)
      for (int jj=ii+1; jj<num_comp; ++jj)
          similarityMatrix(ii, jj) = (muLists[ii] - muLists[jj]).norm();

  MatrixXd similarityMatrix_flattened;
  similarityMatrix_flattened = similarityMatrix.transpose(); 
  similarityMatrix_flattened.resize(1, (similarityMatrix.rows() * similarityMatrix.cols()) );  


  Eigen::MatrixXf::Index min_index;
  similarityMatrix_flattened.row(0).minCoeff(&min_index);

  int merge_i;
  int merge_j;
  int min_index_int = min_index;

  merge_i = min_index_int / num_comp;
  merge_j = min_index_int % num_comp;

  vector<vector<int>> merge_indexLists;

  merge_indexLists.push_back(indexLists[merge_i]);
  merge_indexLists.push_back(indexLists[merge_j]);

 return merge_indexLists;
}



template class DPMM<NIW<double>>;


/*---------------------------------------------------*/
//Following class methods are currently not being used 
/*---------------------------------------------------*/

/*

template <class dist_t> 
void DPMM<dist_t>::sampleCoefficients(const uint32_t index_i, const uint32_t index_j)
{
  VectorXi Nk(2);
  Nk.setZero();

  for(uint32_t ii=0; ii<indexList_.size(); ++ii)
  {
    if (z_[indexList_[ii]]==z_[index_i]) Nk(0)++;
    else Nk(1)++;
  }

  //`````testing``````````````
  // std::cout << Nk <<std::endl;
  //`````testing``````````````


  VectorXd Pi(2);
  for (uint32_t k=0; k<2; ++k)
  {
    boost::random::gamma_distribution<> gamma_(Nk(k), 1);
    Pi(k) = gamma_(rndGen_);
  }
  Pi_ = Pi / Pi.sum();
  // std::cout << Pi_ <<std::endl;
}




template <class dist_t> 
void DPMM<dist_t>::sampleParameters(const uint32_t index_i, const uint32_t index_j)
{
  components_.clear();
  parameters_.clear();

  vector<int> indexList_i;
  vector<int> indexList_j;

  for (uint32_t ii = 0; ii<indexList_.size(); ++ii)  //To-do: This can be combined with sampleCoefficients
  {
    if (z_[indexList_[ii]]==z_[index_i]) 
    indexList_i.push_back(indexList_[ii]); 
    else if (z_[indexList_[ii]]==z_[index_j])
    indexList_j.push_back(indexList_[ii]);
  }
  assert(indexList_i.size() + indexList_j.size() == indexList_.size());
  MatrixXd x_i(indexList_i.size(), x_.cols()); 
  MatrixXd x_j(indexList_j.size(), x_.cols()); 


  //`````testing``````````````
  // std::cout << indexList_i.size() << std::endl << indexList_j.size() << std::endl;
  //`````testing``````````````


  x_i = x_(indexList_i, all);
  x_j = x_(indexList_j, all);

  components_.push_back(H_.posterior(x_i));
  components_.push_back(H_.posterior(x_j));
  parameters_.push_back(components_[0].sampleParameter());
  parameters_.push_back(components_[1].sampleParameter());
}

template <class dist_t> 
Normal<double> DPMM<dist_t>::sampleParameters(vector<int> indexList)
{ 
  return H_.posterior(x_(indexList, all)).sampleParameter();
}




template <class dist_t> 
void DPMM<dist_t>::sampleCoefficientsParameters(uint32_t index_i, uint32_t index_j)
{
  vector<int> indexList_i;
  vector<int> indexList_j;

  // std::cout << index_i <<std::endl << index_j << std::endl;
  assert(z_[index_i] !=  z_[index_j]);
  // #pragma omp parallel for num_threads(8) schedule(dynamic,100)
  for (uint32_t ii = 0; ii<indexList_.size(); ++ii) 
  {
    if (z_[indexList_[ii]]==z_[index_i]) 
    indexList_i.push_back(indexList_[ii]); 
    else if (z_[indexList_[ii]]==z_[index_j])
    indexList_j.push_back(indexList_[ii]);
  }
  assert(indexList_i.size() + indexList_j.size() == indexList_.size());

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


  // //`````testing``````````````
  // std::cout << Nk <<std::endl;
  // //`````testing``````````````


  VectorXd Pi(2);
  for (uint32_t k=0; k<2; ++k)
  {
    boost::random::gamma_distribution<> gamma_(Nk(k), 1);
    Pi(k) = gamma_(rndGen_);
  }
  Pi_ = Pi / Pi.sum();


  // //`````testing``````````````
  // std::cout << Pi_ <<std::endl;  
  // //`````testing``````````````
}




template <class dist_t> 
void DPMM<dist_t>::sampleLabels(const uint32_t index_i, const uint32_t index_j)
{
  uint32_t z_i = z_[index_i];
  uint32_t z_j = z_[index_j];
  assert(z_i!=z_j);
  boost::random::uniform_01<> uni_;    //maybe put in constructor?
  #pragma omp parallel for num_threads(4) schedule(static) private(rndGen_)
  for(uint32_t i=0; i<indexList_.size(); ++i)
  {
    VectorXd x_i;
    x_i = x_(indexList_[i], all); //current data point x_i from the index_list
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
    if (uni_draw < prob[0]) z_[indexList_[i]] = z_i;
    else z_[indexList_[i]] = z_j;
  }
  z_[index_i] = z_i;
  z_[index_j] = z_j;
  // //`````testing``````````````
  // std::cout << z_i << std::endl << z_j <<std::endl;
  // //`````testing``````````````
}



template <class dist_t> 
void DPMM<dist_t>::sampleSplit(uint32_t z_i, uint32_t z_j)
{
  vector<int> indexList_i;
  vector<int> indexList_j;

  // #pragma omp parallel for num_threads(8) schedule(dynamic,100)
  for (uint32_t ii = 0; ii<indexList_.size(); ++ii) 
  {
    if (z_[indexList_[ii]]==z_i) 
    indexList_i.push_back(indexList_[ii]); 
    else if (z_[indexList_[ii]]==z_j)
    indexList_j.push_back(indexList_[ii]);
  }
  assert(indexList_i.size() + indexList_j.size() == indexList_.size());

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


  // //`````testing``````````````
  // std::cout << Nk <<std::endl;
  // //`````testing``````````````


  VectorXd Pi(2);
  for (uint32_t k=0; k<2; ++k)
  {
    boost::random::gamma_distribution<> gamma_(Nk(k), 1);
    Pi(k) = gamma_(rndGen_);
  }
  Pi_ = Pi / Pi.sum();


  // //`````testing``````````````
  // std::cout << Pi_ <<std::endl;  
  // //`````testing``````````````

  boost::random::uniform_01<> uni_;    //maybe put in constructor?
  #pragma omp parallel for num_threads(4) schedule(static) private(rndGen_)
  for(uint32_t i=0; i<indexList_.size(); ++i)
  {
    VectorXd x_i;
    x_i = x_(indexList_[i], all); //current data point x_i from the index_list
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
    if (uni_draw < prob[0]) z_[indexList_[i]] = z_i;
    else z_[indexList_[i]] = z_j;
  }
}

template <class dist_t> 
int DPMM<dist_t>::splitProposal(vector<int> indexList)
{ 
  
  boost::random::uniform_int_distribution<> uni_(0, indexList.size()-1);
  uint32_t index_i = indexList[uni_(rndGen_)];
  uint32_t index_j = indexList[uni_(rndGen_)];
  while (index_i == index_j)
  {
    index_i = indexList[uni_(rndGen_)];
    index_j = indexList[uni_(rndGen_)];
  }
  assert(index_i!=index_j);
  

  VectorXi z_launch = z_; //original assignment vector
  uint32_t z_split_i = z_launch.maxCoeff() + 1;
  uint32_t z_split_j = z_launch[index_j];


  boost::random::uniform_int_distribution<> uni_01(0, 1);
  for (uint32_t ii = 0; ii<indexList.size(); ++ii)
  {
    if (uni_01(rndGen_) == 0) z_launch[indexList[ii]] = z_split_i;
    else z_launch[indexList[ii]] = z_split_j;
  }


  z_launch[index_i] = z_split_i;
  z_launch[index_j] = z_split_j;
  
  

  
  VectorXi z_launch = z_; //original assignment vector including all xs
  uint32_t z_split_i = z_launch.maxCoeff() + 1;
  uint32_t z_split_j = z_launch[indexList[0]];

  boost::random::uniform_int_distribution<> uni_01(0, 1);
  for (uint32_t ii = 0; ii<indexList.size(); ++ii)
  {
    if (uni_01(rndGen_) == 0) z_launch[indexList[ii]] = z_split_i;
    else z_launch[indexList[ii]] = z_split_j;
  }
  
  
  DPMM<dist_t> dpmm_split(x_, z_launch, indexList, alpha_, H_, rndGen_);

  for (uint32_t t=0; t<500; ++t)
  {
    // std::cout << t << std::endl;
    // dpmm_split.sampleCoefficients(index_i, index_j);
    // dpmm_split.sampleParameters(index_i, index_j); 
    dpmm_split.sampleCoefficientsParameters(index_i, index_j);
    dpmm_split.sampleLabels(index_i, index_j);  
    // dpmm_split.sampleSplit(z_split_i, z_split_j);  
  }
  
  vector<int> indexList_i = dpmm_split.getIndexLists()[z_split_i];
  vector<int> indexList_j = dpmm_split.getIndexLists()[z_split_j];

  double transitionRatio = 1.0 / dpmm_split.transitionProb(index_i, index_j);
  double posteriorRatio = dpmm_split.posteriorRatio(indexList_i, indexList_j, indexList);
  double acceptanceRatio = transitionRatio * posteriorRatio;
  

  if (acceptanceRatio >= 1) 
  {
    z_ = dpmm_split.z_;
    K_ += 1;
    this -> updateIndexLists();
    this -> sampleCoefficients();
    this -> sampleParameters();
    std::cout << "Component " << z_split_j <<": Split proposal Aceepted" << std::endl;
    // std::cout << Pi_ << std::endl;
    // std::cout << parameters_.size() << std::endl;
    return 0;
  }
  else
  {
    std::cout << "Component " << z_split_j <<": Split proposal Rejected" << std::endl;
    return 1;
  }
}


template <class dist_t> 
int DPMM<dist_t>::mergeProposal(vector<int> indexList_i, vector<int> indexList_j)
{ 
  VectorXi z_split = z_; //original split state
  
  boost::random::uniform_int_distribution<> uni_i(0, indexList_i.size()-1);
  boost::random::uniform_int_distribution<> uni_j(0, indexList_j.size()-1);
  uint32_t index_i = indexList_i[uni_i(rndGen_)];
  uint32_t index_j = indexList_j[uni_j(rndGen_)];
  assert(index_i!=index_j);

  VectorXi z_launch = z_; 
  uint32_t z_split_i = z_[index_i];
  uint32_t z_split_j = z_[index_j];

  vector<int> indexList;
  indexList.reserve( indexList_i.size() + indexList_j.size() ); // preallocate memory
  indexList.insert( indexList.end(), indexList_i.begin(), indexList_i.end() );
  indexList.insert( indexList.end(), indexList_j.begin(), indexList_j.end() );
  assert(indexList.size() == indexList_i.size() + indexList_j.size());

  boost::random::uniform_int_distribution<> uni_01(0, 1);
  for (uint32_t ii = 0; ii<indexList.size(); ++ii)
  {
    if (uni_01(rndGen_) == 0) z_launch[indexList[ii]] = z_split_i;
    else z_launch[indexList[ii]] = z_split_j;
  }
  z_launch[index_i] = z_split_i;
  z_launch[index_j] = z_split_j;


  DPMM<dist_t> dpmm_merge(x_, z_launch, indexList, alpha_, H_, rndGen_);
  for (uint32_t t=0; t<50; ++t)
  {
    // std::cout << t << std::endl;
    dpmm_merge.sampleCoefficients(index_i, index_j);
    dpmm_merge.sampleParameters(index_i, index_j); 
    // dpmm_split.sampleCoefficientsParameters(index_i, index_j);
    dpmm_merge.sampleLabels(index_i, index_j);  
  }

  indexList_i = dpmm_merge.getIndexLists()[z_split_i];
  indexList_j = dpmm_merge.getIndexLists()[z_split_j];

  double transitionRatio = dpmm_merge.transitionProb(index_i, index_j, z_);
  double posteriorRatio = 1.0 / dpmm_merge.posteriorRatio(indexList_i, indexList_j, indexList);
  double acceptanceRatio = transitionRatio * posteriorRatio;

  if (acceptanceRatio >= 1) 
  {
    std::cout << "Component " << z_split_i << " and " << z_split_j << ": Merge proposal Aceepted" << std::endl; 
    z_ = dpmm_merge.z_;
    K_ -= 1;
    this ->updateIndexLists();
    this -> sampleCoefficients();
    this -> sampleParameters();
    std::cout << "Component " << z_split_i << " and " << z_split_j << ": Merge proposal Aceepted" << std::endl;
    // std::cout << Pi_ << std::endl;
    // std::cout << parameters_.size() << std::endl;
    return 0;
  }
  else
  {
    std::cout  << "Component " << z_split_i << " and " << z_split_j << ": Merge proposal Rejected" << std::endl;
    return 1;
  }
}






template <class dist_t> 
double DPMM<dist_t>::transitionProb(const uint32_t index_i, const uint32_t index_j)
{
  assert(!components_.empty());
  double transitionProb = 1;
  for (uint32_t ii=0; ii < indexList_.size(); ++ii)
  {
    if (z_[indexList_[ii]] == z_[index_i])
    transitionProb *= Pi_(0) * components_[0].prob(x_(indexList_[ii], all))/
    (Pi_(0) * components_[0].prob(x_(indexList_[ii], all)) + Pi_(1) *  components_[1].prob(x_(indexList_[ii], all)));
    else
    transitionProb *= Pi_(1) * components_[1].prob(x_(indexList_[ii], all))/
    (Pi_(0) * components_[0].prob(x_(indexList_[ii], all)) + Pi_(1) *  components_[1].prob(x_(indexList_[ii], all)));
  }
  return transitionProb;
}

template <class dist_t> 
double DPMM<dist_t>::transitionProb(const uint32_t index_i, const uint32_t index_j,VectorXi z_original)
{
  z_ = z_original;
  return this->transitionProb(index_i, index_j);
}


template <class dist_t>
double DPMM<dist_t>::posteriorRatio(vector<int> indexList_i, vector<int> indexList_j, vector<int> indexList_ij)
{
  Normal<double> parameter_ij = H_.posterior(x_(indexList_ij, all)).sampleParameter();
  Normal<double> parameter_i  = H_.posterior(x_(indexList_i, all)).sampleParameter();
  Normal<double> parameter_j  = H_.posterior(x_(indexList_j, all)).sampleParameter();

  double logPosteriorRatio = 0;
  for (uint32_t ii=0; ii < indexList_i.size(); ++ii)
  {
    logPosteriorRatio += log(indexList_i.size()) + parameter_i.logProb(x_(indexList_i[ii], all)) ;
    logPosteriorRatio -= parameter_ij.logProb(x_(indexList_i[ii], all));
  }
  for (uint32_t jj=0; jj < indexList_j.size(); ++jj)
  {
    logPosteriorRatio += log(indexList_j.size()) + parameter_j.logProb(x_(indexList_j[jj], all)) ;
    logPosteriorRatio -= parameter_ij.logProb(x_(indexList_j[jj], all));
  }

  return exp(logPosteriorRatio);
}


template <class dist_t>
void DPMM<dist_t>::removeEmptyClusters()
{
  for(uint32_t k=parameters_.size()-1; k>=0; --k)
  {
    bool haveCluster_k = false;
    for(uint32_t i=0; i<z_.size(); ++i)
      if(z_(i)==k)
      {
        haveCluster_k = true;
        break;
      }
    if (!haveCluster_k)
    {
      for (uint32_t i=0; i<z_.size(); ++i)
        if(z_(i) >= k) z_(i) --;
      parameters_.erase(parameters_.begin()+k);
    }
  }
}



  vector<int> i_array = generateRandom(N_);
  for(uint32_t j=0; j<N_; ++j)
  // for(uint32_t i=0; i<N_; ++i)
  {
    int i = i_array[j];
    // cout << "number of data point: " << i << endl;
    VectorXi Nk(K_);
    Nk.setZero();
    for(uint32_t ii=0; ii<N_; ++ii)
    {
      if (ii != i) Nk(z_(ii))++;
    }
    // cout<< Nk << endl;
    VectorXd pi(K_+1); 
    // VectorXd pi(K_); 

    VectorXd x_i;
    x_i = x_(i, all); //current data point x_i

    // #pragma omp parallel for
    for (uint32_t k=0; k<K_; ++k)
    { 
      vector<int> indexList_k;
      for (uint32_t ii = 0; ii<N_; ++ii)
      {
        if (ii!= i && z_[ii] == k) indexList_k.push_back(ii); 
      }
      // if (indexList_k.empty()) 
      // cout << "no index" << endl;
      // cout << "im here" <<endl;


      MatrixXd x_k(indexList_k.size(), x_.cols()); 
      x_k = x_(indexList_k, all);
      // cout << "x_i" << x_i << endl;
      // cout << "x_k" << x_k << endl;
      // cout << "component:" <<k  <<endl;
      // cout << x_k << endl;
      // cout << Nk(k) << endl;
      if (Nk(k)!=0)
      pi(k) = log(Nk(k))-log(N_+alpha_) + parameters_[k].logPosteriorProb(x_i, x_k);
      else
      pi(k) = - std::numeric_limits<float>::infinity();
    }
    pi(K_) = log(alpha_)-log(N_+alpha_) + H_.logProb(x_i);


    // cout << pi <<endl;
    // exit(1);


    
    
    double pi_max = pi.maxCoeff();
    pi = (pi.array()-(pi_max + log((pi.array() - pi_max).exp().sum()))).exp().matrix();
    pi = pi / pi.sum();


    for (uint32_t ii = 1; ii < pi.size(); ++ii){
      pi[ii] = pi[ii-1]+ pi[ii];
    }
   
    boost::random::uniform_01<> uni_;   
    boost::random::variate_generator<boost::random::mt19937&, 
                           boost::random::uniform_01<> > var_nor(*H_.rndGen_, uni_);
    double uni_draw = var_nor();
    uint32_t k = 0;
    while (pi[k] < uni_draw) k++;
    z_[i] = k;
    this -> reorderAssignments();
  }

};



template <class dist_t> 
void DPMM<dist_t>::sampleLabelsCollapsed(const vector<int> &indexList)
{
  int dimPos;
  if (x_.cols()==4) dimPos=1;
  else if (x_.cols()==6) dimPos=2;
  int index_i = z_[indexLists_[0][0]];
  int index_j = z_[indexLists_[1][0]];


  boost::random::uniform_01<> uni_;
  vector<int> indexList_i;
  vector<int> indexList_j;

  // #pragma omp parallel for num_threads(4) schedule(static) private(rndGen_)
  for(int i=0; i<indexList.size(); ++i)
  {
    VectorXd x_i;
    x_i = x_(indexList[i], seq(0,dimPos)); //current data point x_i from the index_list
    VectorXd prob(2);

    for (int ii=0; ii < indexList.size(); ++ii)
    {
      if (z_[indexList[ii]] == index_i && ii!=i) indexList_i.push_back(indexList[ii]);
      else if (z_[indexList[ii]] == index_j && ii!=i) indexList_j.push_back(indexList[ii]);
    }

    if (indexList_i.empty()==true || indexList_j.empty()==true)
    {
      indexLists_.clear();
      indexLists_.push_back(indexList_i);
      indexLists_.push_back(indexList_j);
      return;
    } 

    prob[0] = log(indexList_i.size()) + H_.logPosteriorPredictiveProb(x_i, x_(indexList_i, seq(0,dimPos))); 
    prob[1] = log(indexList_j.size()) + H_.logPosteriorPredictiveProb(x_i, x_(indexList_j, seq(0,dimPos))); 

    double prob_max = prob.maxCoeff();
    prob = (prob.array()-(prob_max + log((prob.array() - prob_max).exp().sum()))).exp().matrix();
    prob = prob / prob.sum();
    for (uint32_t ii = 1; ii < prob.size(); ++ii)
    {
      prob[ii] = prob[ii-1]+ prob[ii];
    }
    double uni_draw = uni_(rndGen_);
    if (uni_draw < prob[0]) z_[indexList[i]] = index_i;
    else z_[indexList[i]] = index_j;
    
    indexList_i.clear();
    indexList_j.clear();
  }


  for (int i=0; i < indexList.size(); ++i)
  {
    if (z_[indexList[i]] == index_i) indexList_i.push_back(indexList[i]);
    else if (z_[indexList[i]] == index_j)indexList_j.push_back(indexList[i]);
  }
  indexLists_.clear();
  indexLists_.push_back(indexList_i);
  indexLists_.push_back(indexList_j);
}



*/