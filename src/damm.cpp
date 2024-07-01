#include <iostream>
#include <limits>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>


#include "dpmm.hpp"
#include "damm.hpp"
#include "niw.hpp"
#include "niwDamm.hpp"



template <class dist_t> 
Damm<dist_t>::Damm(const MatrixXd &x, int init_cluster, double alpha, const dist_t &H, const boost::mt19937 &rndGen)
: alpha_(alpha), H_(H), rndGen_(rndGen), N_(x.rows())
{
  dim_   = x.cols()/2;
  x_     = x;
  xPos_  = x_(all, seq(0, dim_-1));
  xDir_  = x_(all, seq(dim_, last));


  VectorXi z(x.rows());

  if (init_cluster == 1) 
    z.setZero();
  else if (init_cluster > 1)  {
    boost::random::uniform_int_distribution<> uni_(0, init_cluster-1);
    for (int ii=0; ii<N_; ++ii) 
      z[ii] = uni_(rndGen_); 
  }
  else  { 
    cout<< "Number of initial clusters not supported yet" << endl;
    exit(1);
  }


  z_ = z;
  K_ = z_.maxCoeff() + 1; 
  logZ_.push_back(z_);
  logNum_.push_back(K_);
  this ->updateIndexLists();
};



template <class dist_t> 
Damm<dist_t>::Damm(const MatrixXd &x, int init_cluster, double alpha, const dist_t &H, const boost::mt19937 &rndGen, VectorXi z)
: alpha_(alpha), H_(H), rndGen_(rndGen), N_(x.rows())
{
  // incremental learning framework when assignment array, z is provided

  dim_   = x.cols()/2;
  x_     = x;
  xPos_  = x_(all, seq(0, dim_-1));
  xDir_  = x_(all, seq(dim_, last));

  // store the index list of new data

  std::cout << "here" << std::endl;

  // ArrayXi index = (z.array() == -1).cast<int>();
  // vector<int> indexList_new_; 
  int K_old = -1;

  for (int ii=0; ii<N_; ++ii){
    if (z[ii] == -1){
      indexList_new_.push_back(ii);
    }
    if (z[ii] > K_old)
      K_old = z[ii];
  }


  std::cout << K_old << std::endl;
  std::cout << indexList_new_.size() << std::endl;

  boost::random::uniform_int_distribution<> uni_(1+K_old, K_old+init_cluster);
  for (int ii=0; ii<indexList_new_.size(); ++ii){
    z[indexList_new_[ii]] = uni_(rndGen_);
  }

  // std::cout << z << std::endl;

  z_ = z;
  K_ = z_.maxCoeff() + 1; 
  // logZ_.push_back(z_);
  // logNum_.push_back(K_);
  this ->updateIndexLists();
};




template <class dist_t> 
void Damm<dist_t>::sampleCoefficientsParameters()
{ 
  parameters_.clear();
  components_.clear();
  VectorXd Pi(K_);

  for (uint32_t kk=0; kk<K_; ++kk)  {
    boost::random::gamma_distribution<> gamma_(indexLists_[kk].size(), 1);
    Pi(kk) = gamma_(rndGen_);
    parameters_.push_back(H_.posterior(x_(indexLists_[kk], all)));
    components_.push_back(parameters_[kk].sampleParameter());
  }
  Pi_ = Pi / Pi.sum();
}

template <class dist_t> 
void Damm<dist_t>::sampleLabels_increm()
{
  // double logLik = 0;
  #pragma omp parallel for num_threads(8) schedule(dynamic, 300) private(rndGen_)
  for(uint32_t ii=0; ii<indexList_new_.size(); ++ii) {
    VectorXd prob(K_);
    // double logLik_i = 0;

    for (uint32_t kk=0; kk<K_; ++kk) { 
      double logProb =  components_[kk].logProb(x_(indexList_new_[ii], all));
      prob[kk] = log(Pi_[kk]) + logProb;
      // logLik_i += Pi_[kk] * exp(logProb);
    }
    // logLik += log(logLik_i);
    double max_prob = prob.maxCoeff();
    prob = (prob.array() - max_prob).exp() / (prob.array() - max_prob).exp().sum();
    // prob = (prob.array()-(prob.maxCoeff() + log((prob.array() - prob.maxCoeff()).exp().sum()))).exp().matrix();
    prob = prob / prob.sum();
    for (uint32_t kk = 1; kk < prob.size(); ++kk) 
      prob[kk] = prob[kk-1]+ prob[kk];
    
    boost::random::uniform_01<> uni_;   
    double uni_draw = uni_(rndGen_);
    uint32_t kk = 0;
    while (prob[kk] < uni_draw) 
      kk++;
    z_[indexList_new_[ii]] = kk;
  } 
  // logLogLik_.push_back(logLik);
  // logZ_.push_back(z_);
}

template <class dist_t> 
void Damm<dist_t>::sampleLabels()
{
  double logLik = 0;
  #pragma omp parallel for num_threads(8) schedule(dynamic, 300) private(rndGen_)
  for(uint32_t ii=0; ii<N_; ++ii) {
    VectorXd prob(K_);
    double logLik_i = 0;

    for (uint32_t kk=0; kk<K_; ++kk) { 
      double logProb =  components_[kk].logProb(x_(ii, all));
      prob[kk] = log(Pi_[kk]) + logProb;
      logLik_i += Pi_[kk] * exp(logProb);
    }
    logLik += log(logLik_i);
    double max_prob = prob.maxCoeff();
    prob = (prob.array() - max_prob).exp() / (prob.array() - max_prob).exp().sum();
    // prob = (prob.array()-(prob.maxCoeff() + log((prob.array() - prob.maxCoeff()).exp().sum()))).exp().matrix();
    prob = prob / prob.sum();
    for (uint32_t kk = 1; kk < prob.size(); ++kk) 
      prob[kk] = prob[kk-1]+ prob[kk];
    
    boost::random::uniform_01<> uni_;   
    double uni_draw = uni_(rndGen_);
    uint32_t kk = 0;
    while (prob[kk] < uni_draw) 
      kk++;
    z_[ii] = kk;
  } 
  logLogLik_.push_back(logLik);
  logZ_.push_back(z_);
}


template <class dist_t> 
int Damm<dist_t>::splitProposal(const vector<int> &indexList)
{ 
  /**
   * This method proposes a split of the given indexList
   * 
   * @note While performing intermediate Gibbs scans, if one of two groups vanishes; i.e., the if condition below meets,
   * the split proposal should be immediately rejected
   * @note Notice in split proposals, no need to call reorderAssignments(), as the newly added group are already 
   * taken care by z_split_i
   */


  uint32_t z_split_i = z_.maxCoeff() + 1;
  uint32_t z_split_j = z_[indexList[0]];


  Dpmm<Niw<double>> dpmm_split(x_, z_, indexList, alpha_, * H_.NIW_ptr, rndGen_);
  
 
  for (int tt=0; tt<50; ++tt) {
    dpmm_split.sampleCoefficientsParameters(indexList);
    dpmm_split.sampleLabels(indexList);
    if (dpmm_split.indexLists_[0].empty()==true || dpmm_split.indexLists_[1].empty()==true)
      return 1;
  }

  
  vector<int> indexList_i = dpmm_split.indexLists_[0];
  vector<int> indexList_j = dpmm_split.indexLists_[1];


  double logAcceptanceRatio = 0;

  logAcceptanceRatio -= dpmm_split.logProposalRatio(indexList_i, indexList_j);
  logAcceptanceRatio += dpmm_split.logTargetRatio(indexList_i, indexList_j);

  if (logAcceptanceRatio > 0) {
    z_(indexList_i) = VectorXi::Constant(indexList_i.size(), z_split_i);
    z_(indexList_j) = VectorXi::Constant(indexList_j.size(), z_split_j);

    logZ_.push_back(z_);
    K_ += 1;
    logNum_.push_back(K_);
    std::cout << "Component " << z_split_j + 1 <<": Split proposal Aceepted with Log Acceptance Ratio " << logAcceptanceRatio << std::endl;
    return 0;
  }
  else{
    // std::cout << "Component " << z_split_j + 1 <<": Split proposal Rejected with Log Acceptance Ratio " << logAcceptanceRatio << std::endl;
    return 1;
  }
}


template <class dist_t> 
int Damm<dist_t>::mergeProposal(const vector<int> &indexList_i, const vector<int> &indexList_j)
{  
  /**
   * This method proposes a merge between two given indexList_i and indexList_j
   * 
   * @note While performing intermediate Gibbs scans, if one of two groups vanishes; i.e., the if condition below meets,
   * the merge proposal should be immediately accepted
   * 
   * @note Notice in merge proposals reorderAssignments() needs to be called (Outside) unlike in split proposal 
   * because the vanishing group results in a void among assignment labels, hence requiring an re-order
   * 
   * @note By default, if merge accepts, z_merge_i vanishes and merges into z_merge_j
   * 
   * @note Calling logProposalRatio would always use the launch state which is stored in 
   * class member as the one used in final Gibbs scan. In this case, we are as if generating
   * the original split state from the launch state
   */


  uint32_t z_merge_i = z_[indexList_i[0]];
  uint32_t z_merge_j = z_[indexList_j[0]];

  vector<int> indexList;
  indexList.reserve(indexList_i.size() + indexList_j.size() ); // preallocate memory
  indexList.insert( indexList.end(), indexList_i.begin(), indexList_i.end() );
  indexList.insert( indexList.end(), indexList_j.begin(), indexList_j.end() );


  Dpmm<Niw<double>> dpmm_merge(x_, z_, indexList, alpha_, * H_.NIW_ptr, rndGen_);  
  for (int tt=0; tt<50; ++tt)  {    
    dpmm_merge.sampleCoefficientsParameters(indexList);
    dpmm_merge.sampleLabels(indexList);
    if (dpmm_merge.indexLists_[0].empty()==true || dpmm_merge.indexLists_[1].empty()==true) {
      z_(indexList) = VectorXi::Constant(indexList.size(), z_merge_j);
      std::cout << "Component " << z_merge_j + 1 << " and " << z_merge_i + 1 <<": Merge proposal Accepted" << std::endl;
      return 0;
    }
  }

  double logAcceptanceRatio = 0;

  logAcceptanceRatio += dpmm_merge.logProposalRatio(indexList_i, indexList_j);
  logAcceptanceRatio -= dpmm_merge.logTargetRatio(indexList_i, indexList_j);

  if (logAcceptanceRatio > 0) {
    z_(indexList) = VectorXi::Constant(indexList.size(), z_merge_j);
    std::cout << "Component " << z_merge_j << " and " << z_merge_i <<": Merge proposal Accepted with Log Acceptance Ratio " << logAcceptanceRatio << std::endl;
    return 0;
  }
  // std::cout << "Component " << z_merge_j << " and " << z_merge_i <<": Merge proposal Rejected with Log Acceptance Ratio " << logAcceptanceRatio << std::endl;
  return 1;
}




template <class dist_t>
void Damm<dist_t>::reorderAssignments()  //mainly called after clusters vanish during parallel sampling
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
  logNum_.push_back(K_);
}


template <class dist_t>
vector<vector<int>> Damm<dist_t>::getIndexLists()
{
  this ->updateIndexLists();
  return indexLists_;
}


template <class dist_t>
void Damm<dist_t>::updateIndexLists()
{
  vector<vector<int>> indexLists(K_);
  for (uint32_t ii = 0; ii<N_; ++ii) 
    indexLists[z_[ii]].push_back(ii); 
  
  indexLists_ = indexLists;
}


template <class dist_t> 
vector<array<int, 2>>  Damm<dist_t>::computeSimilarity(int mergeNum, int mergeIdx)
{
  // std::cout << "Sim Matrix Idx: " << mergeIdx << std::endl;
  vector<vector<int>>     indexLists = this-> getIndexLists();
  vector<MatrixXd>        muLists;
  vector<MatrixXd>        SigmaLists;
  vector<array<int, 2>>   mergeIndexLists;


  if (mergeIdx ==0){
    for (int ii=0; ii<mergeNum/2; ++ii)
      mergeIndexLists.push_back({ii, ii+1});
    return mergeIndexLists;
  }
  else if (mergeIdx==1){
    for (int ii=0; ii<mergeNum/2; ++ii)
      mergeIndexLists.push_back({K_-ii-1, K_-ii-2});
    return mergeIndexLists;
  }

  for (int kk=0; kk< K_; ++kk)  {
    MatrixXd x_k = x_(indexLists[kk],  seq(0, (x_.cols()/2)-1));
    MatrixXd centered = x_k.rowwise() - x_k.colwise().mean();
    MatrixXd cov = (centered.adjoint() * centered) / double(x_k.rows() - 1);
    muLists.push_back(x_k.colwise().mean().transpose());
    SigmaLists.push_back(cov);
  }

  MatrixXd similarityMatrix = MatrixXd::Constant(mergeNum, mergeNum, numeric_limits<float>::infinity());  
  for (int ii=0; ii<mergeNum; ++ii)
      for (int jj=ii+1; jj<mergeNum; ++jj){
        if (mergeIdx==2)
          similarityMatrix(ii, jj) = (muLists[ii] - muLists[jj]).norm();
        else if (mergeIdx==3)
          similarityMatrix(ii, jj) = this->KL_div(SigmaLists[ii], SigmaLists[jj], muLists[ii], muLists[jj]);
      }

  MatrixXd similarityMatrix_flattened;
  similarityMatrix_flattened = similarityMatrix.transpose(); 
  similarityMatrix_flattened.resize(1, (similarityMatrix.rows() * similarityMatrix.cols()) );  


  for (int ii=0; ii<mergeNum; ++ii){
    Eigen::MatrixXf::Index minIdx;
    similarityMatrix_flattened.row(0).minCoeff(&minIdx);

    int merge_i = int(minIdx) / K_;
    int merge_j = int(minIdx) % K_;

    mergeIndexLists.push_back({merge_i, merge_j});
    similarityMatrix_flattened(minIdx) = numeric_limits<float>::infinity();
  }


  return mergeIndexLists;
}

template <class dist_t> 
double Damm<dist_t>::KL_div(const MatrixXd& Sigma_p, const MatrixXd& Sigma_q, const MatrixXd& mu_p, const MatrixXd& mu_q)
{
  double div = 0;
  LLT<MatrixXd> lltObjp(Sigma_p);
  LLT<MatrixXd> lltObjq(Sigma_q);

  div += 2*log(lltObjq.matrixL().determinant());
  div -= 2*log(lltObjp.matrixL().determinant());
  div -= Sigma_p.cols();
  div += (lltObjq.matrixL().solve(mu_p-mu_q)).squaredNorm();
  div += (Sigma_q.inverse() * Sigma_p).trace();

  return div;
}


template class Damm<NiwDamm<double>>;



/*---------------------------------------------------*/
//-------------------Inactive Methods-----------------
/*---------------------------------------------------*/

/*


template <class dist_t> 
Damm<dist_t>::Damm(const MatrixXd& x, const VectorXi& z, const vector<int> indexList, const double alpha, const dist_t& H, boost::mt19937 &rndGen)
: alpha_(alpha), H_(H), rndGen_(rndGen), x_(x), N_(x.rows()), z_(z), K_(z.maxCoeff() + 1), indexList_(indexList)
{
  
  // Initialize the data points of given indexList by randomly assigning them into one of the two clusters


  vector<int> indexList_i;
  vector<int> indexList_j;
  int z_i = z_.maxCoeff() + 1;
  int z_j = z_[indexList[0]];


  boost::random::uniform_int_distribution<> uni_01(0, 1);
  for (int ii = 0; ii<indexList_.size(); ++ii)
  {
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
void Damm<dist_t>::sampleCoefficients()
{
  VectorXd Pi(K_);
  for (uint32_t kk=0; kk<K_; ++kk)
  {
    boost::random::gamma_distribution<> gamma_(indexLists_[kk].size(), 1);
    Pi(kk) = gamma_(rndGen_);
  }
  Pi_ = Pi / Pi.sum();
}


template <class dist_t> 
void Damm<dist_t>::sampleParameters()
{ 
  parameters_.clear();
  components_.clear();

  for (uint32_t kk=0; kk<K_; ++kk)
  {
    parameters_.push_back(H_.posterior(x_(indexLists_[kk], all)));     
    components_.push_back(parameters_[kk].sampleParameter());          
  }
}

template <class dist_t> 
void Damm<dist_t>::sampleCoefficientsParameters(vector<int> indexList)
{
  parameters_.clear();
  components_.clear();

  parameters_.push_back(H_.posterior(x_(indexLists_[0], all)));
  parameters_.push_back(H_.posterior(x_(indexLists_[1], all)));
  components_.push_back(parameters_[0].sampleParameter());
  components_.push_back(parameters_[1].sampleParameter());
  

  VectorXd Pi(2);
  for (uint32_t kk=0; kk<2; ++kk)  {
    boost::random::gamma_distribution<> gamma_(indexLists_[kk].size(), 1);
    Pi(kk) = gamma_(rndGen_);
  }
  Pi_ = Pi / Pi.sum();
}


template <class dist_t> 
void Damm<dist_t>::sampleLabels(vector<int> indexList)
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
double Damm<dist_t>::logProposalRatio(vector<int> indexList_i, vector<int> indexList_j)
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
double Damm<dist_t>::logTargetRatio(vector<int> indexList_i, vector<int> indexList_j)
{
  vector<int> indexList_ij;
  indexList_ij.reserve(indexList_i.size() + indexList_j.size() ); // preallocate memory
  indexList_ij.insert( indexList_ij.end(), indexList_i.begin(), indexList_i.end() );
  indexList_ij.insert( indexList_ij.end(), indexList_j.begin(), indexList_j.end() );

  NiwDamm<double> parameter_ij = H_.posterior(x_(indexList_ij, all));
  NiwDamm<double> parameter_i  = H_.posterior(x_(indexList_i, all));
  NiwDamm<double> parameter_j  = H_.posterior(x_(indexList_j, all));

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
void Damm<dist_t>::sampleLabelsCollapsed(vector<int> indexList)
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

    prob[0] = log(indexList_i.size()) + H_.logPosteriorProb(x_i, x_(indexList_i, seq(0,dimPos))); 
    prob[1] = log(indexList_j.size()) + H_.logPosteriorProb(x_i, x_(indexList_j, seq(0,dimPos))); 

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





template <class dist_t> 
double Damm<dist_t>::logTransitionProb(vector<int> indexList_i, vector<int> indexList_j)
{
  double logTransitionProb = 0;

  for (uint32_t ii=0; ii < indexList_i.size(); ++ii)
  {
    logTransitionProb += log(Pi_(0) * components_[0].prob(x_(indexList_i[ii], all))) -
    log(Pi_(0) * components_[0].prob(x_(indexList_i[ii], all)) + Pi_(1) *  components_[1].prob(x_(indexList_i[ii], all)));
  }

  for (uint32_t ii=0; ii < indexList_j.size(); ++ii)
  {
    logTransitionProb += log(Pi_(0) * components_[0].prob(x_(indexList_j[ii], all))) -
    log(Pi_(0) * components_[0].prob(x_(indexList_j[ii], all)) + Pi_(1) *  components_[1].prob(x_(indexList_j[ii], all)));
  }
  
  // std::cout << transitionProb << std::endl;

  return logTransitionProb;
}


template <class dist_t>
double Damm<dist_t>::logTargetRatio(vector<int> indexList_i, vector<int> indexList_j)
{
  vector<int> indexList_ij;
  indexList_ij.reserve(indexList_i.size() + indexList_j.size() ); // preallocate memory
  indexList_ij.insert( indexList_ij.end(), indexList_i.begin(), indexList_i.end() );
  indexList_ij.insert( indexList_ij.end(), indexList_j.begin(), indexList_j.end() );

  NiwDamm<double> parameter_ij = H_.posterior(x_(indexList_ij, all));
  NiwDamm<double> parameter_i  = H_.posterior(x_(indexList_i, all));
  NiwDamm<double> parameter_j  = H_.posterior(x_(indexList_j, all));

  double logTargetRatio = 0;
  for (uint32_t ii=0; ii < indexList_i.size(); ++ii) {
    logTargetRatio += log(indexList_i.size()) + parameter_i.logPredProb(x_(indexList_i[ii], all)) ;
    logTargetRatio -= log(indexList_ij.size()) - parameter_ij.logPredProb(x_(indexList_i[ii], all));
  }
  for (uint32_t jj=0; jj < indexList_j.size(); ++jj)
  {
    logTargetRatio += log(indexList_j.size()) + parameter_j.logPredProb(x_(indexList_j[jj], all)) ;
    logTargetRatio -= log(indexList_ij.size()) - parameter_ij.logPredProb(x_(indexList_j[jj], all));
  }

  return logTargetRatio;
}

*/