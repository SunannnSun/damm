#include <iostream>
#include <limits>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include "dpmm.hpp"


template <class Dist_t> 
DPMM<Dist_t>::DPMM(const MatrixXd& x, const int init_cluster, const double alpha, const Dist_t& H, boost::mt19937* pRndGen)
: alpha_(alpha), H_(H), pRndGen_(pRndGen), x_(x), N_(x.rows())
{
  VectorXi z(x.rows());
  if (init_cluster == 1) 
  {
    z.setZero();
  }
  else if (init_cluster >= 1)
  {
    boost::random::uniform_int_distribution<> uni_(0, init_cluster-1);
    // #pragma omp parallel for num_threads(8) schedule(static)
    for (int i=0; i<N_; ++i)z[i] = uni_(*pRndGen_); 
  }
  else
  { 
    cout<< "Number of initial clusters not supported yet" << endl;
    exit(1);
  }
  z_ = z;

  K_ = z_.maxCoeff() + 1; // equivalent to the number of initial clusters
};


template <class Dist_t> 
DPMM<Dist_t>::DPMM(const MatrixXd& x, const VectorXi& z, const vector<int> indexList, const double alpha, const Dist_t& H, boost::mt19937* pRndGen)
: alpha_(alpha), H_(H), pRndGen_(pRndGen), x_(x), N_(x.rows()), z_(z), K_(z.maxCoeff() + 1), indexList_(indexList)
{};


template <class Dist_t> 
void DPMM<Dist_t>::splitProposal(const uint32_t index_i, const uint32_t index_j)
{ 
  VectorXi z_launch = z_; //original assignment vector
  uint32_t z_split_i = z_launch.maxCoeff() + 1;
  uint32_t z_split_j = z_launch[index_j];

  vector<int> sIndexList;  
  boost::random::uniform_int_distribution<> uni_(0, 1);
  for (uint32_t ii = 0; ii<N_; ++ii)
  {
    if (z_launch[ii] == z_split_j) 
    {
      if (uni_(*pRndGen_) == 0) z_launch[ii] = z_split_i;
      else z_launch[ii] = z_split_j;
      sIndexList.push_back(ii); //set S including index_i and index_j
    }
  }
  z_launch[index_i] = z_split_i;
  z_launch[index_j] = z_split_j;


  DPMM<Dist_t> dpmm_split(x_, z_launch, sIndexList, alpha_, H_, this->pRndGen_);
  for (uint32_t t=0; t<50; ++t)
  {
    // dpmm_split.sampleCoefficients(index_i, index_j);
    // dpmm_split.sampleParameters(index_i, index_j); 
    dpmm_split.sampleCoefficientsParameters(index_i, index_j);
    dpmm_split.sampleLabels(index_i, index_j);  
  }
  
  std::cout << dpmm_split.transitionProb(index_i, index_j) << std::endl;
  z_ = dpmm_split.z_;
}


template <class Dist_t> 
double DPMM<Dist_t>::transitionProb(const uint32_t index_i, const uint32_t index_j)
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


template <class Dist_t> 
void DPMM<Dist_t>::sampleCoefficients()
{
  VectorXi Nk(K_+1);
  Nk.setZero();
  #pragma omp parallel for num_threads(8) schedule(static)
  for(uint32_t ii=0; ii<N_; ++ii)
  {
    Nk(z_(ii))++;
  }
  Nk(K_) = alpha_;

  VectorXd Pi(K_+1);
  for (uint32_t k=0; k<K_+1; ++k)
  {
    boost::random::gamma_distribution<> gamma_(Nk(k), 1);
    Pi(k) = gamma_(*pRndGen_);
  }
  Pi_ = Pi / Pi.sum();
}


template <class Dist_t> 
void DPMM<Dist_t>::sampleCoefficients(const uint32_t index_i, const uint32_t index_j)
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
    Pi(k) = gamma_(*pRndGen_);
  }
  Pi_ = Pi / Pi.sum();
  std::cout << Pi_ <<std::endl;
}


template <class Dist_t> 
void DPMM<Dist_t>::sampleParameters()
{ 
  components_.clear();
  parameters_.clear();

  for (uint32_t k=0; k<K_; ++k)
  {
    vector<int> x_k_index;
    for (uint32_t ii = 0; ii<N_; ++ii)
    {
      if (z_[ii] == k) x_k_index.push_back(ii); 
    }
    MatrixXd x_k(x_k_index.size(), x_.cols()); 
    x_k = x_(x_k_index, all);

    components_.push_back(H_.posterior(x_k));
    parameters_.push_back(components_[k].sampleParameter());
  }
}

template <class Dist_t> 
void DPMM<Dist_t>::sampleParameters(const uint32_t index_i, const uint32_t index_j)
{
  components_.clear();
  parameters_.clear();

  vector<int> x_i_index;
  vector<int> x_j_index;

  for (uint32_t ii = 0; ii<indexList_.size(); ++ii)  //To-do: This can be combined with sampleCoefficients
  {
    if (z_[indexList_[ii]]==z_[index_i]) x_i_index.push_back(ii); 
    else x_j_index.push_back(ii); 
  }
  MatrixXd x_i(x_i_index.size(), x_.cols()); 
  MatrixXd x_j(x_j_index.size(), x_.cols()); 


  //`````testing``````````````
  // std::cout << x_i_index.size() << std::endl << x_j_index.size() << std::endl;
  //`````testing``````````````


  x_i = x_(x_i_index, all);
  x_j = x_(x_j_index, all);

  components_.push_back(H_.posterior(x_i));
  components_.push_back(H_.posterior(x_j));
  parameters_.push_back(components_[0].sampleParameter());
  parameters_.push_back(components_[1].sampleParameter());
}


template <class Dist_t> 
void DPMM<Dist_t>::sampleCoefficientsParameters(const uint32_t index_i, const uint32_t index_j)
{
  vector<int> x_i_index;
  vector<int> x_j_index;
  #pragma omp parallel for num_threads(8) schedule(dynamic,100)
  for (uint32_t ii = 0; ii<indexList_.size(); ++ii) 
  {
    if (z_[indexList_[ii]]==z_[index_i]) x_i_index.push_back(ii); 
    else x_j_index.push_back(ii); 
  }
  MatrixXd x_i(x_i_index.size(), x_.cols()); 
  MatrixXd x_j(x_j_index.size(), x_.cols()); 
  x_i = x_(x_i_index, all);
  x_j = x_(x_j_index, all);
  
  components_.clear();
  parameters_.clear();
  components_.push_back(H_.posterior(x_i));
  components_.push_back(H_.posterior(x_j));
  parameters_.push_back(components_[0].sampleParameter());
  parameters_.push_back(components_[1].sampleParameter());
  

  VectorXi Nk(2);
  Nk(0) = x_i_index.size();
  Nk(1) = x_j_index.size();


  // //`````testing``````````````
  // std::cout << Nk <<std::endl;
  // //`````testing``````````````


  VectorXd Pi(2);
  for (uint32_t k=0; k<2; ++k)
  {
    boost::random::gamma_distribution<> gamma_(Nk(k), 1);
    Pi(k) = gamma_(*pRndGen_);
  }
  Pi_ = Pi / Pi.sum();


  // //`````testing``````````````
  // std::cout << Pi_ <<std::endl;  
  // //`````testing``````````````
}



template <class Dist_t> 
void DPMM<Dist_t>::sampleLabels(const uint32_t index_i, const uint32_t index_j)
{
  uint32_t z_i = z_[index_i];
  uint32_t z_j = z_[index_j];
  boost::random::uniform_01<> uni_;    //maybe put in constructor?
  #pragma omp parallel for num_threads(8) schedule(static)
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
    double uni_draw = uni_(*this->pRndGen_);
    if (uni_draw < prob[0]) z_[indexList_[i]] = z_i;
    else z_[indexList_[i]] = z_j;
  }
  z_[index_i] = z_i;
  z_[index_j] = z_j;
}


template <class Dist_t> 
void DPMM<Dist_t>::sampleLabels()
{
  #pragma omp parallel for num_threads(8) schedule(static)
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
    double uni_draw = uni_(*this->pRndGen_);
    uint32_t k = 0;
    while (prob[k] < uni_draw) k++;
    z_[i] = k;
  }
}



//   vector<int> i_array = generateRandom(N_);
//   for(uint32_t j=0; j<N_; ++j)
//   // for(uint32_t i=0; i<N_; ++i)
//   {
//     int i = i_array[j];
//     // cout << "number of data point: " << i << endl;
//     VectorXi Nk(K_);
//     Nk.setZero();
//     for(uint32_t ii=0; ii<N_; ++ii)
//     {
//       if (ii != i) Nk(z_(ii))++;
//     }
//     // cout<< Nk << endl;
//     VectorXd pi(K_+1); 
//     // VectorXd pi(K_); 

//     VectorXd x_i;
//     x_i = x_(i, all); //current data point x_i

//     // #pragma omp parallel for
//     for (uint32_t k=0; k<K_; ++k)
//     { 
//       vector<int> x_k_index;
//       for (uint32_t ii = 0; ii<N_; ++ii)
//       {
//         if (ii!= i && z_[ii] == k) x_k_index.push_back(ii); 
//       }
//       // if (x_k_index.empty()) 
//       // cout << "no index" << endl;
//       // cout << "im here" <<endl;


//       MatrixXd x_k(x_k_index.size(), x_.cols()); 
//       x_k = x_(x_k_index, all);
//       // cout << "x_i" << x_i << endl;
//       // cout << "x_k" << x_k << endl;
//       // cout << "component:" <<k  <<endl;
//       // cout << x_k << endl;
//       // cout << Nk(k) << endl;
//       if (Nk(k)!=0)
//       pi(k) = log(Nk(k))-log(N_+alpha_) + parameters_[k].logPosteriorProb(x_i, x_k);
//       else
//       pi(k) = - std::numeric_limits<float>::infinity();
//     }
//     pi(K_) = log(alpha_)-log(N_+alpha_) + H_.logProb(x_i);


//     // cout << pi <<endl;
//     // exit(1);


    
    
//     double pi_max = pi.maxCoeff();
//     pi = (pi.array()-(pi_max + log((pi.array() - pi_max).exp().sum()))).exp().matrix();
//     pi = pi / pi.sum();


//     for (uint32_t ii = 1; ii < pi.size(); ++ii){
//       pi[ii] = pi[ii-1]+ pi[ii];
//     }
   
//     boost::random::uniform_01<> uni_;   
//     boost::random::variate_generator<boost::random::mt19937&, 
//                            boost::random::uniform_01<> > var_nor(*H_.pRndGen_, uni_);
//     double uni_draw = var_nor();
//     uint32_t k = 0;
//     while (pi[k] < uni_draw) k++;
//     z_[i] = k;
//     this -> reorderAssignments();
//   }

// };


// template <class Dist_t>
// void DPMM<Dist_t>::reorderAssignments()
// { 
//   // cout << z_ << endl;
//   vector<uint8_t> rearrange_list;
//   for (uint32_t i=0; i<N_; ++i)
//   {
//     if (rearrange_list.empty()) rearrange_list.push_back(z_[i]);
//     // cout << *rearrange_list.begin() << endl;
//     // cout << *rearrange_list.end()  << endl;
//     std::vector<uint8_t>::iterator it;
//     it = find (rearrange_list.begin(), rearrange_list.end(), z_[i]);
//     if (it == rearrange_list.end())
//     {
//       rearrange_list.push_back(z_[i]);
//       // z_[i] = rearrange_list.end() - rearrange_list.begin();
//       z_[i] = rearrange_list.size() - 1;
//     }
//     else if (it != rearrange_list.end())
//     {
//       int index = it - rearrange_list.begin();
//       z_[i] = index;
//     }
//   }
//   K_ = z_.maxCoeff() + 1;
//   if(K_>parameters_.size())
//   parameters_.push_back(H_);
//   else if(K_<parameters_.size())
//   parameters_.pop_back();
//     // std::cout << "Element found in myvector: " << *it << '\n';
//     // else
//     // std::cout << "Element not found in myvector\n";
// }
//     // cout << z_ << endl;




// template <class Dist_t>
// void DPMM<Dist_t>::removeEmptyClusters()
// {
//   for(uint32_t k=parameters_.size()-1; k>=0; --k)
//   {
//     bool haveCluster_k = false;
//     for(uint32_t i=0; i<z_.size(); ++i)
//       if(z_(i)==k)
//       {
//         haveCluster_k = true;
//         break;
//       }
//     if (!haveCluster_k)
//     {
//       for (uint32_t i=0; i<z_.size(); ++i)
//         if(z_(i) >= k) z_(i) --;
//       parameters_.erase(parameters_.begin()+k);
//     }
//   }
// }

template class DPMM<NIW<double>>;

