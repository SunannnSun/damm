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
{};


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
//       vector<int> indexList_k;
//       for (uint32_t ii = 0; ii<N_; ++ii)
//       {
//         if (ii!= i && z_[ii] == k) indexList_k.push_back(ii); 
//       }
//       // if (indexList_k.empty()) 
//       // cout << "no index" << endl;
//       // cout << "im here" <<endl;


//       MatrixXd x_k(indexList_k.size(), x_.cols()); 
//       x_k = x_(indexList_k, all);
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
//                            boost::random::uniform_01<> > var_nor(*H_.rndGen_, uni_);
//     double uni_draw = var_nor();
//     uint32_t k = 0;
//     while (pi[k] < uni_draw) k++;
//     z_[i] = k;
//     this -> reorderAssignments();
//   }

// };


template <class dist_t>
void DPMMDIR<dist_t>::reorderAssignments()
{ 
  // cout << z_ << endl;
  vector<uint8_t> rearrange_list;
  for (uint32_t i=0; i<N_; ++i)
  {
    if (rearrange_list.empty()) rearrange_list.push_back(z_[i]);
    // cout << *rearrange_list.begin() << endl;
    // cout << *rearrange_list.end()  << endl;
    vector<uint8_t>::iterator it;
    it = find (rearrange_list.begin(), rearrange_list.end(), z_[i]);
    if (it == rearrange_list.end())
    {
      rearrange_list.push_back(z_[i]);
      // z_[i] = rearrange_list.end() - rearrange_list.begin();
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



// template <class dist_t>
// void DPMM<dist_t>::removeEmptyClusters()
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

template class DPMMDIR<NIWDIR<double>>;

