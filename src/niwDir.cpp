#include "niwDir.hpp"
// #include "karcher.hpp"

// template<typename T>
// NIW<T>::NIW(){};


template<typename T>
NIWDIR<T>::NIWDIR(const Matrix<T,Dynamic,Dynamic>& sigma, 
  const Matrix<T,Dynamic,Dynamic>& mu, T nu,  T kappa, boost::mt19937 *pRndGen)
: Distribution<T>(pRndGen), sigma_(sigma), mu_(mu), nu_(nu), kappa_(kappa), dim_(mu.size())
{
  assert(sigma_.rows()==mu_.size()); 
  assert(sigma_.cols()==mu_.size());
};


template<typename T>
NIWDIR<T>::~NIWDIR()
{};


template<typename T>
T NIWDIR<T>::logPosteriorProb(const Vector<T,Dynamic>& x_i, const Matrix<T,Dynamic, Dynamic>& x_k)
{
  NIWDIR<T> posterior = this ->posterior(x_k);

  // std::cout << posterior.sigma_ <<std::endl;
  // std::cout << posterior.mu_ <<std::endl;
  

  return posterior.logProb(x_i, x_k);
  // return x_i.rows() + x_k.cols();
};

template<typename T>
NIWDIR<T> NIWDIR<T>::posterior(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  // std::cout << "im hereeee" <<std::endl;
  getSufficientStatistics(x_k);
  return NIWDIR<T>(
    sigma_+scatter_ + ((kappa_*count_)/(kappa_+count_))
      *(mean_-mu_)*(mean_-mu_).transpose(), 
    (kappa_*mu_+ count_*mean_)/(kappa_+count_),
    nu_+count_,
    kappa_+count_,
    this->pRndGen_);
};

template<typename T>
void NIWDIR<T>::getSufficientStatistics(const Matrix<T,Dynamic, Dynamic>& x_k)
{
  mean_.setZero(dim_);
	mean_(seq(0, dim_-2)) = x_k(all, (seq(0, dim_-2))).colwise().mean().transpose();
  // std::cout << "mean_" << mean_ <<std::endl;

  Matrix<T,Dynamic, Dynamic> x_k_mean;
  // // x_k_mean = x_k.rowwise() - mean.transpose();
  x_k_mean = x_k.rowwise() - x_k.colwise().mean();

  scatter_.setZero(dim_, dim_);
  scatter_(seq(0, dim_-2), seq(0, dim_-2)) = (x_k_mean.adjoint() * x_k_mean)(seq(0, dim_-2), seq(0, dim_-2));
  
  // std::cout << "scatter_" << scatter_ <<std::endl;

  count_ = x_k.rows();
};

template<typename T>
T NIWDIR<T>::logProb(const Matrix<T,Dynamic,1>& x_i)
{
  Matrix<T,Dynamic,1>  x_i_dir(dim_);
  x_i_dir.setZero();
  x_i_dir(seq(0,dim_-2)) = x_i(seq(0,dim_-2));

  T doF = nu_ - dim_ + 1.;
  Matrix<T,Dynamic,Dynamic> scaledSigma = sigma_*(kappa_+1.)/(kappa_*(nu_-dim_+1));                       
  // T logProb = 0;
  T logProb = boost::math::lgamma(0.5*(doF + dim_));  
  logProb -= boost::math::lgamma(0.5*(doF));
  logProb -= 0.5*dim_*log(doF);
  logProb -= 0.5*dim_*log(PI);
  logProb -= 0.5*log(scaledSigma.determinant());
  // logProb -= 0.5*((scaledSigma.eigenvalues()).array().log().sum()).real();
  logProb -= (0.5*(doF + dim_))
    *log(1.+ 1/doF*((x_i_dir-mu_).transpose()*scaledSigma.inverse()*(x_i_dir-mu_)).sum());
  // approximate using moment-matched Gaussian; Erik Sudderth PhD essay
  return logProb;
};


template<typename T>
T NIWDIR<T>::logProb(const Matrix<T,Dynamic,1>& x_i, const Matrix<T,Dynamic,Dynamic>& x_k)
{
  Matrix<T,Dynamic,1>  x_i_dir(dim_);
  x_i_dir.setZero();
  x_i_dir(seq(0,dim_-2)) = x_i(seq(0,dim_-2));

  // cout << x_i_dir << endl;

  Matrix<T,Dynamic,1>  x_i_dirrr(dim_-1); // just direction no position
  x_i_dirrr = x_i(seq(dim_-1, dim_));

  // cout << x_k << endl;
  // cout << karcherMean(x_k) << endl;

  x_i_dir(dim_-1) = rie_log(x_i_dirrr, karcherMean(x_k)).sum();


  // cout << x_i_dirrr << endl;

  // std::cout << x_i << std::endl;
  // std::cout << x_i_dir << std::endl;
  // std::cout << x_i << std::endl;
  // using multivariate student-t distribution; missing terms?
  // https://en.wikipedia.org/wiki/Multivariate_t-distribution
  // https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf pg.21
  T doF = nu_ - dim_ + 1.;
  Matrix<T,Dynamic,Dynamic> scaledSigma = sigma_*(kappa_+1.)/(kappa_*(nu_-dim_+1));   

      
  // scaledSigma(dim_-1, dim_-1) = 0.001;   
  // std::cout << scaledSigma << std::endl;           
  // T logProb = 0;
  T logProb = boost::math::lgamma(0.5*(doF + dim_));  
  logProb -= boost::math::lgamma(0.5*(doF));
  logProb -= 0.5*dim_*log(doF);
  logProb -= 0.5*dim_*log(PI);
  logProb -= 0.5*log(scaledSigma.determinant());
  // logProb -= 0.5*((scaledSigma.eigenvalues()).array().log().sum()).real();
  logProb -= (0.5*(doF + dim_))
    *log(1.+ 1/doF*((x_i_dir-mu_).transpose()*scaledSigma.inverse()*(x_i_dir-mu_)).sum());

  // approximate using moment-matched Gaussian; Erik Sudderth PhD essay
  return logProb;
};



// template<typename T>
// Matrix<T, Dynamic, 1> NIWDIR<T>::karcherMean(const Matrix<T,Dynamic, Dynamic>& x_k)
// {
//   float tolerance = 0.01;
//   T angle;
//   Matrix<T, Dynamic, 1> angle_sum(x_k.cols()/2);
//   Matrix<T, Dynamic, 1> x_tp(x_k.cols()/2);   // x in tangent plane
//   Matrix<T, Dynamic, 1> x(x_k.cols()/2);
//   Matrix<T, Dynamic, 1> p(x_k.cols()/2);
  
//   p = x_k(0, seq(x_k.cols()/2, last)).transpose();
//   if (x_k.rows() == 1) return p;

//   while (1)
//   { 
//     angle_sum.setZero();
//     for (int i=0; i<x_k.rows(); ++i)
//     {
//       x = x_k(i, seq(x_k.cols()/2, last)).transpose();
//       angle_sum = angle_sum + rie_log(p, x);
//     }
//     x_tp = 1. / x_k.rows() * angle_sum;
//     // cout << x_tp.norm() << endl;
//     if (x_tp.norm() < tolerance) 
//     {
//       return p;
//     }
//     p = rie_exp(p, x_tp);
//   }
// };

