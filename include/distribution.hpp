/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */

#pragma once

#include <Eigen/Dense>
#include "global.hpp"
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/special_functions/gamma.hpp>



template<typename T>
class Distribution
{
public:
  Distribution(boost::mt19937* pRndGen): pRndGen_(pRndGen)
  {};
  ~Distribution()
  {};

  boost::mt19937* pRndGen_;
private:
};

// inline double lgamma_mult(double x,uint32_t p)
// {
//   assert(x+0.5*(1.-p) > 0.);
//   double lgam_p = p*(p-1.)*0.25*LOG_PI;
//   for (uint32_t i=1; i<p+1; ++i)
//   {
// //    cout<<"digamma_mult of "<<(x + (1.0-double(i))/2)<<" = "<<digamma(x + (1.0-double(i))/2)<<endl;
//     lgam_p += boost::math::lgamma(x + 0.5*(1.0-double(i)));
//   }
//   return lgam_p;
// }