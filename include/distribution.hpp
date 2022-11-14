/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */

#pragma once

#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>


template<typename T>
class Distribution
{
public:
  Distribution(boost::mt19937* pRndGen): pRndGen(pRndGen)
  {};
  ~Distribution()
  {};

  boost::mt19937* pRndGen;
private:
};