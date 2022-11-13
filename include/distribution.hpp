#pragma once

#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>


typedef boost::mt19937 rng_t;

template<typename T>
class Distribution
{
public:
  Distribution(rng_t* ptr_rng) : ptr_rng_(ptr_rng)
  {};
  virtual ~Distribution()
  {};

  rng_t ptr_rng_;
private:
};