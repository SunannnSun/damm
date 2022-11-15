/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>, Randi Cabezas <rcabezas@csail.mit.edu>                    
 * Licensed under the MIT license. See the license file LICENSE.                
 */
// #include <stdint.h>

#include <Eigen/Dense>
// #include <boost/shared_ptr.hpp>

//using namespace boost;
// using boost::shared_ptr;

typedef Eigen::Matrix<uint32_t,Eigen::Dynamic,1> VectorXu;
typedef Eigen::Matrix<uint32_t,Eigen::Dynamic,Eigen::Dynamic> MatrixXu;

// shared pointer typedefs
// typedef shared_ptr<Eigen::MatrixXd> spMatrixXd;
// typedef shared_ptr<Eigen::MatrixXf> spMatrixXf;

// typedef shared_ptr<Eigen::VectorXd> spVectorXd;
// typedef shared_ptr<Eigen::VectorXf> spVectorXf;
// typedef shared_ptr<Eigen::VectorXi> spVectorXi;
// typedef shared_ptr<VectorXu> spVectorXu;

#ifndef PI
#  define PI 3.141592653589793
#endif
#define LOG_PI 1.1447298858494002
#define LOG_2 0.69314718055994529
#define LOG_2PI 1.8378770664093453

// #ifndef NDEBUG
// #   define ASSERT(condition, message) \
//   do { \
//     if (! (condition)) { \
//       std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
//       << " line " << __LINE__ << ": " << message << " " << std::endl; \
//       assert(false); \
//       std::exit(EXIT_FAILURE); \
//     } \
//   } while (false)
// #else
// #   define ASSERT(condition, message) do { } while (false)
// #endif
