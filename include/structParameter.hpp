#pragma once

#include <Eigen/Dense>


using namespace Eigen;


struct Parameter {
  VectorXd mu;
  MatrixXd sigma;
};