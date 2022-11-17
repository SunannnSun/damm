// This file is not used, and only serves testing purpose
#include <iostream>
#include <Eigen/Core>
#include <limits>

// #include "karcher.hpp"

using namespace std;
using namespace Eigen;

int main ()
{
    int K_ = 10;
    int kk = 5;
    VectorXd pi(K_); 
    cout << pi << endl;
    for (int i=0; i<K_; ++i)
    {
        if (i!=kk)
        pi(i) = 1;
        else
        pi(i) = - std::numeric_limits<float>::infinity();
    }
    cout << pi << endl;
    double pi_max = pi.maxCoeff();
    pi = (pi.array()-(pi_max + log((pi.array() - pi_max).exp().sum()))).exp().matrix();
    pi = pi / pi.sum();
    cout << pi << endl;

    return 0;
}