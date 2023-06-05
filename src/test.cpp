#include "normal.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>


using namespace Eigen;


int main(){
    uint64_t seed = time(0);
    boost::mt19937 rndGen(seed);


    VectorXd mu(2, 1);
    mu << 1, 2;
    
    MatrixXd Sigma(2, 2);
    Sigma << 1, 0, 0, 1;
    
    VectorXd x(2, 1);
    x << 2, 1;
    

    Normal<double> normal(mu, Sigma, rndGen);

    // std::cout << normal.mu_;
    // std::cout << normal.Sigma_;
    std::cout << normal.prob(x) << std::endl;

    std::cout << normal.logProb(x) << std::endl;

    return 0;
}