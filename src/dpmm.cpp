#include <iostream>
#include <boost/program_options.hpp>
#include <Eigen/Dense>
 

namespace po = boost::program_options;
using namespace std;


int main(int argc, char **argv)
{
    cout << "Hello World" << endl;

    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("iteration,t", po::value<int>(), "Numer of Sampler Iteration")
        ("params,p", po::value< vector<double> >()->multitoken(), "parameters of the base measure")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    if (vm.count("iteration")) {
        cout << "Number of sampler iteration was set to " << vm["iteration"].as<int>() << ".\n";
        uint16_t T = vm["iteration"].as<int>();
    } 


    uint32_t D=7;
    Eigen::MatrixXd Delta(D,D);
    Eigen::VectorXd theta(D);
    double nu = 10.0;
    double kappa = 10.0;
    if(vm.count("params"))
    {
        cout << "Parameters received.\n";
        vector<double> params = vm["params"].as< vector<double> >();
        cout<<"params length="<<params.size()<<" D="<<D<<endl;
        nu = params[0];
        kappa = params[1];
        for(uint32_t i=0; i<D; ++i)
            theta(i) = params[2+i];
        for(uint32_t i=0; i<D; ++i)
            for(uint32_t j=0; j<D; ++j)
            Delta(i,j) = params[2+D+i+D*j];
        cout <<"nu="<<nu<<endl;
        cout <<"kappa="<<kappa<<endl;
        cout <<"theta="<<theta<<endl;
        cout <<"Delta="<<Delta<<endl;
    }
}