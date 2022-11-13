#include <iostream>
#include <fstream>
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
        ("number,n", po::value<int>(), "number of data points")
        ("dimension,m", po::value<int>(), "dimension of data points")
        ("input,i", po::value<string>(), "path to input dataset .csv file rows: dimensions; cols: numbers")
        ("onput,o", po::value<string>(), "path to output dataset .csv file rows: dimensions; cols: numbers")
        ("iteration,t", po::value<int>(), "Numer of Sampler Iteration")
        ("params,p", po::value< vector<double> >()->multitoken(), "parameters of the base measure")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    uint16_t num = 0;
    if (vm.count("number")) num = vm["number"].as<int>();

    uint8_t dim = 0;
    if (vm.count("dimension")) dim = vm["dimension"].as<int>();


    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    uint16_t T = 0;
    if (vm.count("iteration")) {
        cout << "Number of sampler iteration was set to " << vm["iteration"].as<int>() << ".\n";
        T = vm["iteration"].as<int>();
    } 

    double nu = 0;
    double kappa = 0;
    Eigen::MatrixXd Delta(dim,dim);
    Eigen::VectorXd theta(dim);
    if(vm.count("params")){
        cout << "Parameters received.\n";
        vector<double> params = vm["params"].as< vector<double> >();
        cout<<"params length="<<params.size()<<endl;
        nu = params[0];
        kappa = params[1];
        for(uint8_t i=0; i<dim; ++i)
            theta(i) = params[4+i];
        for(uint8_t i=0; i<dim; ++i)
            for(uint8_t j=0; j<dim; ++j)
                Delta(i,j) = params[4+dim+i+dim*j];
        // cout <<"nu="<<nu<<endl;
        // cout <<"kappa="<<kappa<<endl;
        // cout <<"theta="<<theta<<endl;
        // cout <<"Delta="<<Delta<<endl;
    }

    string input_path ="";
    Eigen::MatrixXd data(num, dim);
    if (vm.count("input")) input_path = vm["input"].as<string>();
    if (!input_path.compare("")){
        cout<<"please specify an input dataset"<<endl;
        exit(1);
    }
    else{
        cout<<"loading data from "<<input_path<<endl;
        fin.open("reportcard.csv", ios::in);

    }

    string output_path ="";
    if (vm.count("output")) output_path = vm["output"].as<string>();

}