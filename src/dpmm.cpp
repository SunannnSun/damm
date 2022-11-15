#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <Eigen/Dense>
 
#include "niw.hpp"
// #include "normal.hpp"
#include "dpmm.hpp"



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
        ("alpha,a", po::value<double>(), "Concentration value")
        ("params,p", po::value< vector<double> >()->multitoken(), "parameters of the base measure")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    


    uint64_t seed = time(0);
    if(vm.count("seed"))
        seed = static_cast<uint64_t>(vm["seed"].as<int>());
    boost::mt19937 rndGen(seed);
    std::srand(seed);

    // cout << rndGen << endl;
    // cout << rndGen& << endl;


    double alpha = 1;
    if(vm.count("alpha")) alpha = vm["alpha"].as<double>();

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
    Eigen::MatrixXd sigma(dim,dim);
    Eigen::VectorXd mu(dim);
    if(vm.count("params")){
        cout << "Parameters received.\n";
        vector<double> params = vm["params"].as< vector<double> >();
        cout<<"params length="<<params.size()<<endl;
        nu = params[0];
        kappa = params[1];
        for(uint8_t i=0; i<dim; ++i)
            mu(i) = params[2+i];
        for(uint8_t i=0; i<dim; ++i)
            for(uint8_t j=0; j<dim; ++j)
                sigma(i,j) = params[2+dim+i+dim*j];
        // cout <<"nu="<<nu<<endl;
        // cout <<"kappa="<<kappa<<endl;
        // cout <<"theta="<<theta<<endl;
        // cout <<"Delta="<<Delta<<endl;
    }
    // cout << nu << endl;
    NIW<double> niw(sigma, mu, nu, kappa, &rndGen);
    // cout << niw.nu_ << endl;
    // DPMM<NIW<double>>* ptr_dpmm;
    DPMM<NIW<double>> dpmm(alpha, niw);
    // ptr_dpmm = &dpmm;


    // shared_ptr<Eigen::MatrixXd> spx(new Eigen::MatrixXd(num, dim));
    // Eigen::MatrixXd& data(*spx);

    Eigen::MatrixXd data(num, dim);

    string pathIn ="";
    if(vm.count("input")) pathIn = vm["input"].as<string>();
    if (!pathIn.compare(""))
    {
        cout<<"please specify an input dataset"<<endl;
        exit(1);
    }
    else
    {
        ifstream  fin(pathIn);
        string line;
        vector<vector<string> > parsedCsv;
        while(getline(fin,line))
        {
            stringstream lineStream(line);
            string cell;
            vector<string> parsedRow;
            while(getline(lineStream,cell,','))
            {
                parsedRow.push_back(cell);
            }
            parsedCsv.push_back(parsedRow);
        }
    fin.close();
    for (uint32_t i=0; i<num; ++i)
        for (uint32_t j=0; j<dim; ++j)
            data(i, j) = stod(parsedCsv[i][j]);
    }
    // cout << data<< endl;
    // ptr_dpmm->initialize(data);
    dpmm.initialize(data);

    // assert(dpmm!=NULL);
    // dpmm->initialize(x);
    for (uint32_t t=0; t<T; ++t)
    {
      cout<<"------------ t="<<t<<" -------------"<<endl;
      
      dpmm.sampleLabels();
    }
}
