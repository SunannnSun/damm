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
using namespace Eigen;


int main(int argc, char **argv)
{   

    // cout << "Hello World" << endl;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("number,n", po::value<int>(), "number of data points")
        ("dimension,m", po::value<int>(), "dimension of data points")
        ("input,i", po::value<string>(), "path to input dataset .csv file rows: dimensions; cols: numbers")
        ("output,o", po::value<string>(), "path to output dataset .csv file rows: dimensions; cols: numbers")
        ("iteration,t", po::value<int>(), "Numer of Sampler Iteration")
        ("alpha,a", po::value<double>(), "Concentration value")
        ("params,p", po::value< vector<double> >()->multitoken(), "parameters of the base measure")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);   

    if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
    } 

    uint64_t seed = time(0);
    if(vm.count("seed"))
        seed = static_cast<uint64_t>(vm["seed"].as<int>());
    boost::mt19937 rndGen(seed);
    // std::srand(seed);


    double alpha = 1;
    if(vm.count("alpha")) 
    {
        alpha = vm["alpha"].as<double>();
        cout << "Concentration factor set to " << vm["alpha"].as<double>() << ".\n";
    }
    int num = 0;
    if (vm.count("number")) num = vm["number"].as<int>();

    int dim = 0;
    if (vm.count("dimension")) dim = vm["dimension"].as<int>();

    int T = 0;
    if (vm.count("iteration")) 
    {
        cout << "Sampler iteration set to " << vm["iteration"].as<int>() << ".\n";
        T = vm["iteration"].as<int>();
    } 

    double nu;
    double kappa;
    MatrixXd sigma(dim,dim);
    VectorXd mu(dim);
    if(vm.count("params")){
        // cout << "Parameters received.\n";
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
        // cout <<"Sigma="<<sigma<<endl;
    }
    // cout << nu << endl;
    NIW<double> niw(sigma, mu, nu, kappa, &rndGen);

    // Log Probability Debugging Test Block
    // VectorXd x_tilde {{10, 10}};
    // cout << niw.nu_ << niw.kappa_ << niw.mu_ << niw.sigma_ << endl;
    // cout << niw.logProb(x_tilde) << endl;
    // cout << niw.nu_ << endl;


    // DPMM<NIW<double>>* ptr_dpmm;
    DPMM<NIW<double>> dpmm(alpha, niw);

    // VectorXd aaa(3);
    // aaa = {1, 3};
    // cout << niw.logProb(aaa) << endl;
    // ptr_dpmm = &dpmm;


    // shared_ptr<Eigen::MatrixXd> spx(new Eigen::MatrixXd(num, dim));
    // Eigen::MatrixXd& data(*spx);

    MatrixXd data(num, dim);

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

    dpmm.initialize(data);

    for (uint32_t t=0; t<T; ++t)
    {
      cout<<"------------ t="<<t<<" -------------"<<endl;
      dpmm.sampleLabels();
    }


    const VectorXi& z = dpmm.getLabels();
    string pathOut;
    if(vm.count("output")) 
    pathOut = vm["output"].as<string>();
    if (!pathOut.compare(""))
    {
        cout<<"please specify an output data file"<<endl;
        exit(1);
    }
    else cout<<"output to "<<pathOut<<endl;
    ofstream fout(pathOut.data(),ofstream::out);
    for (uint16_t i=0; i < z.size(); ++i)
        fout << z[i] << endl;
    fout.close();


    return 0;
}   
