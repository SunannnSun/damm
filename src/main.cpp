#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <Eigen/Dense>
#include "niw.hpp"
#include "dpmm.hpp"



namespace po = boost::program_options;
using namespace std;
using namespace Eigen;


int main(int argc, char **argv)
{   
    cout << "Hello Parallel World" << endl;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("number,n", po::value<int>(), "number of data points")
        ("dimension,m", po::value<int>(), "dimension of data points")
        ("input,i", po::value<string>(), "path to input dataset .csv file rows: dimensions; cols: numbers")
        ("output,o", po::value<string>(), "path to output dataset .csv file rows: dimensions; cols: numbers")
        ("iteration,t", po::value<int>(), "Numer of Sampler Iteration")
        ("alpha,a", po::value<double>(), "Concentration value")
        ("init", po::value<int>(), "Number of initial clusters")
        ("params,p", po::value< vector<double> >()->multitoken(), "parameters of the base measure")
    ;

    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);   


    if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
    } 


    // uint64_t seed = time(0);
    uint64_t seed = 1671503159;
    if(vm.count("seed"))
        seed = static_cast<uint64_t>(vm["seed"].as<int>());
    boost::mt19937 rndGen(seed);
    std::srand(seed);


    int T = 0;
    if (vm.count("iteration")) T = vm["iteration"].as<int>();


    double alpha = 0;
    if(vm.count("alpha")) alpha = vm["alpha"].as<double>();


    int num = 0;
    if (vm.count("number")) num = vm["number"].as<int>();
    assert(num != 0);


    int dim = 0;
    if (vm.count("dimension")) dim = vm["dimension"].as<int>();
    assert(dim != 0);


    int init_cluster = 0;
    if (vm.count("init")) init_cluster = vm["init"].as<int>();


    cout << "Iteration: " << T <<  "; Concentration: " << alpha << endl
         <<"Number: " << num << "; Dimension:" << dim <<endl;


    double nu;
    double kappa;
    MatrixXd sigma(dim,dim);
    VectorXd mu(dim);
    if(vm.count("params")){
        // cout << "Parameters received.\n";
        vector<double> params = vm["params"].as< vector<double> >();
        // cout<<"params length="<<params.size()<<endl;
        nu = params[0];
        kappa = params[1];
        for(uint8_t i=0; i<dim; ++i)
            mu(i) = params[2+i];
        for(uint8_t i=0; i<dim; ++i)
            for(uint8_t j=0; j<dim; ++j)
                sigma(i,j) = params[2+dim+i+dim*j];
    }


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
    
 
    NIW<double> niw(sigma, mu, nu, kappa, &rndGen);
    DPMM<NIW<double>> dpmm(data, init_cluster, alpha, niw, &rndGen);
    // dpmm.initialize(data, init_cluster);


    for (uint32_t t=0; t<T; ++t)
    {
        cout<<"------------ t="<<t<<" -------------"<<endl;
        cout << "Number of components: " << dpmm.K_ << endl;
        dpmm.splitProposal(10, 100);
        // std::cout<<dpmm.z_<<std::endl;



        // dpmm.sampleCoefficients();
        // dpmm.sampleParameters();
        // dpmm.sampleLabels();
    }


    const VectorXi& z = dpmm.getLabels();
    string pathOut;
    if(vm.count("output")) pathOut = vm["output"].as<string>();
    if (!pathOut.compare(""))
    {
        cout<<"please specify an output data file"<<endl;
        exit(1);
    }
    else cout<<"Output to "<<pathOut<<endl;
    ofstream fout(pathOut.data(),ofstream::out);
    for (uint16_t i=0; i < z.size(); ++i)
        fout << z[i] << endl;
    fout.close();

    return 0;
}   