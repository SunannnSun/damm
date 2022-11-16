#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <Eigen/Dense>
#include "niw.hpp"
#include "niwDir.hpp"
#include "dpmm.hpp"



namespace po = boost::program_options;
using namespace std;
using namespace Eigen;


int main(int argc, char **argv)
{   
    cout << "Hello Directional World" << endl;

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
        ("base", po::value<int>(), "Base type: 0 euclidean, 1 euclidean + directional")
        ("params,p", po::value< vector<double> >()->multitoken(), "parameters of the base measure")
    ;

    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);   

    if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
    } 

    int base = 0;
    if(vm.count("base"))
        base = static_cast<uint64_t>(vm["base"].as<int>());

    uint64_t seed = time(0);
    if(vm.count("seed"))
        seed = static_cast<uint64_t>(vm["seed"].as<int>());
    boost::mt19937 rndGen(seed);
    // std::srand(seed);


    double alpha = 1;
    if(vm.count("alpha")) 
    {
        alpha = vm["alpha"].as<double>();
        cout << "Concentration factor: " << vm["alpha"].as<double>() << ".\n";
    }
    int num = 0;
    if (vm.count("number")) num = vm["number"].as<int>();
    assert(num != 0);
    cout << "Number of data point: " << num << endl;


    
    int dim = 0;
    if (vm.count("dimension")) dim = vm["dimension"].as<int>();
    // DIRECTIONAL
    if (base == 1) dim = dim/2+1;
    assert(dim != 0);
    cout << "Dimension of data: " << dim << endl;

    int T = 0;
    if (vm.count("iteration")) 
    {
        cout << "Sampler iteration: " << vm["iteration"].as<int>() << ".\n";
        T = vm["iteration"].as<int>();
    } 

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
        // cout <<"nu="<<nu<<endl;
        // cout <<"kappa="<<kappa<<endl;
        // cout <<"theta="<<theta<<endl;
        // cout <<"Sigma="<<sigma<<endl;
    }
    // cout << nu << endl;

    

  
    // Log Probability Debugging Test Block
    // VectorXd x_tilde {{0, 0}};
    // cout << niw.nu_ << niw.kappa_ << niw.mu_ << niw.sigma_ << endl;
    // cout << niw.logProb(x_tilde) << endl;
    // cout << niw.nu_ << endl;
    // Sufficient Statistics Test Block
    // MatrixXd x_kkk {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
    // cout << niw.logPosteriorProb(x_tilde, x_kkk) << endl;
    // cout << niw.scatter_ << endl;
    // cout << niw.nu_ << niw.kappa_ << niw.mu_ << niw.sigma_ << endl;
    // cout << niw.logProb(x_tilde) << endl;
    // cout << niw.nu_ << endl;

    // DPMM<NIW<double>>* ptr_dpmm;

    // if (base == 0)
    // DPMM<NIW<double>> dpmm(alpha, niw);
    // DPMM<NIWDIR<double>> dpmm(alpha, niw);
    // VectorXd x_tilde {{1, 1, 1}};
    // cout << niw.logProb(x_tilde) <<endl;



    // VectorXd aaa(3);
    // aaa = {1, 3};
    // cout << niw.logProb(aaa) << endl;
    // ptr_dpmm = &dpmm;


    // shared_ptr<Eigen::MatrixXd> spx(new Eigen::MatrixXd(num, dim));
    // Eigen::MatrixXd& data(*spx);

    int data_dim;
    if (base==1) data_dim = (dim-1)*2;
    else  data_dim = dim;

    MatrixXd data(num, data_dim);

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
        for (uint32_t j=0; j<data_dim; ++j)
            data(i, j) = stod(parsedCsv[i][j]);
    }
    
    // niw.karcherMean(x_tilde.transpose());
    // cout << niw.karcherMean(data(0, all)) << endl;

    // MatrixXd test_angle {{100, 1000, std::cos(0), std::sin(0)}, 
    //                      {100, 1000, std::cos(-PI/2), std::sin(-PI/2)}};

    // cout << niw.karcherMean(data({0, 1}, all)) << endl;
                     
    // cout << niw.karcherMean(data) << endl;



    // int init_cluster = 0;
    // if (vm.count("init")) init_cluster = vm["init"].as<int>();
    // dpmm.initialize(data, init_cluster);
    
    // DPMM<NIW<double>> dpmm();
    // DPMM<NIWDIR<double>> dpmmDir();
    // T = 1;
    if (base==0) 
    {
        NIW<double> niw(sigma, mu, nu, kappa, &rndGen);
        DPMM<NIW<double>> dpmm(alpha, niw);
        int init_cluster = 0;
        if (vm.count("init")) init_cluster = vm["init"].as<int>();
        dpmm.initialize(data, init_cluster);

        for (uint32_t t=0; t<T; ++t)
        {
            cout<<"------------ t="<<t<<" -------------"<<endl;
            cout << "Number of components: " << dpmm.K_ << endl;
            dpmm.sampleLabels();
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
    }


    else if (base==1) 
    {
        NIWDIR<double> niwDir(sigma, mu, nu, kappa, &rndGen);
        DPMM<NIWDIR<double>> dpmmDir(alpha, niwDir);
        int init_cluster = 0;
        if (vm.count("init")) init_cluster = vm["init"].as<int>();
        dpmmDir.initialize(data, init_cluster);
        for (uint32_t t=0; t<T; ++t)
        {
            cout<<"------------ t="<<t<<" -------------"<<endl;
            cout << "Number of components: " << dpmmDir.K_ << endl;
            dpmmDir.sampleLabels();
        }

        const VectorXi& z = dpmmDir.getLabels();
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
    }
    return 0;
}   
