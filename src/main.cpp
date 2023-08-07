#include <iostream>
#include <fstream>
#include <limits>

#include <Eigen/Dense>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/program_options.hpp>

#include "niw.hpp"
#include "niwDir.hpp"
#include "dpmm.hpp"
#include "dpmmDir.hpp"



namespace po = boost::program_options;
using namespace std;
using namespace Eigen;


int main(int argc, char **argv)
{   

    /*---------------------------------------------------*/
    //------------------Arguments Parsing-----------------
    /*---------------------------------------------------*/

    // std::srand(seed);
    // if(vm.count("seed"))
    // seed = static_cast<uint64_t>(vm["seed"].as<int>());
    uint64_t seed = time(0);
    // uint64_t seed = 1671503159;
    boost::mt19937 rndGen(seed);
    

    cout << "Hello Parallel World" << endl;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help"                                 , "produce help message")
        ("number,n"     , po::value<int>()      , "number of data")
        ("dimension,m"  , po::value<int>()      , "dimension of data")
        ("output,o"     , po::value<string>()   , "path to output dataset .csv file: rows: dimensions; cols: numbers")
        ("iteration,t"  , po::value<int>()      , "number of iteration")
        ("alpha,a"      , po::value<double>()   , "concentration value")
        ("init"         , po::value<int>()      , "number of initial clusters")
        ("base"         , po::value<int>()      , "Base type: 0 Euclidean, 1 Euclidean + directional")
        ("params,p"     , po::value< vector<double> >()->multitoken(), "hyperparameters")
        ("log"          , po::value<string>()   , "path to log all the data")
    ;

    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);   


    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    } 


    int T = 0;
    if (vm.count("iteration")) T = vm["iteration"].as<int>();


    double alpha = 0;
    if(vm.count("alpha")) alpha = vm["alpha"].as<double>();
    assert(alpha != 0);

    
    int init_cluster = 0;
    if (vm.count("init")) init_cluster = vm["init"].as<int>();
    assert(init_cluster != 0);


    int num = 0;
    if (vm.count("number")) num = vm["number"].as<int>();
    assert(num != 0);


    int dim = 0;
    if (vm.count("number")) dim = vm["dimension"].as<int>();
    assert(dim != 0);


    int base = 0;
    if(vm.count("base")) base = static_cast<uint64_t>(vm["base"].as<int>());

  
    double nu, kappa;
    VectorXd mu(dim);
    MatrixXd Sigma(dim/2+1, dim/2+1);  
    if(vm.count("params")){
        vector<double> params = vm["params"].as< vector<double> >();
        nu = params[0];
        kappa = params[1];
        for(uint8_t i=0; i<mu.rows(); ++i)
            mu(i) = params[2+i];
        for(uint8_t i=0; i<Sigma.rows(); ++i)
            for(uint8_t j=0; j<Sigma.cols(); ++j)
                Sigma(i,j) = params[2+mu.rows()+i*Sigma.cols()+j];
    }


    MatrixXd Data(num, dim);              
    string pathIn ="";
    if(vm.count("log"))
        pathIn = vm["log"].as<string>() + "input.csv";
    if (!pathIn.compare("")){
        cout<<"please specify an input dataset"<<endl;
        return 1;
    }
    ifstream  fin(pathIn);
    string line;
    vector<vector<string> > parsedCsv;
    while(getline(fin,line)){
        stringstream lineStream(line);
        string cell;
        vector<string> parsedRow;
        while(getline(lineStream,cell,','))  
            parsedRow.push_back(cell);
        parsedCsv.push_back(parsedRow);
    }
    fin.close();
    for (uint32_t i=0; i<num; ++i)
        for (uint32_t j=0; j<dim; ++j)
            Data(i, j) = stod(parsedCsv[i][j]);

    
    /*---------------------------------------------------*/
    //----------------------Sampler----------------------
    /*---------------------------------------------------*/

    VectorXi z;
    vector<VectorXi> logZ;
    vector<int> logNum;
    vector<double> logLogLik;

    if (base==0)  {
        NIW<double> niw(Sigma, mu, nu, kappa, rndGen);
        DPMM<NIW<double>> dpmm(Data, init_cluster, alpha, niw, rndGen);
        for (uint32_t t=0; t<T; ++t){
            cout<<"------------ t="<<t<<" -------------"<<endl;
            // if (dpmm.sampleLabelsCollapsed())
            //     break;
            dpmm.sampleCoefficientsParameters();
            // dpmm.sampleLabelsCollapsedParallel();
            dpmm.sampleLabels();
            dpmm.reorderAssignments();
            dpmm.updateIndexLists();
            cout << "Number of components: " << dpmm.K_ << endl;
        }
        z = dpmm.getLabels();
        logZ.push_back(z);
        logZ        = dpmm.logZ_;
        logNum      = dpmm.logNum_;
        logLogLik   = dpmm.logLogLik_;
    }
    else if (base==1){
        boost::random::uniform_int_distribution<> uni(0, 3);  
        NIWDIR<double> niwDir(Sigma, mu, nu, kappa, rndGen);
        DPMMDIR<NIWDIR<double>> dpmmDir(Data, init_cluster, alpha, niwDir, rndGen);
        for (int t=1; t<T+1; ++t)    {
            cout<<"------------ t="<<t<<" -------------"<<endl;
            
            // vector<vector<int>> indexLists = dpmmDir.getIndexLists();
            // for (int l=0; l<indexLists.size(); ++l) 
            //     dpmmDir.splitProposal(indexLists[l]);
            // dpmmDir.updateIndexLists();
            if (t%30==0 && t> 15 && t<150){
                vector<vector<int>> indexLists = dpmmDir.getIndexLists();
                for (int l=0; l<indexLists.size(); ++l) 
                    dpmmDir.splitProposal(indexLists[l]);
                dpmmDir.updateIndexLists();
            }
            else if (t%3==0 && t>30 && t<175){ 
                vector<vector<int>> indexLists = dpmmDir.getIndexLists();
                vector<array<int, 2>>  mergeIndexLists = dpmmDir.computeSimilarity(int(dpmmDir.K_), uni(rndGen));
                for (int i =0; i < mergeIndexLists.size(); ++i){
                    if (!dpmmDir.mergeProposal(indexLists[mergeIndexLists[i][0]], indexLists[mergeIndexLists[i][1]]))
                        break;
                }
                dpmmDir.updateIndexLists();
            }
            else{
                dpmmDir.sampleCoefficientsParameters();
                dpmmDir.sampleLabels();
                dpmmDir.reorderAssignments();
                dpmmDir.updateIndexLists();
            }
            cout << "Number of components: " << dpmmDir.K_ << endl;
        }
        
        // NIW<double> H_NIW = * niwDir.NIW_ptr;  
        // DPMM<NIW<double>> dpmm(dpmmDir.x_, dpmmDir.z_, dpmmDir.alpha_, H_NIW, dpmmDir.rndGen_);
        // for (int t=0; t<0; ++t){
        //     cout<<"------------ t="<<t+T<<" -------------"<<endl;
        //     dpmm.sampleCoefficientsParameters();
        //     dpmm.sampleLabels();
        //     dpmm.reorderAssignments();
        //     dpmm.updateIndexLists();
        //     cout << "Number of components: " << dpmmDir.K_ << endl;
        // }
        // z           = dpmm.getLabels();     

        z           = dpmmDir.getLabels();
        logZ        = dpmmDir.logZ_;
        logNum      = dpmmDir.logNum_;
        logLogLik   = dpmmDir.logLogLik_;
    }



    /*---------------------------------------------------*/
    //------------------Export the Output-----------------
    /*---------------------------------------------------*/


    string pathOut;
    if(vm.count("log")) pathOut = vm["log"].as<string>();


    string pathOut_output = pathOut + "output.csv";
    ofstream fout(pathOut_output.data(),ofstream::out);
    for (uint16_t i=0; i < z.size(); ++i)
        fout << z[i] << endl;
    fout.close();

    string pathOut_logNum = pathOut + "logNum.csv";
    ofstream fout_logNum(pathOut_logNum.data(),ofstream::out);
    for (uint16_t i=0; i < logNum.size(); ++i)
        fout_logNum << logNum[i] << endl;
    fout_logNum.close();

    string pathOut_logLogLik = pathOut + "logLogLik.csv";
    ofstream fout_logLogLik(pathOut_logLogLik.data(),ofstream::out);
    for (uint16_t i=0; i < logLogLik.size(); ++i)
        fout_logLogLik << logLogLik[i] << endl;
    fout_logLogLik.close();



    // Populate the vector with Eigen::VectorXd elements
    // ... (add your data)

    ofstream outputFile(pathOut + "logZ.csv");
    if (outputFile.is_open()) {
        for (const auto& vector : logZ) {
            outputFile << vector.transpose() << "\n";
        }
        outputFile.close();
        std::cout << "Data exported successfully.\n";
    } else {
        std::cout << "Failed to open the file.\n";
    }

        
    return 0;
}   