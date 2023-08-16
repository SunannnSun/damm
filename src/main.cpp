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
    int num, dim;

    std::cin >> num >> dim;

    Eigen::MatrixXd Data(num, dim);
    std::cout << num << dim << std::endl;

    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < dim; ++j) {
            std::cin >> Data(i, j);
        }
    }



    /*---------------------------------------------------*/
    //------------------Arguments Parsing-----------------
    /*---------------------------------------------------*/

    // std::srand(seed);
    uint64_t seed = time(0);
    // uint64_t seed = 1671503159;
    boost::mt19937 rndGen(seed);
    

    cout << "Hello Parallel World" << endl;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help"                                 , "produce help message")
        ("iteration,t"  , po::value<int>()      , "number of iteration")
        ("init"         , po::value<int>()      , "number of initial clusters")
        ("base"         , po::value<int>()      , "Base type: 0 Euclidean, 1 Euclidean + directional")
        ("alpha,a"      , po::value<double>()   , "concentration value")
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


    int base = 0;
    if(vm.count("base")) base = static_cast<uint64_t>(vm["base"].as<int>());

  
    double nu, kappa;
    VectorXd mu(dim);
    // MatrixXd Sigma(dim/2+1, dim/2+1);  
    MatrixXd Sigma(dim, dim);  

    if(vm.count("params")){
        vector<double> params = vm["params"].as< vector<double> >();
        nu = params[0];
        kappa = params[1];
        for(int i=0; i<mu.rows(); ++i)
            mu(i) = params[2+i];
        for(int i=0; i<Sigma.rows(); ++i)
            for(int j=0; j<Sigma.cols(); ++j)
                Sigma(i,j) = params[2+mu.rows()+i*Sigma.cols()+j];
    }


    string logPath = "";
    if(vm.count("log")) logPath = vm["log"].as<string>();

    // string logPath_input = logPath + "input.csv";
    // ifstream  fin(logPath_input);
    // string line;
    // vector<vector<string> > parsedCsv;
    // while(getline(fin,line)){
    //     stringstream lineStream(line);
    //     string cell;
    //     vector<string> parsedRow;
    //     while(getline(lineStream,cell,','))  
    //         parsedRow.push_back(cell);
    //     parsedCsv.push_back(parsedRow);
    // }
    // fin.close();

    // MatrixXd Data(num, dim);              
    // for (int i=0; i<num; ++i)
    //     for (int j=0; j<dim; ++j)
    //         Data(i, j) = stod(parsedCsv[i][j]);

    
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
        for (int t=0; t<T; ++t){
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



    string logPath_output = logPath + "output.csv";
    ofstream fout(logPath_output.data(),ofstream::out);
    for (int i=0; i < z.size(); ++i)
        fout << z[i] << endl;
    fout.close();

    string logPath_logNum = logPath + "logNum.csv";
    ofstream fout_logNum(logPath_logNum.data(),ofstream::out);
    for (int i=0; i < logNum.size(); ++i)
        fout_logNum << logNum[i] << endl;
    fout_logNum.close();

    string logPath_logLogLik = logPath + "logLogLik.csv";
    ofstream fout_logLogLik(logPath_logLogLik.data(),ofstream::out);
    for (int i=0; i < logLogLik.size(); ++i)
        fout_logLogLik << logLogLik[i] << endl;
    fout_logLogLik.close();


    ofstream outputFile(logPath + "logZ.csv");
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