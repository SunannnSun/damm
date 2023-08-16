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


int main(int argc, char **argv)
{   
    /*---------------------------------------------------*/
    //------------------Arguments Parsing-----------------
    /*---------------------------------------------------*/

    // std::srand(seed);
    uint64_t seed = time(0);
    // uint64_t seed = 1671503159;
    boost::mt19937 rndGen(seed);
    

    std::cout << "Hello Parallel World" << std::endl;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("base"         , po::value<int>()->required()      , "Base type: 0 damm, 1 pos, 2 pos+dir")
        ("init"         , po::value<int>()->required()      , "number of initial clusters")
        ("iter"         , po::value<int>()->required()      , "number of iteration")
        ("alpha"        , po::value<double>()->required()   , "concentration value")
        ("log"          , po::value<string>()->required()   , "path to log all the data")
    ;


    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }


    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    } 


    int base = vm["base"].as<int>();
    int init = vm["init"].as<int>();
    int iter = vm["iter"].as<int>();
    double alpha = vm["alpha"].as<double>();
    string logPath = vm["log"].as<string>();


    /*---------------------------------------------------*/
    //------------------Standard Input -------------------
    /*---------------------------------------------------*/
    int num, dim;
    std::cin >> num >> dim;

    Eigen::MatrixXd Data(num, dim);
    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < dim; ++j) {
            std::cin >> Data(i, j);
        }
    }

    double sigma_dir_0, nu, kappa;
    std::cin >> sigma_dir_0 >> nu >> kappa; 

    Eigen::VectorXd mu(dim);
    for(int i=0; i < dim; ++i)
        std::cin >> mu(i);
        
    Eigen::MatrixXd Sigma(dim, dim);  
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            std::cin >> Sigma(i,j);
        }
    }

    /*---------------------------------------------------*/
    //----------------------Sampler----------------------
    /*---------------------------------------------------*/

    VectorXi z;
    vector<VectorXi> logZ;
    vector<int> logNum;
    vector<double> logLogLik;

    if (base==0)  {
        NIW<double> niw(Sigma, mu, nu, kappa, rndGen);
        DPMM<NIW<double>> dpmm(Data, init, alpha, niw, rndGen);
        for (int t=0; t<iter; ++t){
            std::cout<<"------------ t="<<t<<" -------------"<<std::endl;
            // if (dpmm.sampleLabelsCollapsed())
            //     break;
            dpmm.sampleCoefficientsParameters();
            // dpmm.sampleLabelsCollapsedParallel();
            dpmm.sampleLabels();
            dpmm.reorderAssignments();
            dpmm.updateIndexLists();
            std::cout << "Number of components: " << dpmm.K_ << std::endl;
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
        DPMMDIR<NIWDIR<double>> dpmmDir(Data, init, alpha, niwDir, rndGen);
        for (int t=1; t<iter+1; ++t)    {
            std::cout<<"------------ t="<<t<<" -------------"<<std::endl;
            
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
            std::cout << "Number of components: " << dpmmDir.K_ << endl;
        }
        
        // NIW<double> H_NIW = * niwDir.NIW_ptr;  
        // DPMM<NIW<double>> dpmm(dpmmDir.x_, dpmmDir.z_, dpmmDir.alpha_, H_NIW, dpmmDir.rndGen_);
        // for (int t=0; t<0; ++t){
        //     std::cout<<"------------ t="<<t+T<<" -------------"<<endl;
        //     dpmm.sampleCoefficientsParameters();
        //     dpmm.sampleLabels();
        //     dpmm.reorderAssignments();
        //     dpmm.updateIndexLists();
        //     std::cout << "Number of components: " << dpmmDir.K_ << endl;
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