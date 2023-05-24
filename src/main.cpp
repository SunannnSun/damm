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
    // uint64_t seed = time(0);
    uint64_t seed = 1671503159;
    boost::mt19937 rndGen(seed);
    

    cout << "Hello Parallel World" << endl;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help"                                 , "produce help message")
        ("number,n"     , po::value<int>()      , "number of data")
        ("dimension,m"  , po::value<int>()      , "dimension of data")
        ("input,i"      , po::value<string>()   , "path to input dataset .csv file: rows: dimensions; cols: numbers")
        ("output,o"     , po::value<string>()   , "path to output dataset .csv file: rows: dimensions; cols: numbers")
        ("iteration,t"  , po::value<int>()      , "number of iteration")
        ("alpha,a"      , po::value<double>()   , "concentration value")
        ("init"         , po::value<int>()      , "number of initial clusters")
        ("base"         , po::value<int>()      , "Base type: 0 Euclidean, 1 Euclidean + directional")
        ("params,p"     , po::value< vector<double> >()->multitoken(), "hyperparameters")
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
    if(vm.count("input"))
        pathIn = vm["input"].as<string>();
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
    vector<int> logNum;
    vector<double> logLogLik;

    if (base==0)  {
        NIW<double> niw(Sigma, mu, nu, kappa, rndGen);
        DPMM<NIW<double>> dpmm(Data, init_cluster, alpha, niw, rndGen);
        for (uint32_t t=0; t<T; ++t){
            cout<<"------------ t="<<t<<" -------------"<<endl;

            if (t!=0 && t%50==0 && t<700){
                vector<vector<int>> indexLists = dpmm.getIndexLists();
                for (int l=0; l<indexLists.size(); ++l) 
                    dpmm.splitProposal(indexLists[l]);
                dpmm.updateIndexLists();
            }
            else if (t!=0 && t%51==0 && t<700){   
                vector<vector<int>> merge_indexLists = dpmm.computeSimilarity();
                dpmm.mergeProposal(merge_indexLists[0], merge_indexLists[1]);
                dpmm.updateIndexLists();
            }
            else{
                dpmm.sampleCoefficientsParameters();
                dpmm.sampleLabels();
                dpmm.reorderAssignments();
                dpmm.updateIndexLists();
            }
            cout << "Number of components: " << dpmm.K_ << endl;
        }
        z = dpmm.getLabels();
    }
    else if (base==1){
        NIWDIR<double> niwDir(Sigma, mu, nu, kappa, rndGen);
        DPMMDIR<NIWDIR<double>> dpmmDir(Data, init_cluster, alpha, niwDir, rndGen);
        for (uint32_t t=0; t<T; ++t)    {
            cout<<"------------ t="<<t<<" -------------"<<endl;
            
            if (t!=0 && t%10==0 && t<700){
                vector<vector<int>> indexLists = dpmmDir.getIndexLists();
                for (int l=0; l<indexLists.size(); ++l) 
                    dpmmDir.splitProposal(indexLists[l]);
                dpmmDir.updateIndexLists();
            }
            else{
                dpmmDir.sampleCoefficientsParameters();
                dpmmDir.sampleLabels();
                dpmmDir.reorderAssignments();
                dpmmDir.updateIndexLists();
            }
            if (t!=0 && t%5==0 && t<700){   
                vector<vector<vector<int>>> merge_indexLists = dpmmDir.computeSimilarity(int(dpmmDir.K_/2));
                for (int i_merge =0; i_merge < merge_indexLists.size(); ++i_merge){
                    vector<vector<int>> merge_indexList = merge_indexLists[i_merge];
                    if (!dpmmDir.mergeProposal(merge_indexList[0], merge_indexList[1]))
                        break;
                }
                dpmmDir.updateIndexLists();
            }
            cout << "Number of components: " << dpmmDir.K_ << endl;
        }
        
        z = dpmmDir.getLabels();
        logNum = dpmmDir.logNum_;
        logLogLik = dpmmDir.logLogLik_;
    }



    /*---------------------------------------------------*/
    //------------------Export the Output-----------------
    /*---------------------------------------------------*/


    string pathOut;
    if(vm.count("output")) pathOut = vm["output"].as<string>();
    pathOut += "output.csv";
    if (!pathOut.compare("")){
        cout<<"please specify an output data file"<<endl;
        return 1;
    }
    cout<<"Output to "<<pathOut<<endl;
    ofstream fout(pathOut.data(),ofstream::out);
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
        
    return 0;
}   