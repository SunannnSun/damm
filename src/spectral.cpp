#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace cv::ml;

// Spectral Clustering function
vector<int> spectralClustering(const MatrixXd& affinityMatrix, int numClusters) {
    int numPoints = affinityMatrix.rows();

    Mat kmeansInputMat(affinityMatrix.rows(), affinityMatrix.cols(), CV_32F);

    eigen2cv(affinityMatrix, kmeansInputMat);

    kmeansInputMat.convertTo(kmeansInputMat, CV_32F);
    // Mat kmeansInputMat(affinityMatrix.rows(), affinityMatrix.cols(), CV_32F, *scalarArray);
    std::cout << kmeansInputMat.type() << std::endl;

    // Mat kmeansInputMat(kmeansInput.rows(), kmeansInput.cols(), CV_32F);
    // eigen2cv(kmeansInput, kmeansInputMat);
    // Mat data0 = kmeansInputMat.getMat();
    // std::cout << data0.dims << std::endl;

    // std::cout << kmeansInput.rows() << std::endl;
    // std::cout << kmeansInput.cols() << std::endl;

    // Create the k-means model

    
    Mat labels;
    std::vector<Point2f> centers;

    kmeans(kmeansInputMat, numClusters, labels,
            TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 1.0),
            3, KMEANS_RANDOM_CENTERS,  centers
         );

    std::cout << labels << std::endl;
    std::cout << centers << std::endl;

    vector<int> clusterAssignments;

    const int* data = labels.ptr<int>();

    // Copy the data to the vector
    clusterAssignments.assign(data, data + labels.total());


    return clusterAssignments;
}

int main() {
    string pathIn ="/Users/sunansun/Developer/dpmm/data/ab.csv";
    ifstream  fin(pathIn);
    string line;
    vector<vector<string> > parsedCsv;
    while(getline(fin,line)){
        // vector<double> row;
        stringstream lineStream(line);
        string cell;
        vector<string> parsedRow;
        while(getline(lineStream,cell,','))  
            parsedRow.push_back(cell);
        parsedCsv.push_back(parsedRow);
    }
    fin.close();

    int num = parsedCsv.size();
    int dim = (num > 0) ? parsedCsv[0].size() : 0;


    MatrixXd Data(num, dim);              
    for (uint32_t i=0; i<num; ++i)
        for (uint32_t j=0; j<dim; ++j)
            Data(i, j) = stod(parsedCsv[i][j]);

    Data = Data(all, seq(0, 1));






    // // Example usage
    // int numPoints = 20;
    // int numClusters = 2;
    // double minVal = -1.0;
    // double maxVal = 1.0;

    // // Generate a random affinity matrix (can be replaced with your own data)
    // MatrixXd affinityMatrix = MatrixXd::Random(numPoints, numClusters);
    // // randomMatrix = (randomMatrix.array() * (maxVal - minVal)) + minVal;

    // std::cout << affinityMatrix << std::endl;
    // // affinityMatrix = (affinityMatrix + affinityMatrix.transpose()) / 2.0;

    vector<int> clusterAssignments = spectralClustering(Data, 2);

    // // Print the cluster assignments
    // cout << "Cluster Assignments:" << endl;
    // for (int i = 0; i < numPoints; ++i) {
    //     cout << "Point " << i << ": Cluster " << clusterAssignments[i] << endl;
    // }
    string pathOut = "/Users/sunansun/Developer/dpmm/data/ab_output.csv";
    ofstream fout(pathOut.data(),ofstream::out);
    for (uint16_t i=0; i < clusterAssignments.size(); ++i)
        fout << clusterAssignments[i] << endl;
    fout.close();

    return 0;
}
