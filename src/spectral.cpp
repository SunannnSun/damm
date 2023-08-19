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

vector<int> kmeans(const MatrixXd& Data, int numClusters) {
    int numPoints = Data.rows();
    Mat kmeansInputMat(Data.rows(), Data.cols(), CV_32F);
    eigen2cv(Data, kmeansInputMat);
    kmeansInputMat.convertTo(kmeansInputMat, CV_32F);
    // std::cout << kmeansInputMat.type() << std::endl;
    
    Mat labels;
    // std::vector<Point3f> centers;

    kmeans(kmeansInputMat, numClusters, labels,
            TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 1.0),
            3, KMEANS_RANDOM_CENTERS
         );

    // std::cout << labels << std::endl;

    vector<int> clusterAssignments(numPoints);
    const int* data = labels.ptr<int>();
    clusterAssignments.assign(data, data + labels.total());
    for (int i = 0; i < numPoints; ++i) {
        clusterAssignments[i] = labels.at<int>(i);
    }

    return clusterAssignments;
}

// int main() {


//     Data = Data(all, seq(0, 1));


//     // vector<int> clusterAssignments = spectralClustering(Data, 2);

//     // // Print the cluster assignments
//     // cout << "Cluster Assignments:" << endl;
//     // for (int i = 0; i < numPoints; ++i) {
//     //     cout << "Point " << i << ": Cluster " << clusterAssignments[i] << endl;
//     // }


//     return 0;
// }
