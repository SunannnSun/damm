// Open_MP Tutorial

#include <iostream>

using namespace std;

int main ()
{
    //create a new team of threads and explicitly specify the number of threads
    
    //
    #pragma omp parallel for schedule(static) num_threads(8)
    for(int n=0; n<5; ++n) 
    {
        vector<int> x_k_index;
        for (int i=0; i<3; ++i)
        {
            if (i == n) 
            x_k_index.push_back(i); 
        }
        std::cout << x_k_index.size() << std::endl;

    }
    return 0;
}


// Note: #pragma omp for only delegates portions of the loop for different threads in the current team. 
// A team is the group of threads executing the program.