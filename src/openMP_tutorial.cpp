// Open_MP Tutorial

#include <iostream>

using namespace std;

int main ()
{
    //create a new team of threads and explicitly specify the number of threads
    #pragma omp parallel num_threads(8) 
    {
    //
    #pragma omp for collapse(2) schedule(static)
    for(int n=0; n<100000; ++n) 
        for(int m=0; m<10000; ++m) 
            int i = n+m;
            // printf("%d %d\n", n, m);
    }
    // printf(".\n");

    return 0;
}



// Note: #pragma omp for only delegates portions of the loop for different threads in the current team. 
// A team is the group of threads executing the program.