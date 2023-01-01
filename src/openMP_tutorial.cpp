// Open_MP Tutorial

#include <iostream>
#include <algorithm>
#define ARRAY_SIZE 1000000000
#define ARRAY_VALUE 1231
int main()
{
    int *arr = new int[ARRAY_SIZE];
    std::fill_n(arr, ARRAY_SIZE, ARRAY_VALUE);
    #pragma omp parallel for num_threads(8) schedule(dynamic,10000000)
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        // do some relatively long operation
        arr[i] = arr[i] / arr[i] + arr[i] / 5 - 14;
    }
    return 0;
}


// Note: #pragma omp for only delegates portions of the loop for different threads in the current team. 
// A team is the group of threads executing the program.