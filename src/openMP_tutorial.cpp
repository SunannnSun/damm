#include <iostream>
#include <algorithm>
#include <omp.h>
#define ARRAY_SIZE 100000000
#define ARRAY_VALUE 1231
int main()
{
    int *arr = new int[ARRAY_SIZE];
    std::fill_n(arr, ARRAY_SIZE, ARRAY_VALUE);
    #pragma omp parallel for num_threads(4) 
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        arr[i] = arr[i] / arr[i] + arr[i] / 5 - 14;
    }
    return 0;
}