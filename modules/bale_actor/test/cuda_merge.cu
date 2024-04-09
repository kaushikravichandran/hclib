#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/set_operations.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>

__global__ void merge(uint64_t *a, int64_t *b, uint64_t size1, uint64_t size2, uint64_t *counter)
{
    // printf("In kernel: %d %d", size1, size2);
    int i = (int) size1 / 8 * threadIdx.x;
    int j = (int) size2 / 8 * threadIdx.x;
    int endI = (int) size1 / 8 * (threadIdx.x + 1);
    int endJ = (int) size2 / 8 * (threadIdx.x + 1);
    uint64_t lastNum = 0;

    // Merge the two lists
    while (i < endI || j < endJ)
    {
        // printf("Here: %d %d", a[i], b[j]);
        if (a[i] == lastNum || b[j] == lastNum)
        {
            if (a[i] == lastNum)
            {
                i++;
            }
            else
            {
                j++;
            }

            continue;
        }

        (*counter)++;
        if (a[i] < b[j])
        {
            lastNum = a[i];
            i++;
        }
        else
        {
            lastNum = b[j];
            j++;
        }
    }

    // Copy the remaining elements from the first list
    while (i < endI) {
        (*counter)++;
        i++;
    }

    // Copy the remaining elements from the second list
    while (j < endJ) {
        (*counter)++;
        j++;
    }
}

// #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
// {
//    if (code != cudaSuccess) 
//    {
//       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//       if (abort) exit(code);
//    }
// }

uint64_t cuda_merge (uint64_t *a, uint64_t *b, int64_t* output, uint64_t size1, uint64_t size2, uint64_t size3)
{
    // if (size1 == 0 || size2 == 0)
    // {
    //     return 0;
    // }
    // uint64_t size3 = (size1 < size2) ? size1 : size2;
    // int64_t *output;// = (int64_t*)malloc(sizeof(int64_t) * size3);
    // gpuErrchk(cudaMalloc((void**)&output, size3 * sizeof(int64_t)));
    // gpuErrchk(cudaMemset(output, -1, sizeof(int64_t) * size3));

    thrust::device_ptr<uint64_t> a_start(a);
    thrust::device_ptr<uint64_t> a_end(a + size1);
    thrust::device_ptr<uint64_t> b_start(b);
    thrust::device_ptr<uint64_t> b_end(b + size2);
    thrust::device_ptr<int64_t> output_start(output);

    // printf("In cuda_merge (before): %d %d %d %d %d %d\n", size1, size2, size3, sizeof(a), sizeof(b), sizeof(output));
    thrust::device_ptr<int64_t> output_end = thrust::set_intersection(a_start, a_end, b_start, b_end, output_start, thrust::less<int>());
    //thrust::device_ptr<int64_t> output_end = thrust::set_intersection(thrust::host, a, a + size1, b, b + size2, output_start, thrust::less<int>());
    // cudaMemcpy(output_host, output, sizeof(int64_t) * size3, cudaMemcpyDeviceToHost);
    // printf("In cuda_merge (after): %d %d %d %d %d %d\n", size1, size2, size3, sizeof(a), sizeof(b), sizeof(output));
    // thrust::device_vector<int64_t> output_vec(output_start, output_start + size3);    
    // int count = thrust::count(output_vec.begin(), output_vec.end(), -1);
    // return size3 - count;
    return (output_end - output_start);
    // printf("In kernel func: %d %d\n", size1, size2);
    // merge<<<1, 8>>>(a, b, size1, size2, counter);
}
