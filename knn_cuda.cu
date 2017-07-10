#include <stdio.h>
#include <iostream>
#include <math.h>
#include "cuda.h"
#include <time.h>

#define BLOCK_DIM 16

__global__ void computeDistance(float* A, int wA, int pA, float* B, int wB, int pB, int dim,  float* AB) {

    // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
    // They will contain, for each thread, the current coordinates of A and B - block_dim in each step
	__shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
	__shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

    // Sub-matrix of A (begin, step, end) and sub-matrix of B (begin, step)
    __shared__ int begin_A;
    __shared__ int begin_B;
    __shared__ int step_A;
    __shared__ int step_B;
    __shared__ int end_A;
	
    // Thread index
    int tx = threadIdx.x;   // Local query point index
    int ty = threadIdx.y;   // Local ref point index
	
	// Other variables
	float tmp;
    float ssd = 0;
	
    // Loop parameters
    begin_A = BLOCK_DIM * blockIdx.y;   // Each block has its own start on ref points
    begin_B = BLOCK_DIM * blockIdx.x; // Each block has its own start on query points
    step_A  = BLOCK_DIM * pA; // next step = next row of the big matrix
    step_B  = BLOCK_DIM * pB;
    end_A   = begin_A + (dim-1) * pA; // Each submatrix treated by given block has BLOCK_DIM columns and dim rows
    
    // Conditions
	int cond0 = (begin_A + tx < wA); // current column is out of A
    int cond1 = (begin_B + tx < wB); // current column is out of B
    int cond2 = (begin_A + ty < wA); // ty is column number in A
    
    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {
        
        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        // ty corresponds to row, tx to column in the resulting matrix, as well as ref and query points in input, 
        // but when copying to local memory, they work just as numbers for indeces (tx is column number in both cases)
        // a/pA + ty is the row number in A corresponding to this thread in this block
        if (a/pA + ty < dim){
            shared_A[ty][tx] = (cond0)? A[a + pA * ty + tx] : 0;
            shared_B[ty][tx] = (cond1)? B[b + pB * ty + tx] : 0; 
        }
        else {
            shared_A[ty][tx] = 0;
            shared_B[ty][tx] = 0;
        }
        
        // Synchronize to make sure the matrices are loaded
        __syncthreads();
        
        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
        if (cond2 && cond1) {
            for (int k = 0; k < BLOCK_DIM; ++k){
				tmp = shared_A[k][ty] - shared_B[k][tx];
                ssd += tmp*tmp;
			}
        }
        
        // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
    // Write the block sub-matrix to device memory; each thread writes one element
    if(cond2 && cond1)
        AB[ (begin_A + ty) * pB + begin_B + tx ] = ssd;

}

// Selection sort
__global__ void sort(float *dist, int *ind, int width, int pitch, int ind_pitch, int height, int k){

	// Variables
    int l, i, min_index;
    float *p_dist;
    int *p_ind;
    float min_value, tmp;
    // xIndex is column in the sorted matrix
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (xIndex < width) {
        
        p_dist      = dist+xIndex;
        p_ind       = ind+xIndex;
        min_value = *p_dist;
        
        for (l = 0; l < k; l++) {
            min_index = l;
            min_value = *(p_dist+l*pitch);
            for (i=l+1; i < height; i++) {
                if (*(p_dist+i*pitch) < min_value) {
                    min_index = i;
                    min_value = *(p_dist+i*pitch);
                }
            } 

            if (min_index != l) {
                tmp = *(p_dist+min_index*pitch);
                *(p_dist+min_index*pitch) = *(p_dist+l*pitch);
                *(p_dist+l*pitch) = tmp;
            }
            p_ind[l*ind_pitch] = min_index;
        }
   }
}

__global__ void parallelSqrt(float *dist, int width, int pitch, int k) {
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex<width && yIndex<k)
        dist[yIndex*pitch + xIndex] = sqrt(dist[yIndex*pitch + xIndex]);
}


// Compute the mean of the first k elements
__global__ void mean(float *dist, int width, int pitch, float *res, int k) {
    
    float sum;
    float *p;
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (xIndex < width) {
        sum = 0;
        p = dist + xIndex;
        for (int l = 0; l < k*pitch; l += pitch) sum += *(p+l); 
        res[xIndex] = sum/k;
    }
}

void printErrorMessage(cudaError_t error, int memorySize){
    printf("==================================================\n");
    printf("MEMORY ALLOCATION ERROR  : %s\n", cudaGetErrorString(error));
    printf("Wished allocated memory : %d\n", memorySize);
    printf("==================================================\n");
}

void knn(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host, float* res_host, int *ind_host) {

    // Initialize variables
    float *ref_dev;
    float *query_dev;
    float *dist_dev;
    int   *ind_dev;
    float *res_dev;

    size_t ref_pitch_in_bytes;
    size_t query_pitch_in_bytes;
    size_t res_pitch_in_bytes;
    size_t ind_pitch_in_bytes;
    size_t dist_pitch_in_bytes;
    size_t ref_pitch;
    size_t query_pitch;
    // size_t res_pitch;
    size_t ind_pitch;
    cudaError_t  result;

    // Allocate device memory
    result = cudaMallocPitch((void **) &ref_dev,     &ref_pitch_in_bytes,    ref_width * sizeof(float), height);
    if (result){
        cudaFree(ref_dev);
        printErrorMessage(result, ref_width*sizeof(float)*height);
        return;
    }
    result = cudaMallocPitch((void **) &query_dev,   &query_pitch_in_bytes,  query_width*sizeof(float), height);
    if (result){
        cudaFree(query_dev);
        printErrorMessage(result, query_width*sizeof(float)*k);
        return;
    }
    result = cudaMallocPitch((void **) &dist_dev,    &dist_pitch_in_bytes,  query_width*sizeof(float), ref_width);
    if (result){
        cudaFree(dist_dev);
        printErrorMessage(result, query_width*sizeof(float)*ref_width);
        return;
    }
    result = cudaMallocPitch((void **) &ind_dev,     &ind_pitch_in_bytes,    query_width*sizeof(int),   k);
    if (result){
        cudaFree(ind_dev);
        printErrorMessage(result, query_width*sizeof(int)*k);
        return;
    }
    result = cudaMallocPitch((void **) &res_dev,     &res_pitch_in_bytes,    query_width*sizeof(float), 1);
    if (result){
        cudaFree(res_dev);
        printErrorMessage(result, query_width*sizeof(float));
        return;
    }

    // Copy reference and query points to global memory
    cudaMemcpy2D(ref_dev,   ref_pitch_in_bytes,     ref_host,   ref_width*sizeof(float),    ref_width*sizeof(float),    height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(query_dev, query_pitch_in_bytes,   query_host, query_width*sizeof(float),  query_width*sizeof(float),  height, cudaMemcpyHostToDevice);
    
    // Compute the pitches
    ref_pitch = ref_pitch_in_bytes/sizeof(float); 
    query_pitch = query_pitch_in_bytes/sizeof(float);
    // res_pitch = res_pitch_in_bytes/sizeof(float);
    ind_pitch = ind_pitch_in_bytes/sizeof(int);

    // Set kernel dims
    // Each block has 16x16 threads, and processes 1/16 of ref width
    // It creates a local 16x16 matrix, which goes down the rows
    // The number of blocks depends on nb_ref, threads/block is fixed
    dim3 threads_per_block_2D(BLOCK_DIM, BLOCK_DIM, 1);   
    dim3 threads_per_block_1D(BLOCK_DIM * BLOCK_DIM, 1, 1);

    dim3 blocks_2D(std::ceil((float) query_width/BLOCK_DIM), std::ceil((float) ref_width/BLOCK_DIM), 1);
    dim3 blocks_2D_k(std::ceil((float) query_width/BLOCK_DIM), std::ceil((float) k/BLOCK_DIM), 1);
    dim3 blocks_1D(std::ceil((float) query_width/(BLOCK_DIM*BLOCK_DIM)), 1, 1);

    // Start kernels
    computeDistance<<<blocks_2D, threads_per_block_2D>>>(ref_dev, ref_width, ref_pitch, query_dev, query_width, query_pitch, height, dist_dev);
    sort<<<blocks_1D, threads_per_block_1D>>>(dist_dev, ind_dev, query_width, query_pitch, ind_pitch, ref_width, k);
    // insertionSort<<<blocks_1D, threads_per_block_1D>>>(dist_dev, query_pitch, ind_dev, ind_pitch, query_width, ref_width, k);


    parallelSqrt<<<blocks_2D_k, threads_per_block_2D>>>(dist_dev, query_width, query_pitch, k);
    mean<<<blocks_1D, threads_per_block_1D>>>(dist_dev, query_width, query_pitch, res_dev, k);


    // Copy memory from device to host
    cudaMemcpy2D(res_host, query_width*sizeof(float), res_dev, query_pitch_in_bytes, query_width*sizeof(float), 1, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(dist_host, query_width*sizeof(float), dist_dev, dist_pitch_in_bytes, query_width*sizeof(float),k , cudaMemcpyDeviceToHost);
    cudaMemcpy2D(ind_host, query_width*sizeof(int)   , ind_dev, ind_pitch_in_bytes,   query_width*sizeof(int)  , k, cudaMemcpyDeviceToHost);

    cudaFree(ref_dev); cudaFree(query_dev); cudaFree(res_dev); cudaFree(ind_dev);
}

int main() {
    
    // Initialize variables
    float *ref;
    float *query;
    float *dist;
    float *res;
    int *ind;
    int ref_nb      = 4096;
    int query_nb    = 4096;
    int dim         = 32;
    int k           = 20;

    // Allocate host memory
    ref     = (float *)  malloc(ref_nb   * dim * sizeof(float));
    query   = (float *)  malloc(query_nb * dim * sizeof(float));
    dist    = (float *)  malloc(query_nb * k   * sizeof(float));
    res     = (float *)  malloc(query_nb * 1   * sizeof(float)); // Mean of the first k distances in the sorted matrix
    ind     = (int   *)  malloc(query_nb * k   * sizeof(int));
    
    // Generate random data
    srand(time(NULL));
    for (int i = 0; i<ref_nb * dim; i++)   ref[i]      = (float) (rand() % 100);
    for (int i = 0; i<query_nb * dim; i++) query[i]    = (float) (rand() % 100);

    knn(ref, ref_nb, query, query_nb, dim, k, dist, res, ind);

    for (int j = 0; j < 10; j++) {
        std::cout << "( ";
        for (int i = 0; i < dim; i++) std::cout << query[i*query_nb+j] << " ";
        std::cout << ")" << std::endl;
        std::cout << res[j] << std::endl;
        for (int i = 0; i < k; i++) std::cout << ind[i*query_nb+j] << " ";
        std::cout << std::endl << std::endl;
    }


    for (int i = 0; i < k; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << dist[i*query_nb + j] << " ";
        }
        std::cout << std::endl;
    }

    free(ref); free(query); free(dist); free(ind);
    return 0;
}
