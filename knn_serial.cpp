#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <time.h>

using namespace std;

float distance(float *A, int iA, float *B, int jB, int dim) {
    float sum = 0;
    float *a = A+iA;
    float *b = B+jB;
    
    for (int i = 0; i < dim; i++) 
        sum += (a[i]-b[i])*(a[i]-b[i]);

    sum = sqrt(sum);
    return sum;
}

void computeDistanceMatrix(float* A, int wA, float* B, int wB, int dim,  float* AB) {
    for (int i = 0; i < wA-1; i++) {
        for (int j = i+1; j < wB; j++) {
           AB[i*wB + j] = distance(A, i, B, j, dim);
           AB[j*wA + i] = AB[i*wB +j];
        }
    } 
}

void sort(float *dist, int *ind, int width, int height, int k){

    int l, i, j, min_index;
    float *p_dist;
    int *p_ind;
    float min_value, tmp;
	
    for (j = 0; j < width; j++) {
        
        p_dist      = dist+j;
        p_ind       = ind+j;
        min_value = *p_dist;
        
        for (l = 0; l <= k; l++) {
            min_index = l;
            min_value = *(p_dist+l*width);
            for (i=l+1; i < height; i++) {
                if (*(p_dist+i*width) < min_value) {
                    min_index = i;
                    min_value = *(p_dist+i*width);
                }
            } 

            if (min_index != l) {
                tmp = *(p_dist+min_index*width);
                *(p_dist+min_index*width) = *(p_dist+l*width);
                *(p_dist+l*width) = tmp;
            }
            p_ind[l*width] = min_index;
        }
   }
}

void knn(float* ref, int ref_width, float* query, int query_width, int height, int k, float* dist, int *ind) {
    computeDistanceMatrix(ref, ref_width, query, query_width, height, dist);
    sort(dist, ind, query_width, ref_width, k);
}

int main() {
    
    float *ref;
    float *query;
    float *dist;
    //float *res;
    int *ind;
    int ref_nb      = 64;
    int query_nb    = 64;
    int dim         = 10;
    int k           = 5;

    // Allocate host memory
    ref     = (float *)  malloc(ref_nb   * dim * sizeof(float));
    query   = (float *)  malloc(query_nb * dim * sizeof(float));
    dist    = (float *)  malloc(query_nb * ref_nb   * sizeof(float));
    //res     = (float *)  malloc(query_nb * 1   * sizeof(float)); // Mean of the first k distances in the sorted matrix
    ind     = (int   *)  malloc(query_nb * (k+1)   * sizeof(int));
    
    // Generate random data
    srand(time(NULL));
    for (int i = 0; i<ref_nb * dim; i++)   ref[i]      = (float) (rand() % 100);
    for (int i = 0; i<query_nb * dim; i++) query[i]    = (float) (rand() % 100);

    knn(ref, ref_nb, query, query_nb, dim, k, dist, ind);

    for (int j = 0; j < 10; j++) {
        cout << "( ";
        for (int i = 0; i < dim; i++) cout << query[i*query_nb+j] << " ";
        cout << ")" << endl;
        //cout << res[j] << endl;
        for (int i = 1; i <= k; i++) cout << ind[i*query_nb+j] << " ";
        cout << endl << endl;
    }


    for (int i = 1; i <= k; i++) {
        for (int j = 0; j < 10; j++) {
            cout << dist[i*query_nb + j] << " ";
        }
        cout << endl;
    }

    free(ref); free(query); free(dist); free(ind); //free(res);
    return 0;
}
