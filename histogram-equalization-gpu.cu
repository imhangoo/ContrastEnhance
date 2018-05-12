#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

__global__
void calculate_hist(int * hist_out, unsigned char * img_in, int img_size)
{
 
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < img_size) atomicAdd(&hist_out[img_in[i]],1);
}

void histogram_gpu(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    int * hist_out_gpu;
    unsigned char * img_in_gpu;

    cudaMalloc(&hist_out_gpu, nbr_bin*sizeof(int)); 
    cudaMemset(&hist_out_gpu, 0, nbr_bin*sizeof(int));

    cudaMalloc(&img_in_gpu, img_size*sizeof(unsigned char));

    cudaMemcpy(img_in_gpu, img_in, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice);

    calculate_hist<<<(img_size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(hist_out_gpu, img_in_gpu, img_size);

    cudaMemcpy(hist_out, hist_out_gpu, nbr_bin*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(hist_out_gpu);
    cudaFree(img_in_gpu);   

}

__global__
void transform(unsigned char * img_out, int * lut, unsigned char * img_in, int img_size)
{
  extern __shared__ unsigned char s[];
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < img_size) {
    if(lut[img_in[i]] > 255){
            img_out[i] = 255;
    } else {
            img_out[i] = (unsigned char)lut[img_in[i]];
    }
  }

}



void histogram_equalization_gpu(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }

    }


    int * lut_gpu;
    unsigned char * img_in_gpu;
    unsigned char * img_out_gpu;

    cudaMalloc(&lut_gpu, nbr_bin*sizeof(int)); 
    cudaMalloc(&img_in_gpu, img_size*sizeof(unsigned char)); 

    cudaMalloc(&img_out_gpu, img_size*sizeof(unsigned char));
    cudaMemset(&img_out_gpu, 0, img_size*sizeof(unsigned char));

    cudaMemcpy(lut_gpu, lut, nbr_bin*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(img_in_gpu, img_in, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice);

    transform<<<(img_size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(img_out_gpu, lut_gpu, img_in_gpu, img_size);

    cudaMemcpy(img_out, img_out_gpu, img_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(lut_gpu);
    cudaFree(img_in_gpu);
    cudaFree(img_out_gpu);  
}

