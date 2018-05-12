#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

PGM_IMG contrast_enhancement_g_gpu(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    histogram_gpu(hist, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization_gpu(result.img,img_in.img,hist,result.w*result.h, 256);
    return result;
}

PPM_IMG contrast_enhancement_c_rgb_gpu(PPM_IMG img_in)
{
    PPM_IMG result;
    int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    histogram_gpu(hist, img_in.img_r, img_in.h * img_in.w, 256);
    histogram_equalization_gpu(result.img_r,img_in.img_r,hist,result.w*result.h, 256);
    histogram_gpu(hist, img_in.img_g, img_in.h * img_in.w, 256);
    histogram_equalization_gpu(result.img_g,img_in.img_g,hist,result.w*result.h, 256);
    histogram_gpu(hist, img_in.img_b, img_in.h * img_in.w, 256);
    histogram_equalization_gpu(result.img_b,img_in.img_b,hist,result.w*result.h, 256);

    return result;
}


PPM_IMG contrast_enhancement_c_yuv_gpu(PPM_IMG img_in)
{
    YUV_IMG yuv_med;
    PPM_IMG result;
    
    unsigned char * y_equ;
    int hist[256];
    
    yuv_med = rgb2yuv_gpu(img_in);
    y_equ = (unsigned char *)malloc(yuv_med.h*yuv_med.w*sizeof(unsigned char));
    
    histogram_gpu(hist, yuv_med.img_y, yuv_med.h * yuv_med.w, 256);
    histogram_equalization_gpu(y_equ,yuv_med.img_y,hist,yuv_med.h * yuv_med.w, 256);

    free(yuv_med.img_y);
    yuv_med.img_y = y_equ;
    
    result = yuv2rgb_gpu(yuv_med);
    free(yuv_med.img_y);
    free(yuv_med.img_u);
    free(yuv_med.img_v);
    
    return result;
}

PPM_IMG contrast_enhancement_c_hsl_gpu(PPM_IMG img_in)
{
    HSL_IMG hsl_med;
    PPM_IMG result;
    
    unsigned char * l_equ;
    int hist[256];

    hsl_med = rgb2hsl_gpu(img_in);

    l_equ = (unsigned char *)malloc(hsl_med.height*hsl_med.width*sizeof(unsigned char));

    histogram_gpu(hist, hsl_med.l, hsl_med.height * hsl_med.width, 256);

    histogram_equalization_gpu(l_equ, hsl_med.l,hist,hsl_med.width*hsl_med.height, 256);
  
    
    free(hsl_med.l);
    hsl_med.l = l_equ;

    result = hsl2rgb_gpu(hsl_med);

    free(hsl_med.h);
    free(hsl_med.s);
    free(hsl_med.l);
    return result;
}

__global__
void rgb3hsl(PPM_IMG img_in, HSL_IMG img_out, int img_size)
{
  float H, S, L;
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i < img_size) {
    float var_r = ( (float)img_in.img_r[i]/255 );//Convert RGB to [0,1]

    float var_g = ( (float)img_in.img_g[i]/255 );
    float var_b = ( (float)img_in.img_b[i]/255 );
    float var_min = (var_r < var_g) ? var_r : var_g;
    var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
    float var_max = (var_r > var_g) ? var_r : var_g;
    var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
    float del_max = var_max - var_min;               //Delta RGB value
    
    L = ( var_max + var_min ) / 2;

    if ( del_max == 0 )//This is a gray, no chroma...
    {
        H = 0;         
        S = 0;    
    }
    else                                    //Chromatic data...
    {
        if ( L < 0.5 )
            S = del_max/(var_max+var_min);
        else
            S = del_max/(2-var_max-var_min );

        float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
        float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
        float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
        if( var_r == var_max ){
            H = del_b - del_g;
        }
        else{       
            if( var_g == var_max ){
                H = (1.0/3.0) + del_r - del_b;
            }
            else{
                    H = (2.0/3.0) + del_g - del_r;
            }   
        }
        
    }
    
    if ( H < 0 )
        H += 1;
    if ( H > 1 )
        H -= 1;

    img_out.h[i] = H;
    img_out.s[i] = S;
    img_out.l[i] = (unsigned char)(L*255);

  }
}

//Convert RGB to HSL, assume R,G,B in [0, 255]
//Output H, S in [0.0, 1.0] and L in [0, 255]
HSL_IMG rgb2hsl_gpu(PPM_IMG img_in)
{
    HSL_IMG img_out;// = (HSL_IMG *)malloc(sizeof(HSL_IMG));
    img_out.width  = img_in.w;
    img_out.height = img_in.h;
    img_out.h = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.s = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.l = (unsigned char *)malloc(img_in.w * img_in.h * sizeof(unsigned char));
    
    int img_size = img_in.w*img_in.h;

    PPM_IMG img_in_gpu;
    HSL_IMG img_out_gpu;

    cudaMalloc(&img_in_gpu.img_r, img_size*sizeof(unsigned char)); 
    cudaMalloc(&img_in_gpu.img_g, img_size*sizeof(unsigned char)); 
    cudaMalloc(&img_in_gpu.img_b, img_size*sizeof(unsigned char)); 

    cudaMemcpy(img_in_gpu.img_r, img_in.img_r, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(img_in_gpu.img_g, img_in.img_g, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(img_in_gpu.img_b, img_in.img_b, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc(&img_out_gpu.h, img_size*sizeof(float)); 
    cudaMalloc(&img_out_gpu.s, img_size*sizeof(float)); 
    cudaMalloc(&img_out_gpu.l, img_size*sizeof(unsigned char));

    rgb3hsl<<<(img_size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(img_in_gpu, img_out_gpu, img_size);

    cudaMemcpy(img_out.h, img_out_gpu.h, img_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.s, img_out_gpu.s, img_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.l, img_out_gpu.l, img_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);

   
    cudaFree(img_in_gpu.img_r);
    cudaFree(img_in_gpu.img_g);
    cudaFree(img_in_gpu.img_b);
    cudaFree(img_out_gpu.h);
    cudaFree(img_out_gpu.s);
    cudaFree(img_out_gpu.l);

    return img_out;
}

__device__
float Hue_2_RGB_gpu( float v1, float v2, float vH )             //Function Hue_2_RGB
{
    if ( vH < 0 ) vH += 1;
    if ( vH > 1 ) vH -= 1;
    if ( ( 6 * vH ) < 1 ) return ( v1 + ( v2 - v1 ) * 6 * vH );
    if ( ( 2 * vH ) < 1 ) return ( v2 );
    if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2.0f/3.0f ) - vH ) * 6 );
    return ( v1 );
}

__global__
void hsl3rgb(HSL_IMG img_in, PPM_IMG img_out, int img_size)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i < img_size) {
    float H = img_in.h[i];
    float S = img_in.s[i];
    float L = img_in.l[i]/255.0f;
    float var_1, var_2;
    
    unsigned char r,g,b;
    
    if ( S == 0 )
    {
        r = L * 255;
        g = L * 255;
        b = L * 255;
    }
    else
    {
        
        if ( L < 0.5 )
            var_2 = L * ( 1 + S );
        else
            var_2 = ( L + S ) - ( S * L );

        var_1 = 2 * L - var_2;
        r = 255 * Hue_2_RGB_gpu( var_1, var_2, H + (1.0f/3.0f) );
        g = 255 * Hue_2_RGB_gpu( var_1, var_2, H );
        b = 255 * Hue_2_RGB_gpu( var_1, var_2, H - (1.0f/3.0f) );
    }
    img_out.img_r[i] = r;
    img_out.img_g[i] = g;
    img_out.img_b[i] = b;

  }
}
//Convert HSL to RGB, assume H, S in [0.0, 1.0] and L in [0, 255]
//Output R,G,B in [0, 255]
PPM_IMG hsl2rgb_gpu(HSL_IMG img_in)
{
    PPM_IMG img_out;
    
    img_out.w = img_in.width;
    img_out.h = img_in.height;
    img_out.img_r = (unsigned char *)malloc(img_out.w * img_out.h * sizeof(unsigned char));
    img_out.img_g = (unsigned char *)malloc(img_out.w * img_out.h * sizeof(unsigned char));
    img_out.img_b = (unsigned char *)malloc(img_out.w * img_out.h * sizeof(unsigned char));

    int img_size = img_in.width*img_in.height;

    HSL_IMG img_in_gpu;
    PPM_IMG img_out_gpu;

    cudaMalloc(&img_in_gpu.h, img_size*sizeof(float)); 
    cudaMalloc(&img_in_gpu.s, img_size*sizeof(float)); 
    cudaMalloc(&img_in_gpu.l, img_size*sizeof(unsigned char)); 

    cudaMemcpy(img_in_gpu.h, img_in.h, img_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(img_in_gpu.s, img_in.s, img_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(img_in_gpu.l, img_in.l, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc(&img_out_gpu.img_r, img_size*sizeof(unsigned char)); 
    cudaMalloc(&img_out_gpu.img_g, img_size*sizeof(unsigned char)); 
    cudaMalloc(&img_out_gpu.img_b, img_size*sizeof(unsigned char));

    hsl3rgb<<<(img_size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(img_in_gpu, img_out_gpu, img_size);

    cudaMemcpy(img_out.img_r, img_out_gpu.img_r, img_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_g, img_out_gpu.img_g, img_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_b, img_out_gpu.img_b, img_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);

   
    cudaFree(img_in_gpu.h);
    cudaFree(img_in_gpu.s);
    cudaFree(img_in_gpu.l);
    cudaFree(img_out_gpu.img_r);
    cudaFree(img_out_gpu.img_g);
    cudaFree(img_out_gpu.img_b);
    //printf("hsl gpu out 10000 pixel r g b%d %d %d\n", img_out.img_r[10000], img_out.img_g[10000], img_out.img_b[10000]);
    
    return img_out;
}

__global__
void rgb3yuv(PPM_IMG img_in, YUV_IMG img_out, int img_size)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < img_size) {
    unsigned char r, g, b;
    unsigned char y, cb, cr;
    r = img_in.img_r[i];
    g = img_in.img_g[i];
    b = img_in.img_b[i];
    
    y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
    cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
    cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
    
    img_out.img_y[i] = y;
    img_out.img_u[i] = cb;
    img_out.img_v[i] = cr;
  }
}

//Convert RGB to YUV, all components in [0, 255]
YUV_IMG rgb2yuv_gpu(PPM_IMG img_in)
{
    YUV_IMG img_out;
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
 
    int img_size = img_in.w*img_in.h;

    PPM_IMG img_in_gpu;
    YUV_IMG img_out_gpu;

    cudaMalloc(&img_in_gpu.img_r, img_size*sizeof(unsigned char)); 
    cudaMalloc(&img_in_gpu.img_g, img_size*sizeof(unsigned char)); 
    cudaMalloc(&img_in_gpu.img_b, img_size*sizeof(unsigned char)); 

    cudaMemcpy(img_in_gpu.img_r, img_in.img_r, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(img_in_gpu.img_g, img_in.img_g, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(img_in_gpu.img_b, img_in.img_b, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc(&img_out_gpu.img_y, img_size*sizeof(unsigned char)); 
    cudaMalloc(&img_out_gpu.img_u, img_size*sizeof(unsigned char)); 
    cudaMalloc(&img_out_gpu.img_v, img_size*sizeof(unsigned char));

    rgb3yuv<<<(img_size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(img_in_gpu, img_out_gpu, img_size);

    cudaMemcpy(img_out.img_y, img_out_gpu.img_y, img_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_u, img_out_gpu.img_u, img_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_v, img_out_gpu.img_v, img_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);

   
    cudaFree(img_in_gpu.img_r);
    cudaFree(img_in_gpu.img_g);
    cudaFree(img_in_gpu.img_b);
    cudaFree(img_out_gpu.img_y);
    cudaFree(img_out_gpu.img_u);
    cudaFree(img_out_gpu.img_v);
    return img_out;
}

__device__
unsigned char clip_rgb_gpu(int x)
{
    if(x > 255)
        return 255;
    if(x < 0)
        return 0;

    return (unsigned char)x;
}

__global__
void yuv3rgb(YUV_IMG img_in, PPM_IMG img_out, int img_size)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i < img_size) {
    int  rt,gt,bt;
    int y, cb, cr;
    y  = (int)img_in.img_y[i];
    cb = (int)img_in.img_u[i] - 128;
    cr = (int)img_in.img_v[i] - 128;
    
    rt  = (int)( y + 1.402*cr);
    gt  = (int)( y - 0.344*cb - 0.714*cr);
    bt  = (int)( y + 1.772*cb);

    img_out.img_r[i] = clip_rgb_gpu(rt);
    img_out.img_g[i] = clip_rgb_gpu(gt);
    img_out.img_b[i] = clip_rgb_gpu(bt);

  }
}

//Convert YUV to RGB, all components in [0, 255]
PPM_IMG yuv2rgb_gpu(YUV_IMG img_in)
{
    PPM_IMG img_out;   
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    int img_size = img_in.w*img_in.h;

    YUV_IMG img_in_gpu;
    PPM_IMG img_out_gpu;

    cudaMalloc(&img_in_gpu.img_y, img_size*sizeof(unsigned char)); 
    cudaMalloc(&img_in_gpu.img_u, img_size*sizeof(unsigned char)); 
    cudaMalloc(&img_in_gpu.img_v, img_size*sizeof(unsigned char)); 

    cudaMemcpy(img_in_gpu.img_y, img_in.img_y, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(img_in_gpu.img_u, img_in.img_u, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(img_in_gpu.img_v, img_in.img_v, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc(&img_out_gpu.img_r, img_size*sizeof(unsigned char)); 
    cudaMalloc(&img_out_gpu.img_g, img_size*sizeof(unsigned char)); 
    cudaMalloc(&img_out_gpu.img_b, img_size*sizeof(unsigned char));

    yuv3rgb<<<(img_size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(img_in_gpu, img_out_gpu, img_size);

    cudaMemcpy(img_out.img_r, img_out_gpu.img_r, img_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_g, img_out_gpu.img_g, img_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_b, img_out_gpu.img_b, img_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);

   
    cudaFree(img_in_gpu.img_y);
    cudaFree(img_in_gpu.img_u);
    cudaFree(img_in_gpu.img_v);
    cudaFree(img_out_gpu.img_r);
    cudaFree(img_out_gpu.img_g);
    cudaFree(img_out_gpu.img_b);

    //printf("yuv gpu out 10000 pixel r g b%d %d %d\n", img_out.img_r[10000], img_out.img_g[10000], img_out.img_b[10000]);
    return img_out;
}
