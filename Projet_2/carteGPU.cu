#include "utils/commonCUDA.hpp"
#include "utils/chronoGPU.hpp"

#include "carteGPU.hpp"
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <iomanip>

using namespace std;

__global__ void kernelMap(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy)
{
    
    float Dx, Dy, Dz,  D;
    float angle, angle_ref;
    float Cx_dda, Cy_dda; 
    float incX, incY;
    int Lx, Ly;
    for(int indexY = blockDim.y * blockIdx.y + threadIdx.y; indexY < MapHeight; indexY += blockDim.y * gridDim.y)
    {
        for(int indexX = blockDim.x * blockIdx.x + threadIdx.x; indexX < MapWidth; indexX += blockDim.x * gridDim.x)
        {
        // DDA entre le point c (Cx, Cy) et le point P (indexX, indexY);
            
            Dx = indexX - Cx;   // delta x
            Dy = indexY - Cy;   // delta y
            Dz = h_in[indexY * MapWidth + indexX] - h_in[Cy * MapWidth + Cx];   // delta z
            D = max(abs(Dx), abs(Dy));  // delta positif max entre Dx et Dy

            angle_ref = atan(Dz / sqrt((Dx * Dx) + (Dy * Dy)));
            Cx_dda = (float)Cx , Cy_dda = (float)(Cy);
            incX = Dx / D;
            incY = Dy / D;
            

            h_out[indexY * MapWidth + indexX] = 244;
            for (int i = 0; i < D - 1; i++)
            {
                Cx_dda += incX;
                Cy_dda += incY;
                Lx = (int)(round(Cx_dda));
                Ly = (int)(round(Cy_dda));

                Dx = indexX - Lx;
                Dy = indexY - Ly;    
                Dz = h_in[indexY * MapWidth + indexX] - h_in[Ly * MapWidth + Lx];  


                // Calcule Angle 

                angle = atan(Dz / sqrt((Dx * Dx) + (Dy * Dy)));     

                if (angle_ref >= angle)
                {
                    h_out[indexY * MapWidth + indexX] = 0;
                    break;
                }                         
            } 
        }
    }
}



void carteGPU(uint8_t *h_in, uint8_t *h_out, const int MapWidth, const int MapHeight, const int Cx, const int Cy)
{
    ChronoGPU chrGPU;
    uint8_t *dev_h_in;
    uint8_t *dev_h_out;

    size_t size = MapWidth * MapHeight * sizeof(uint8_t);

    cudaMalloc((void**) &dev_h_in, size);
    cudaMalloc((void**) &dev_h_out, size);

    cudaMemcpy(dev_h_in, h_in, size, cudaMemcpyHostToDevice);


    dim3 gridDim(16, 16);
    dim3 blockDim(16, 16);


    kernelMap<<<gridDim, blockDim>>>(dev_h_in, dev_h_out, MapHeight, MapHeight, Cx, Cy);

    cudaMemcpy(h_out, dev_h_out, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_h_in);
    cudaFree(dev_h_out);


}