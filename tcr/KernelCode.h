#ifndef KERNEL_CODE_H
#define KERNEL_CODE_H

#include <MRIData.h>

class KernelArgs
{
	public:
	int thread_idx;
	int num_threads;
	int pixel_start;
	int pixel_length;
	int num_pixels;
	int image_size;
	int channel_size;
	int coil_channel_size;
	int slice_size;
	int coil_slice_size;
	int temp_dim_size;
	MRIDimensions data_size;
	float* meas_data;
	float* estimate;
	float* gradient;
	float alpha;
	float beta;
	float beta_squared;
	float step_size;
};

#ifdef TCR_KERNEL_CUDA
#include <cufft.h>
__global__ void CUDA_LoadGradient( float* estimate, float* gradient, int num_pixels, int thread_load );
__global__ void CUDA_ApplyFidelityDifference( float* estimate, float* gradient, float* meas_data, int image_size, int num_pixels, int thread_load );
__global__ void CUDA_CalcTemporalGradient( float* gradient, float* estimate, int image_size, int num_phases, float beta, float beta_squared, int num_pixels, int thread_load );
__global__ void CUDA_UpdateEstimate( float* gradient, float* estimate, int num_pixels, float step_size, int thread_load );
void CUDA_FFTDirection( float* gradient, bool inverse, cufftHandle& plan, int image_size, int total_images );
#else
void* CPU_LoadGradient( void* args_ptr );
void* CPU_ApplyFidelityDifference( void* args_ptr );
void* CPU_FFT( void* args_ptr );
void* CPU_IFFT( void* args_ptr );
void* CPU_CalcTemporalGradient( void* args_ptr );
void* CPU_UpdateEstimate( void* args_ptr );
#endif


#endif
