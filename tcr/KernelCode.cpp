#include <KernelCode.h>
#include <FilterTool.h>
#include <cmath>

#ifdef TCR_KERNEL_CUDA
	#include <cufft.h>
#else
	void* CPU_FFTDirection( void* args_ptr, bool reverse );
#endif

#ifdef TCR_KERNEL_CUDA
void CUDA_FFTDirection( float* gradient, bool inverse, cufftHandle& plan, int image_size, int total_images )
{
	for( int i = 0; i < total_images; i++ )
	{
		float* gradient_ptr = gradient + ( 2 * i * image_size );
		if( inverse )
			cufftExecC2C( plan, (cufftComplex*)gradient_ptr, (cufftComplex*)gradient_ptr, CUFFT_INVERSE );
		else
			cufftExecC2C( plan, (cufftComplex*)gradient_ptr, (cufftComplex*)gradient_ptr, CUFFT_FORWARD );
	}
}
#else
void* CPU_FFT( void* args_ptr ) { return CPU_FFTDirection( args_ptr, false ); }
void* CPU_IFFT( void* args_ptr ) { return CPU_FFTDirection( args_ptr, true); }
void* CPU_FFTDirection( void* args_ptr, bool reverse )
{
	KernelArgs* args = (KernelArgs*) args_ptr;
	int image_size = args->data_size.Column * args->data_size.Line;
	int total_images = args->num_pixels / image_size;
	int images_per_thread = (int)ceil( (float)total_images / args->num_threads );
	int first_image = args->thread_idx * images_per_thread;
	int last_image = first_image + images_per_thread;
	if( last_image > total_images )
		last_image = total_images;

	for( int i = first_image; i < last_image; i++ )
	{
		float* gradient_ptr = args->gradient + ( 2 * i * image_size );
		FilterTool::FFT2D( gradient_ptr, gradient_ptr, args->data_size.Column, args->data_size.Line, reverse );
	}
}
#endif

#ifdef TCR_KERNEL_CUDA
__global__ void CUDA_ApplyFidelityDifference( float* estimate, float* gradient, float* meas_data, int image_size, int num_pixels, int thread_load )
{
	int pixel_start = ( blockIdx.x*blockDim.x + threadIdx.x ) * thread_load;
#else
void* CPU_ApplyFidelityDifference( void* args_ptr )
{
	KernelArgs* args = (KernelArgs*) args_ptr;
	float* gradient = args->gradient;
	float* estimate = args->estimate;
	float* meas_data = args->meas_data;
	int image_size = args->image_size;
	int pixel_start = args->pixel_start;
	int thread_load = args->pixel_length;
	int num_pixels = args->num_pixels;

#endif
	int last_pixel = pixel_start + thread_load;
	if( last_pixel > num_pixels)
		last_pixel = num_pixels;

	for( int i = pixel_start; i < last_pixel; i++ )
	{
		if( fabs( meas_data[2*i] ) > 1e-20 || fabs( meas_data[2*i+1] ) > 1e-20 )
		{
#ifdef TCR_KERNEL_CUDA
			gradient[2*i] = ( gradient[2*i] - meas_data[2*i] ) / image_size;
			gradient[2*i+1] = ( gradient[2*i+1] - meas_data[2*i+1] ) / image_size;
#else
			gradient[2*i] = ( gradient[2*i] - meas_data[2*i] );
			gradient[2*i+1] = ( gradient[2*i+1] - meas_data[2*i+1] );
#endif
		}
		else
		{
			gradient[2*i] = 0;
			gradient[2*i+1] = 0;
		}
	}
}

#ifdef TCR_KERNEL_CUDA
__global__ void CUDA_CalcTemporalGradient( float* gradient, float* estimate, int image_size, int num_phases, float beta, float beta_squared, int num_pixels, int thread_load )
{
	int pixel_start = ( blockIdx.x*blockDim.x + threadIdx.x ) * thread_load;
#else
void* CPU_CalcTemporalGradient( void* args_ptr )
{
	KernelArgs* args = (KernelArgs*) args_ptr;
	float* gradient = args->gradient;
	float* estimate = args->estimate;
	int image_size = args->image_size;
	int num_phases = args->temp_dim_size;
	float beta = args->beta;
	float beta_squared = args->beta_squared;
	int pixel_start = args->pixel_start;
	int thread_load = args->pixel_length;
	int num_pixels = args->num_pixels;

#endif
	int last_pixel = pixel_start + thread_load;
	if( last_pixel > num_pixels)
		last_pixel = num_pixels;

	for( int i = pixel_start; i < last_pixel; i++ )
	{
		// fft scale
		//gradient[2*i] /= image_size;
		//gradient[2*i+1] /= image_size;

		int image_num = i / image_size;
		int phase = image_num % num_phases;
		int next_phase = (phase+1) % num_phases;
		int prev_phase = (phase+num_phases-1) % num_phases;

		int idx_phase0 = i - ( phase * image_size );
		int idx_next_phase = idx_phase0 + ( next_phase * image_size );
		int idx_prev_phase = idx_phase0 + ( prev_phase * image_size );

		float grad1_real = estimate[2*i] - estimate[2*idx_next_phase];
		float grad1_imag = estimate[2*i+1] - estimate[2*idx_next_phase+1];
		float grad1_squared = grad1_real*grad1_real + grad1_imag*grad1_imag;
	
		float grad2_real = -estimate[2*i] + estimate[2*idx_prev_phase];
		float grad2_imag = -estimate[2*i+1] + estimate[2*idx_prev_phase+1];
		float grad2_squared = grad2_real*grad2_real + grad2_imag*grad2_imag;
	
		grad1_real = grad1_real / sqrt( grad1_squared + beta_squared );
		grad1_imag = grad1_imag / sqrt( grad1_squared + beta_squared );
	
		grad2_real = grad2_real / sqrt( grad2_squared + beta_squared );
		grad2_imag = grad2_imag / sqrt( grad2_squared + beta_squared );

		gradient[2*i] -= ( -grad1_real + grad2_real ) * beta;
		gradient[2*i+1] -= ( -grad1_imag + grad2_imag ) * beta;


		//gradient[2*i] += (2*estimate[2*i] - estimate[2*idx_next_phase] - estimate[2*idx_prev_phase]) * beta;
		//gradient[2*i+1] += (2*estimate[2*i+1] - estimate[2*idx_next_phase+1] - estimate[2*idx_prev_phase+1]) * beta;
	}
}

#ifdef TCR_KERNEL_CUDA
__global__ void CUDA_UpdateEstimate( float* gradient, float* estimate, int num_pixels, float step_size, int thread_load )
{
	int pixel_start = ( blockIdx.x*blockDim.x + threadIdx.x ) * thread_load;
#else
void* CPU_UpdateEstimate( void* args_ptr )
{
	KernelArgs* args = (KernelArgs*) args_ptr;
	float* gradient = args->gradient;
	float* estimate = args->estimate;
	float step_size = args->step_size;
	int pixel_start = args->pixel_start;
	int thread_load = args->pixel_length;
	int num_pixels = args->num_pixels;

#endif
	int last_pixel = pixel_start + thread_load;
	if( last_pixel > num_pixels)
		last_pixel = num_pixels;

	for( int i = pixel_start; i < last_pixel; i++ )
	{
		estimate[2*i] -= step_size * gradient[2*i];
		estimate[2*i+1] -= step_size * gradient[2*i+1];

		//! DEBUG
		//estimate[2*i] = gradient[2*i];
		//estimate[2*i+1] = gradient[2*i+1];
	}
}

#ifdef TCR_KERNEL_CUDA
__global__ void CUDA_LoadGradient( float* gradient, float* estimate, int num_pixels, int thread_load )
{
	int pixel_start = ( blockIdx.x*blockDim.x + threadIdx.x ) * thread_load;
#else
void* CPU_LoadGradient( void* args_ptr )
{
	KernelArgs* args = (KernelArgs*) args_ptr;
	float* gradient = args->gradient;
	float* estimate = args->estimate;
	float step_size = args->step_size;
	int pixel_start = args->pixel_start;
	int thread_load = args->pixel_length;
	int num_pixels = args->num_pixels;

#endif
	int last_pixel = pixel_start + thread_load;
	if( last_pixel > num_pixels)
		last_pixel = num_pixels;

	for( int i = pixel_start; i < last_pixel; i++ )
	{
		gradient[2*i] = estimate[2*i];
		gradient[2*i+1] = estimate[2*i+1];
	}
}
