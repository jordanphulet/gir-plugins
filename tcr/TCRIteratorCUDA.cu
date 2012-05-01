#include <TCRIteratorCUDA.h>
#include <GIRLogger.h>
#include <MRIData.h>
#include <MRIDataTool.h>
#include <FilterTool.h>
#include <KernelCode.h>
#include <vector>
#include <pthread.h>
#include <cmath>
#include <cuda.h>

#define CUDA_THREADS_PER_BLOCK 1024

TCRIteratorCUDA::TCRIteratorCUDA( int new_thread_load, TemporalDimension new_temp_dim ): 
	h_meas_data(0),
	h_estimate(0),
	d_meas_data(0),
	d_gradient(0),
	d_estimate(0),
	thread_load( new_thread_load ),
	cuda_device( -1 ),
	TCRIterator( new_temp_dim )
{
	GIRLogger::LogInfo( "TCRIteratorCUDA::TCRIteratorCUDA -> initializing on GPU...\n" );
}

bool CheckCUDAError( char *tag ) {
	cudaError_t error = cudaGetLastError();
	if( error != cudaSuccess) {
		GIRLogger::LogError( "TCR CUDA error, %s: %s!\n", tag, cudaGetErrorString( error ) );
		return false;
	}
	return true;
}

TCRIteratorCUDA::~TCRIteratorCUDA()
{
	if( h_meas_data != 0 ) { delete [] h_meas_data; h_meas_data = 0; }
	if( h_estimate != 0 ) { delete [] h_estimate; h_estimate = 0; }

	if( d_meas_data != 0 ) { cudaFree( d_meas_data ); d_meas_data = 0; }
	if( d_estimate  != 0 ) { cudaFree( d_estimate ); d_estimate = 0; }
	if( d_gradient  != 0 ) { cudaFree( d_gradient ); d_gradient = 0; }
	CheckCUDAError( "Load -> cudaFrees" );
}

void TCRIteratorCUDA::Load( float alpha, float beta, float beta_squared, float step_size, MRIData& src_meas_data, MRIData& src_estimate )
{
	// call parent
	TCRIterator::Load( alpha, beta, beta_squared, step_size, src_meas_data, src_estimate );

	// set cuda device
	if( cuda_device > -1 )
		cudaSetDevice( cuda_device );

	// make sure sizes are compatible
	if( !src_meas_data.Size().Equals( src_estimate.Size() ) )
	{
		GIRLogger::LogError( "TCRIteratorCUDA::Load -> src_meas_data is not the same size as src_estimate, aborting!\n" );
		return;
	}

	// clear any previous data on host
	if( h_meas_data != 0 ) delete[] h_meas_data;
	if( h_estimate  != 0 ) delete[] h_estimate;

	// clear any previous data on device
	if( d_meas_data != 0 ) cudaFree( d_meas_data );
	if( d_estimate  != 0 ) cudaFree( d_estimate );
	if( d_gradient  != 0 ) cudaFree( d_gradient );
	CheckCUDAError( "Load -> cudaFrees" );

	// allocate on host
	h_meas_data = new float[src_meas_data.NumElements()];
	h_estimate  = new float[src_estimate.NumElements()];

	// allocate on GPU
	//int gpu_bytes = ( src_meas_data.NumElements()*3 + src_coil_map.NumElements() ) * sizeof( float );
	//GIRLogger::LogInfo( "TCRIteratorCUDA::Load -> allocating %d bytes on GPU...\n", gpu_bytes );
	cudaMalloc( (void**)&d_meas_data,   src_meas_data.NumElements()   * sizeof( float ) );
	cudaMalloc( (void**)&d_estimate,    src_estimate.NumElements()    * sizeof( float ) );
	cudaMalloc( (void**)&d_gradient,    src_estimate.NumElements()    * sizeof( float ) );
	CheckCUDAError( "Load -> cudaMallocs" );

	// order on host
	Order( src_meas_data, h_meas_data );
	Order( src_estimate,  h_estimate );

	// copy to GPU
	cudaMemcpy( d_meas_data, h_meas_data, src_meas_data.NumElements() * sizeof( float ), cudaMemcpyHostToDevice ); 
	cudaMemcpy( d_estimate,  h_estimate,  src_estimate.NumElements()  * sizeof( float ), cudaMemcpyHostToDevice ); 
	CheckCUDAError( "Load -> cudaMemcpy" );

	// initialize args
	args.num_pixels = src_meas_data.NumPixels();
	args.image_size = src_meas_data.Size().Column * src_meas_data.Size().Line;
	args.channel_size = args.image_size * temp_dim_size;
	args.coil_channel_size = args.image_size;
	args.slice_size = args.channel_size * src_meas_data.Size().Channel;
	args.coil_slice_size = args.image_size * src_meas_data.Size().Channel;
	args.data_size = src_meas_data.Size();
	args.meas_data = d_meas_data;
	args.estimate = d_estimate;
	args.gradient = d_gradient;
	args.alpha = alpha;
	args.beta = beta;
	args.beta_squared = beta_squared;
	args.step_size = step_size;

	// set block/grid dimensions
	int threads_per_block  = CUDA_THREADS_PER_BLOCK;
	int blocks = (int)ceil( (float)args.num_pixels / ( CUDA_THREADS_PER_BLOCK * thread_load ) );

	GIRLogger::LogDebug( "### thread_load: %d, threads_per_block: %d, blocks: %d\n", thread_load, threads_per_block, blocks );

	dim_block = dim3( threads_per_block );
	dim_grid = dim3( blocks );

	// create CUFFT plan
	cufftPlan2d(&plan, src_meas_data.Size().Line, src_meas_data.Size().Column, CUFFT_C2C);

	GIRLogger::LogInfo( "TCRIteratorCUDA::load-> alpha: %f, beta: %f, beta_squared: %f, step_size: %f.\n", args.alpha, args.beta, args.beta_squared, args.step_size );
}

void TCRIteratorCUDA::Unload( MRIData& dest_estimate )
{
	GIRLogger::LogDebug( "### unloading\n" );
	if( d_estimate == 0 || h_estimate == 0 )
		GIRLogger::LogError( "TCRIteratorCUDA::Unload -> d_estimate == 0 || h_estimate == 0, unload aborting!\n" );
	else
	{
		// copy back to host
		cudaMemcpy( h_estimate, d_estimate, 2 * args.data_size.GetProduct() * sizeof( float ), cudaMemcpyDeviceToHost ); 
		CheckCUDAError( "Unload -> cudaMemcpy" );

		// unorder
		Unorder( dest_estimate, h_estimate );
	
		// free resources on GPU
		cudaFree( d_meas_data ); d_meas_data = 0;
		cudaFree( d_estimate );  d_estimate = 0;
		cudaFree( d_gradient );  d_gradient = 0;
		CheckCUDAError( "Unload -> cudaFrees" );
	}
}

void TCRIteratorCUDA::FFT() 
{
	int total_images = args.num_pixels / args.image_size;
	CUDA_FFTDirection( args.gradient, false, plan, args.image_size, total_images );
}

void TCRIteratorCUDA::IFFT() 
{
	int total_images = args.num_pixels / args.image_size;
	CUDA_FFTDirection( args.gradient, true, plan, args.image_size, total_images );
}

void TCRIteratorCUDA::ApplyFidelityDifference()
{
	CUDA_ApplyFidelityDifference<<< dim_grid, dim_block >>>( args.estimate, args.gradient, args.meas_data, args.num_pixels, thread_load );
	cudaThreadSynchronize();
	CheckCUDAError( "ApplyFidelityDifference" );
}

void TCRIteratorCUDA::CalcTemporalGradient()
{
	CUDA_CalcTemporalGradient<<< dim_grid, dim_block >>>( args.gradient, args.estimate, args.image_size, temp_dim_size, args.beta, args.beta_squared, args.num_pixels, thread_load );
	cudaThreadSynchronize();
	CheckCUDAError( "CalcTemporalGradient" );
}

void TCRIteratorCUDA::UpdateEstimate()
{
	CUDA_UpdateEstimate<<< dim_grid, dim_block >>>( args.gradient, args.estimate, args.num_pixels, args.step_size, thread_load );
	cudaThreadSynchronize();
	CheckCUDAError( "UpdateEstimate" );
}
