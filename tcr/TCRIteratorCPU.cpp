#include <TCRIteratorCPU.h>
#include <GIRLogger.h>
#include <MRIData.h>
#include <MRIDataTool.h>
#include <FilterTool.h>
#include <KernelCode.h>
#include <vector>
#include <pthread.h>
#include <cmath>

TCRIteratorCPU::~TCRIteratorCPU()
{
	if( meas_data != 0 ) { delete [] meas_data; meas_data = 0; }
	if( estimate != 0 ) { delete [] estimate; estimate = 0; }
	if( gradient != 0 ) { delete [] gradient; gradient = 0; }
	if( coil_map != 0 ) { delete [] coil_map; coil_map = 0; }
	if( lambda_map != 0 ) { delete [] lambda_map; lambda_map = 0; }
}

void TCRIteratorCPU::Load( float alpha, float beta, float beta_squared, float step_size, MRIData& src_meas_data, MRIData& src_estimate )
{
	// call parent
	TCRIterator::Load( alpha, beta, beta_squared, step_size, src_meas_data, src_estimate );

	// make sure sizes are compatible
	if( !src_meas_data.Size().Equals( src_estimate.Size() ) )
	{
		GIRLogger::LogError( "TCRIteratorCPU::Load -> src_meas_data is not the same size as src_estimate, aborting!\n" );
		return;
	}

	// load meas_data
	if( meas_data != 0 ) delete[] meas_data;
	meas_data = new float[src_meas_data.NumElements()];
	Order( src_meas_data, meas_data );

	// load estimate
	if( estimate != 0 ) delete[] estimate;
	estimate = new float[src_estimate.NumElements()];
	Order( src_estimate, estimate );

	// allocate gradient
	gradient = new float[src_estimate.NumElements()];

	// set max_pixel and pixels_per_thread
	int num_pixels = src_meas_data.NumPixels();
	int pixels_per_thread = (int)ceil( (float)num_pixels / num_threads );

	// initialize thread data
	for( int i = 0; i < num_threads; i++ )
	{
		args[i].thread_idx = i;
		args[i].num_threads = num_threads;
		args[i].pixel_start = i * pixels_per_thread;
		args[i].pixel_length = pixels_per_thread;
		args[i].num_pixels = num_pixels;
		args[i].image_size = src_meas_data.Size().Column * src_meas_data.Size().Line;
		args[i].channel_size = args[i].image_size * temp_dim_size;
		args[i].coil_channel_size = args[i].image_size;
		args[i].slice_size = args[i].channel_size * src_meas_data.Size().Channel;
		args[i].coil_slice_size = args[i].image_size * src_meas_data.Size().Channel;
		args[i].data_size = src_meas_data.Size();
		args[i].temp_dim_size = temp_dim_size;
		args[i].meas_data = meas_data;
		args[i].estimate = estimate;
		args[i].gradient = gradient;
		args[i].alpha = alpha;
		args[i].beta = beta;
		args[i].beta_squared = beta_squared;
		args[i].step_size = step_size;
	}
}

void TCRIteratorCPU::Unload( MRIData& dest_estimate )
{
	if( estimate == 0 )
		GIRLogger::LogError( "TCRIteratorCPU::Unload -> estimate == 0, unload aborting!\n" );
	else
		Unorder( dest_estimate, estimate );
}

void TCRIteratorCPU::LoadGradient()
{
	// create threads
	for( int i = 0; i < num_threads; i++ )
		pthread_create( &pthreads[i], NULL, CPU_LoadGradient, (void*)(&args[i]) );
	// join threads
	for( int i = 0; i < num_threads; i++ )
		pthread_join( pthreads[i], NULL );
}

void TCRIteratorCPU::FFT() 
{
	// create threads
	for( int i = 0; i < num_threads; i++ )
		pthread_create( &pthreads[i], NULL, CPU_FFT, (void*)(&args[i]) );
	// join threads
	for( int i = 0; i < num_threads; i++ )
		pthread_join( pthreads[i], NULL );
}

void TCRIteratorCPU::IFFT() 
{
	// create threads
	for( int i = 0; i < num_threads; i++ )
		pthread_create( &pthreads[i], NULL, CPU_IFFT, (void*)(&args[i]) );
	// join threads
	for( int i = 0; i < num_threads; i++ )
		pthread_join( pthreads[i], NULL );
}

void TCRIteratorCPU::ApplyFidelityDifference()
{
	// create threads
	for( int i = 0; i < num_threads; i++ )
		pthread_create( &pthreads[i], NULL, CPU_ApplyFidelityDifference, (void*)(&args[i]) );
	// join threads
	for( int i = 0; i < num_threads; i++ )
		pthread_join( pthreads[i], NULL );
}

void TCRIteratorCPU::CalcTemporalGradient()
{
	// create threads
	for( int i = 0; i < num_threads; i++ )
		pthread_create( &pthreads[i], NULL, CPU_CalcTemporalGradient, (void*)(&args[i]) );
	// join threads
	for( int i = 0; i < num_threads; i++ )
		pthread_join( pthreads[i], NULL );
}

void TCRIteratorCPU::UpdateEstimate()
{
	// create threads
	for( int i = 0; i < num_threads; i++ )
		pthread_create( &pthreads[i], NULL, CPU_UpdateEstimate, (void*)(&args[i]) );
	// join threads
	for( int i = 0; i < num_threads; i++ )
		pthread_join( pthreads[i], NULL );
}
