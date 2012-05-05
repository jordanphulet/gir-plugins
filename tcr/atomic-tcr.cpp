#include <FileCommunicator.h>
#include <GIRConfig.h>
#include <GIRLogger.h>
#include <MRIDataTool.h>
//#include <Serializable.h>
#include <MRIDataComm.h>
#include <RadialGridder.h>
#include <MRIDataSplitter.h>
#include <MPIPartitioner.h>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <math.h>

#include <TCRIteratorCPU.h>
#ifndef NO_CUDA
	#include <TCRIteratorCUDA.h>
#endif

bool Reconstruct( MRIData& data, float alpha, float beta, float beta_squared, float step_size, int iterations, bool use_gpu, int cuda_device, int cpu_threads )
{
	GIRLogger::LogInfo( "reconstructing (%s)...\n", data.Size().ToString().c_str() );
	int gpu_thread_load = 1;
	
	// regrid
	{
		RadialGridder gridder;
		gridder.repetition_offset = 0;
		gridder.view_ordering = RadialGridder::VO_GOLDEN_RATIO;
		MRIData temp_data;
		GIRLogger::LogInfo( "gridding...\n" );
		if( !gridder.Grid( data, temp_data, RadialGridder::KERN_TYPE_BILINEAR, 101, true ) )
		{
			GIRLogger::LogError( "Regridding failed, aborting!\n" );
			exit( EXIT_FAILURE );
		}
		data = temp_data;
	}
	
	// shift to corner
	FilterTool::FFTShift( data );

	// generate original estimate
	GIRLogger::LogInfo( "generating original estimate...\n" );
	MRIData estimate;
	MRIDataTool::TemporallyInterpolateKSpace( data, estimate );
	FilterTool::FFT2D( estimate, true );

	// iterate
	if( use_gpu )
	{
#ifndef NO_CUDA
		GIRLogger::LogInfo( "reconstructing on GPU...\n" );
		TCRIteratorCUDA iterator( gpu_thread_load, TCRIterator::TEMP_DIM_REP );
		iterator.cuda_device = cuda_device;
		iterator.Load( alpha, beta, beta_squared, step_size, data, estimate );
		iterator.Iterate( iterations );
		iterator.Unload( data );
#else
		GIRLogger::LogError( "GPU requested but binaries not build with CUDA, aborting!\n" );
		return false;
#endif
	}
	else
	{
		GIRLogger::LogInfo( "reconstructing on CPU(s)...\n" );
		TCRIteratorCPU iterator( cpu_threads, TCRIterator::TEMP_DIM_REP );
		iterator.Load( alpha, beta, beta_squared, step_size, data, estimate );
		iterator.Iterate( iterations );
		iterator.Unload( data );
	}

	// shift to center
	FilterTool::FFTShift( data, true );
	return true;
}


void Execute( const char* input_file, const char* output_file, float alpha, float beta, float beta_squared, float step_size, int iterations, bool use_gpu, int cuda_device, int cpu_threads )
{
	printf( "parameters:\n\talpha %f, beta %f, step_size %f, iterations %d, use_gpu %d, cuda_device: %d, cpu_threads: %d\n", alpha, beta, step_size, iterations, use_gpu, cuda_device, cpu_threads );
		
	// open file communicator
	printf( "opening %s...\n", input_file );
	FileCommunicator communicator;
	if( !communicator.OpenInput( input_file ) || !communicator.OpenOutput( output_file ) )
	{
		GIRLogger::LogError( "Unable to open IO files!\n" );
		exit( EXIT_FAILURE );
	}

	// get request, we don't use right now but we need to get it out of the way
	GIRLogger::LogInfo( "Loading data...\n" );
	MRIReconRequest request;
	communicator.ReceiveReconRequest( request );

	// get data
	MRIData data;
	communicator.ReceiveData( data );

	// reconstruct
	Reconstruct( data, alpha, beta, beta_squared, step_size, iterations, use_gpu, cuda_device, cpu_threads );

	// write output
	GIRLogger::LogInfo( "Writing output...\n" );
	communicator.SendData( data );
	GIRLogger::LogInfo( "Done.\n" );
}

int main( int argc, char** argv )
{
	// check args
	if( argc != 11 )
	{
		fprintf( stderr, "USAGE: atomic-tcr INPUT_FILE OUTPUT_FILE ALPHA BETA BETA_SQUARED STEP_SIZE ITERATIONS USE_GPU CUDA_DEVICE CPU_THREADS\n" );
		exit( EXIT_FAILURE );
	}

	// read in arguments
	const char* input_file = argv[1];
	double alpha;
	double beta;
	double beta_squared;
	double step_size;
	int iterations;
	int use_gpu;
	int cuda_device;
	int cpu_threads;
	std::stringstream str;
	str << " " << argv[3] << " " << argv[4] << " " << argv[5] << " " << argv[6] << " " << argv[7] << " " << argv[8] << " " << argv[9] << " " << argv[10];
	str >> alpha >> beta >> beta_squared >> step_size >> iterations >> use_gpu >> cuda_device >> cpu_threads;
	
	// execute
	GIRLogger::LogInfo( "starting...\n" );
	Execute( argv[1], argv[2], alpha, beta, beta_squared, step_size, iterations, use_gpu, cuda_device, cpu_threads );

	exit( EXIT_SUCCESS );
}
