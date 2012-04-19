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

bool Reconstruct( MRIData& data, float alpha, float beta, float beta_squared, float step_size, int iterations, bool use_gpu, int rep_offset, int cpu_threads )
{
	GIRLogger::LogInfo( "reconstructing (%s)...\n", data.Size().ToString().c_str() );
	int gpu_thread_load = 2;
	//int threads = 16;
	int threads = cpu_threads;
	
	//GIRLogger::LogDebug( "### just gridding...\n" );

	// regrid
	{
		RadialGridder gridder;
		gridder.repetition_offset = rep_offset;
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

	// generate coil map
	MRIData coil_map;
	GIRLogger::LogInfo( "generating coil map...\n" );
	MRIDataTool::GetCoilMap( data, coil_map );

	// generate original estimate
	GIRLogger::LogInfo( "generating original estimate...\n" );
	MRIData estimate;
	MRIDataTool::TemporallyInterpolateKSpace( data, estimate );
	FilterTool::FFT2D( estimate, true );
	estimate.MakeAbs();

	// bogus lambda map for now...
	MRIData lambda_map;
	MRIDimensions lambda_dims( estimate.Size().Column, estimate.Size().Line, 1, 1, 1, 1, 1, 1, 1, 1, 1 );
	lambda_map = MRIData( lambda_dims, false );
	lambda_map.SetAll( 1 );

	// iterate
	if( use_gpu )
	{
#ifndef NO_CUDA
		GIRLogger::LogInfo( "reconstructing on GPU...\n" );
		TCRIteratorCUDA iterator( gpu_thread_load, TCRIterator::TEMP_DIM_REP );
		iterator.Load( alpha, beta, beta_squared, step_size, data, estimate, coil_map, lambda_map );
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
		TCRIteratorCPU iterator( threads, TCRIterator::TEMP_DIM_REP );
		iterator.Load( alpha, beta, beta_squared, step_size, data, estimate, coil_map, lambda_map );
		iterator.Iterate( iterations );
		iterator.Unload( data );
	}

	// shift to center
	FilterTool::FFTShift( data, true );
	return true;
}


void ExecuteMaster( const char* input_file, float alpha, float beta, float beta_squared, float step_size, int iterations, bool use_gpu, int chunks, int cpu_threads )
{
	printf( "parameters:\n\talpha %f, beta %f, step_size %f, iterations %d, use_gpu %d, chunks %d\n", alpha, beta, step_size, iterations, use_gpu, chunks );
		
	// open file communicator
	printf( "opening %s...\n", input_file );
	FileCommunicator communicator;
	if( !communicator.OpenInput( input_file ) || !communicator.OpenOutput( "tcr_data.out" ) )
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

	MRIData new_data;
	bool new_data_initialized = false;

	// reconstruct chunks
	int overlap = 3;
	for( int i = 0; i < chunks; i++ )
	{
		// split off data
		MRIData split_data;
		int split_start;
		int split_end;
		if( !MPIPartitioner::SplitRepetitions( data, split_data, i, chunks, overlap, split_start, split_end ) )
		{
			GIRLogger::LogError( "SplitRepetitions failed for task: %d!\n", i );
			exit( EXIT_FAILURE );
		}
		
		// reconstruct
		Reconstruct( split_data, alpha, beta, beta_squared, step_size, iterations, use_gpu==1, split_start, cpu_threads );

		// resize due to gridding
		if( !new_data_initialized )
		{
			MRIDimensions new_dims = data.Size();
			new_dims.Line = split_data.Size().Line;
			new_dims.Column = split_data.Size().Column;
			new_data = MRIData( new_dims, true );
			new_data.SetAll( 0 );
			new_data_initialized = true;
		}

		// merge
		if( !MPIPartitioner::MergeRepetitions( new_data, split_data, i, chunks, overlap ) )
		{
			GIRLogger::LogError( "MergeRepetitions failed for task: %d!\n", i );
			exit( EXIT_FAILURE );
		}
	}

	// write output
	GIRLogger::LogInfo( "Writing output...\n" );
	communicator.SendData( new_data );
	GIRLogger::LogInfo( "Done.\n" );
}

int main( int argc, char** argv )
{
	// check args
	if( argc != 10 )
	{
		fprintf( stderr, "USAGE: serial-tcr INPUT_FILE ALPHA BETA BETA_SQUARED STEP_SIZE ITERATIONS USE_GPU CHUNKS CPU_THREADS\n" );
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
	int chunks;
	int cpu_threads;
	std::stringstream str;
	str << argv[2]  << " " << argv[3] << " " << argv[4] << " " << argv[5] << " " << argv[6] << " " << argv[7] << " " << argv[8] << " " << argv[9];
	str >> alpha >> beta >> beta_squared >> step_size >> iterations >> use_gpu >> chunks >> cpu_threads;
	
	// execute
	GIRLogger::LogInfo( "starting, %d total tasks...\n", chunks );
	ExecuteMaster( argv[1], alpha, beta, beta_squared, step_size, iterations, use_gpu, chunks, cpu_threads );

	exit( EXIT_SUCCESS );
}
