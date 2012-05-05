#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <ctime>
#include <vector>
#include <math.h>
#include <sys/wait.h>
#include <FileCommunicator.h>
#include <MRIDataComm.h>
#include <MRIData.h>

using namespace std;

class MachineDesc
{
	public:
	string address;
	int cuda_device;

	MachineDesc( string new_address, int new_cuda_device ):
		address( new_address ),
		cuda_device( new_cuda_device )
	{
	}
};

class ReconConfig
{
	public:
	float alpha;
	float beta;
	float beta_sq;
	float step_size;
	float iterations;
	bool use_gpu;
	int cpu_threads;
	string host_io_dir;
	string node_io_dir;
	string input_file;
	string output_file;
	vector<MachineDesc> machine_descs;

	ReconConfig():
		alpha( 1.0 ),
		beta( 0.0001 ),
		beta_sq( 0.00000002 ),
		step_size( 0.08 ),
		iterations( 100 ),
		use_gpu( false ),
		cpu_threads( 1 ),
		host_io_dir( "/DATA" ),
		node_io_dir( "/DATA" ),
		input_file( "input.ser" ),
		output_file( "output.ser" )
	{
	}

	bool Load( const char* config_path )
	{
		ifstream file_in( config_path );
		if( !file_in.good() )
		{
			cerr << "ReconConfig::Load -> unable to read file: '" << config_path << "!'" << endl;
			return false;
		}

		char line_buffer[1024];

		while( file_in.getline( line_buffer, 1024 ) && file_in.good() )
		{
			string key;
			string value;
			stringstream line_stream( line_buffer );
			line_stream >> key;

			if( key.compare( "alpha" ) == 0 )
				line_stream >> alpha;
			else if( key.compare( "beta" ) == 0 )
				line_stream >> beta;
			else if( key.compare( "beta_sq" ) == 0 )
				line_stream >> beta_sq;
			else if( key.compare( "step_size" ) == 0 )
				line_stream >> step_size;
			else if( key.compare( "iterations" ) == 0 )
				line_stream >> iterations;
			else if( key.compare( "use_gpu" ) == 0 )
				line_stream >> use_gpu;
			else if( key.compare( "cpu_threads" ) == 0 )
				line_stream >> cpu_threads;
			else if( key.compare( "host_io_dir" ) == 0 )
				line_stream >> host_io_dir;
			else if( key.compare( "node_io_dir" ) == 0 )
				line_stream >> node_io_dir;
			else if( key.compare( "input_file" ) == 0 )
				line_stream >> input_file;
			else if( key.compare( "output_file" ) == 0 )
				line_stream >> output_file;
			// machine
			else if( key.compare( "machine" ) == 0 )
			{
				string address;
				int cuda_device;
				line_stream >> address >> cuda_device;
				MachineDesc machine_desc( address, cuda_device );
				machine_descs.push_back( machine_desc );
			}
			else if( key.compare( "" ) != 0 )
				cout << "WARNING: unknown key in config '" << key << "'!" << endl;
		}

		return true;
	}

	void Print()
	{
		cout << "alpha: " << alpha << endl;
		cout << "beta: " << beta << endl;
		cout << "beta_sq: " << beta_sq << endl;
		cout << "step_size: " << step_size << endl;
		cout << "iterations: " << iterations << endl;
		cout << "use_gpu: " << use_gpu << endl;
		cout << "cpu_threads: " << cpu_threads << endl;
		cout << "host_io_dir: " << host_io_dir << endl;
		cout << "node_io_dir: " << node_io_dir << endl;
		cout << "input_file: " << input_file << endl;
		cout << "output_file: " << output_file << endl;
		cout << "machines:\t" << endl;
		for( int i = 0; i < machine_descs.size(); i++ )
			cout << "\t" << machine_descs[i].address << ", GPU: " << machine_descs[i].cuda_device << endl;
	}
};

bool GetDataSubset( MRIData& data_in, MRIData& data_out, int channel, int slice )
{
	if( data_in.Size().Channel <= channel || data_in.Size().Slice <= slice )
	{
		cerr << "GetDataSubset -> invalid channel or slice!" << endl;
		return false;
	}

	// resize data_out
	MRIDimensions sub_dims = data_in.Size();
	sub_dims.Channel = 1;
	sub_dims.Slice = 1;
	data_out = MRIData( sub_dims, data_in.IsComplex() );

	// copy data
	int copy_size = data_in.Size().Column * sizeof( float );
	if( data_in.IsComplex() )
		copy_size *= 2;

	for( int line = 0;	line < sub_dims.Line; line++ )
	for( int set = 0; set < sub_dims.Set; set++ )
	for( int phase = 0; phase < sub_dims.Phase; phase++ )
	for( int echo = 0; echo < sub_dims.Echo; echo++ )
	for( int repetition = 0; repetition < sub_dims.Repetition; repetition++ )
	for( int segment = 0; segment < sub_dims.Segment; segment++ )
	for( int partition = 0; partition < sub_dims.Partition; partition++ )
	for( int average = 0; average < sub_dims.Average; average++ )
	{
		float* full_index = data_in.GetDataIndex( 0, line, channel, set, phase, slice, echo, repetition, segment, partition, average );
		float* sub_index = data_out.GetDataIndex( 0, line, 0, set, phase, 0, echo, repetition, segment, partition, average );
		memcpy( sub_index, full_index, copy_size );
	}

	return true;
}

int main( int argc, char** argv )
{
	time_t time_begin = time( 0 );

	// check args
	if( argc != 2 )
	{
		fprintf( stderr, "USAGE: %s CONFIG_FILE\n", argv[0] );
		exit( EXIT_SUCCESS );
	}

	// load config file
	ReconConfig config;
	if( !config.Load( argv[1] ) )
	{
		cerr << "Unable to load config file: '" << argv[1] << "'!" << endl;
		exit( EXIT_FAILURE );
	}
	config.Print();
	
	// load the data
	string full_input_path = config.host_io_dir + config.input_file;
	FileCommunicator communicator;
	if( !communicator.OpenInput( full_input_path.c_str() ) )
	{
		cerr << "Unable to load input file: '" << full_input_path << "!'" << endl;
		exit( EXIT_FAILURE );
	}
	MRIReconRequest request;
	MRIData mri_data;
	communicator.ReceiveReconRequest( request );
	communicator.ReceiveData( mri_data );

	// generate path prefix
	stringstream path_prefix_stream;
	path_prefix_stream << "aws-gpu-" << time(0);

	// fork to execute on all machines
	int num_machines = config.machine_descs.size();
	int num_subsets = mri_data.Size().Channel * mri_data.Size().Slice;
	int subsets_per_machine = (int)ceil( (float)num_subsets / num_machines );
	cout << "num_machines: " << num_machines << endl;
	cout << "num_subsets: " << num_subsets << endl;
	cout << "subsets_per_machine: " << subsets_per_machine << endl;
	vector<pid_t> pids;

	time_t time_init_done = time( 0 );

	for( int i = 0; i < num_machines; i++ )
	{
		// child
		pid_t pid = fork();
		// parent
		if( pid != 0 )
			pids.push_back( pid );
		// child
		else
		{
			MachineDesc this_desc = config.machine_descs[i];
			// get path prefix
			path_prefix_stream << "-" << i;
			string exec_path = config.node_io_dir + path_prefix_stream.str();
			cout << "executing on: " << this_desc.address << ", GPU: " << this_desc.cuda_device << ", path: " << exec_path << endl;
			
			for( int subset = 0; subset < subsets_per_machine; subset++ )
			{
				int real_subset = subset + (i * subsets_per_machine);
				if( real_subset >= num_subsets )
					break;

				// split the data
				int sub_channel = real_subset % mri_data.Size().Channel;
				int sub_slice = real_subset / mri_data.Size().Channel;
				MRIData sub_data;
				if( !GetDataSubset( mri_data, sub_data, sub_channel, sub_slice ) )
				{
					cerr << "GetDataSubset failed!\n" << endl;
					return EXIT_FAILURE;
				}

				cout << "executing subset " << real_subset << " (" << sub_channel << ", " << sub_slice << "): " << subset << endl;

				// write data to disk
				stringstream sub_data_file;
				sub_data_file << config.input_file << ".ch_" << sub_channel << ".sl_" << sub_slice;
				string sub_data_path = config.host_io_dir + sub_data_file.str();
				FileCommunicator out_communicator;
				out_communicator.OpenOutput( sub_data_path.c_str() );
				MRIReconRequest request;
				request.pipeline = "nothing";
				out_communicator.SendReconRequest( request );
				out_communicator.SendData( sub_data );
	
				// create tcr command
				stringstream tcr_command_stream;
				tcr_command_stream << "./atomic-tcr ";
				tcr_command_stream << sub_data_file.str() << " ";
				tcr_command_stream << sub_data_file.str() + ".out" << " ";
				tcr_command_stream << config.alpha << " ";
				tcr_command_stream << config.beta << " ";
				tcr_command_stream << config.beta_sq << " ";
				tcr_command_stream << config.step_size << " ";
				tcr_command_stream << config.iterations << " ";
				tcr_command_stream << config.use_gpu << " ";
				tcr_command_stream << this_desc.cuda_device << " ";
				tcr_command_stream << config.cpu_threads << " ";
	
				// generate payload
				stringstream payload_stream;
				payload_stream << "ssh " << this_desc.address << " mkdir -p " << exec_path << "/;" << endl;
				payload_stream << "scp aws-bins/* " << this_desc.address << ":" << exec_path << "/;" << endl;
				payload_stream << "scp " <<  sub_data_path << " "<< this_desc.address << ":" << exec_path << "/;" << endl;
				payload_stream << "ssh " << this_desc.address << " \"(cd " << exec_path << "/; " << tcr_command_stream.str() << "&> atomic-tcr.out)\";" << endl;
				cout << "payload: " << endl << "----------" << endl << payload_stream.str() << endl;
	
				// execute payload
				system( payload_stream.str().c_str() );
			}

			// forked child is done
			return EXIT_SUCCESS;
		}
	}

	// wait for forks to finish
	for( int i = 0; i < pids.size(); ++i )
	{
		int status;
		while( waitpid( pids[i], &status, 0 ) == -1 );
		if( !WIFEXITED( status ) || WEXITSTATUS( status ) != 0 )
		{
			cerr << "Process " << i << " (pid " << pids[i] << ") failed" << endl;
        	return EXIT_FAILURE;
		}
	}

	time_t time_all_done = time( 0 );

	double init_time = time_init_done - time_begin;
	double recon_time = time_all_done - time_init_done;
	double total_time = time_all_done - time_begin;

	// done
	cout << "all forks done, exiting..." << endl;
	cout << "initialzation time: " << init_time << endl;
	cout << "reconstruction time: " << recon_time << endl;
	cout << "total time: " << total_time << endl;

	return EXIT_SUCCESS;
}
