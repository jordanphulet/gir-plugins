#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <ctime>
#include <vector>
#include <math.h>
#include <sys/wait.h>
#include <GIRLogger.h>
#include <FileCommunicator.h>
#include <MRIDataComm.h>
#include <MRIData.h>

using namespace std;

class MachineDesc
{
	public:
	string address;
	int port;
	int cuda_device;
	string authentication;

	MachineDesc( string new_address, int new_port, int new_cuda_device, string new_authentication ):
		address( new_address ),
		port( new_port ),
		cuda_device( new_cuda_device ),
		authentication( new_authentication )
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

			// commented line
			if( key.length() > 0 && key[0] == '#' )
				continue;

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
				int port;
				int cuda_device;
				string authentication;
				line_stream >> address >> port >> cuda_device >> authentication;
				MachineDesc machine_desc( address, port, cuda_device, authentication );
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
		{
			cout << "\t" << machine_descs[i].address << ":" << machine_descs[i].port << ", GPU: " << machine_descs[i].cuda_device;
			if( machine_descs[i].authentication.length() > 0 )
				cout << ", authentication: " << machine_descs[i].authentication;
			cout << endl;
		}
	}
};

bool CopySubData( MRIData& full_data, MRIData& sub_data, int channel, int slice, bool full_to_sub )
{
	if( full_data.Size().Channel <= channel || full_data.Size().Slice <= slice )
	{
		cerr << "GetDataSubset -> invalid channel or slice!" << endl;
		return false;
	}

	// resize sub_data (only if we are copying from full data)
	if( full_to_sub )
	{
		MRIDimensions sub_dims = full_data.Size();
		sub_dims.Channel = 1;
		sub_dims.Slice = 1;
		sub_data = MRIData( sub_dims, full_data.IsComplex() );
	}

	// copy data
	int copy_size = full_data.Size().Column * sizeof( float );
	if( full_data.IsComplex() )
		copy_size *= 2;

	for( int line = 0;	line < full_data.Size().Line; line++ )
	for( int set = 0; set < full_data.Size().Set; set++ )
	for( int phase = 0; phase < full_data.Size().Phase; phase++ )
	for( int echo = 0; echo < full_data.Size().Echo; echo++ )
	for( int repetition = 0; repetition < full_data.Size().Repetition; repetition++ )
	for( int segment = 0; segment < full_data.Size().Segment; segment++ )
	for( int partition = 0; partition < full_data.Size().Partition; partition++ )
	for( int average = 0; average < full_data.Size().Average; average++ )
	{
		float* full_index = full_data.GetDataIndex( 0, line, channel, set, phase, slice, echo, repetition, segment, partition, average );
		float* sub_index = sub_data.GetDataIndex( 0, line, 0, set, phase, 0, echo, repetition, segment, partition, average );
		if( full_to_sub )
			memcpy( sub_index, full_index, copy_size );
		else
			memcpy( full_index, sub_index, copy_size );
	}
	return true;
}

bool NodeRecon( int node_num, MachineDesc& desc, ReconConfig& config, string exec_path, MRIData& input_data )
{
	string auth_string = "";
	if( desc.authentication.length() > 0 )
		auth_string = "-i " + desc.authentication + " ";

	// make recon directory and copy over binaries
	stringstream prime_stream;
	prime_stream << "ssh " << auth_string << "-p " << desc.port << " " << desc.address << " mkdir -p " << exec_path << "/;" << endl;
	prime_stream << "scp " << auth_string << "-P " << desc.port << " aws-bins/* " << desc.address << ":" << exec_path << "/;" << endl;
	cout << "priming:" << endl << prime_stream.str() << endl;
	system( prime_stream.str().c_str() );

	int num_machines = config.machine_descs.size();
	int num_subsets = input_data.Size().Channel * input_data.Size().Slice;
	int subsets_per_machine = (int)ceil( (float)num_subsets / num_machines );

	for( int subset = 0; subset < subsets_per_machine; subset++ )
	{
		int real_subset = subset + (node_num * subsets_per_machine);
		if( real_subset >= num_subsets )
			break;

		// split the data
		int sub_channel = real_subset % input_data.Size().Channel;
		int sub_slice = real_subset / input_data.Size().Channel;
		MRIData sub_data;
		if( !CopySubData( input_data, sub_data, sub_channel, sub_slice, true ) )
		{
			cerr << "GetDataSubset failed!\n" << endl;
			return false;
		}

		// write data to disk
		stringstream sub_data_file;
		sub_data_file << config.input_file << ".ch_" << sub_channel << ".sl_" << sub_slice;
		string sub_data_path = config.host_io_dir + sub_data_file.str();
		FileCommunicator::Write( sub_data, sub_data_path );

		// copy data to node
		stringstream copy_stream;
		copy_stream << "scp " << auth_string << "-P " << desc.port << " " <<  config.host_io_dir << sub_data_file.str() << " "<< desc.address << ":" << exec_path << "/;" << endl;
		cout << "copying:" << endl << copy_stream.str() << endl;
		system( copy_stream.str().c_str() );
	
		// create tcr command
		stringstream tcr_command_stream;
		tcr_command_stream << "export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH; ";
		tcr_command_stream << "./atomic-tcr ";
		tcr_command_stream << sub_data_file.str() << " ";
		tcr_command_stream << sub_data_file.str() + ".out" << " ";
		tcr_command_stream << config.alpha << " ";
		tcr_command_stream << config.beta << " ";
		tcr_command_stream << config.beta_sq << " ";
		tcr_command_stream << config.step_size << " ";
		tcr_command_stream << config.iterations << " ";
		tcr_command_stream << config.use_gpu << " ";
		tcr_command_stream << desc.cuda_device << " ";
		tcr_command_stream << config.cpu_threads << " ";

		// execute
		stringstream exec_stream;
		exec_stream << "ssh " << auth_string << "-p " << desc.port << " " << desc.address << " \"(cd " << exec_path << "/; " << tcr_command_stream.str() << "&> atomic-tcr.out)\";" << endl;
		cout << "executing:" << endl << exec_stream.str() << endl;
		system( exec_stream.str().c_str() );

		// copy reconstructed data to host
		stringstream recopy_stream;
		recopy_stream << "scp " << auth_string << "-P " << desc.port << " " << desc.address << ":" << exec_path << "/" << sub_data_file.str() << ".out " <<  config.host_io_dir << ";" << endl;
		cout << "recopying:" << endl << recopy_stream.str() << endl;
		system( recopy_stream.str().c_str() );
	}
	return true;
}

int main( int argc, char** argv )
{
	time_t time_begin = time( 0 );

	GIRLogger::Instance()->LogToFile( "gir_log.out" );

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
	MRIData input_data;
	FileCommunicator::Read( input_data, full_input_path );

	// generate path prefix
	stringstream path_prefix_stream;
	path_prefix_stream << "aws-gpu-" << time(0);

	// fork to execute on all machines
	int num_machines = config.machine_descs.size();
	int num_subsets = input_data.Size().Channel * input_data.Size().Slice;
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
			stringstream exec_path;
			exec_path << config.node_io_dir << path_prefix_stream.str() << "-" << config.machine_descs[i].cuda_device;
			if( NodeRecon( i, config.machine_descs[i], config, exec_path.str(), input_data ) )
				return EXIT_SUCCESS;
			else
				return EXIT_FAILURE;
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

	// gather the data
	MRIDimensions output_dims = input_data.Size();
	output_dims.Line = output_dims.Column;
	MRIData output_data( output_dims, input_data.IsComplex() );
	for( int i = 0; i < num_machines; i++ )
	{
		for( int subset = 0; subset < subsets_per_machine; subset++ )
		{
			// get indices
			int real_subset = subset + (i * subsets_per_machine);
			if( real_subset >= num_subsets )
				break;
			int sub_channel = real_subset % input_data.Size().Channel;
			int sub_slice = real_subset / input_data.Size().Channel;

			// load the sub data
			stringstream sub_data_file;
			sub_data_file << config.host_io_dir << config.input_file << ".ch_" << sub_channel << ".sl_" << sub_slice << ".out";
			cout << "loading: " << sub_data_file.str() << endl;
			MRIData sub_data;
			if( !FileCommunicator::Read( sub_data, sub_data_file.str() ) )
			{
				cerr << "Unable to read sub data: " << sub_data_file.str() << "!" << endl;
				return EXIT_FAILURE;
			}

			// copy into putput
			if( !CopySubData( output_data, sub_data, sub_channel, sub_slice, false ) )
			{
				cerr << "CopySubData failed for output_data!\n" << endl;
				return EXIT_FAILURE;
			}
		}
	}

	// write final output
	FileCommunicator::Write( output_data, config.host_io_dir + config.output_file );

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
