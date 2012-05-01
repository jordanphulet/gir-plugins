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
	int gpu_num;

	MachineDesc( string new_address, int new_gpu_num ):
		address( new_address ),
		gpu_num( new_gpu_num )
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
			else if( key.compare( "input_file" ) == 0 )
				line_stream >> input_file;
			else if( key.compare( "output_file" ) == 0 )
				line_stream >> output_file;
			// machine
			else if( key.compare( "machine" ) == 0 )
			{
				string address;
				int gpu_num;
				line_stream >> address >> gpu_num;
				MachineDesc machine_desc( address, gpu_num );
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
		cout << "input_file: " << input_file << endl;
		cout << "output_file: " << output_file << endl;
		cout << "machines:\t" << endl;
		for( int i = 0; i < machine_descs.size(); i++ )
			cout << "\t" << machine_descs[i].address << ", GPU: " << machine_descs[i].gpu_num << endl;
	}
};

int main( int argc, char** argv )
{
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
	FileCommunicator communicator;
	if( !communicator.OpenInput( config.input_file.c_str() ) )
	{
		cerr << "Unable to load input file: '" << config.input_file << "!'" << endl;
		exit( EXIT_FAILURE );
	}
	MRIReconRequest request;
	MRIData mri_data;
	communicator.ReceiveReconRequest( request );
	communicator.ReceiveData( mri_data );

	// generate path prefix
	stringstream path_prefix_stream;
	path_prefix_stream << "aws-gpu-" << time(0);

	// split the data and execute
	int num_machines = config.machine_descs.size();
	int num_subsets = mri_data.Size().Channel * mri_data.Size().Slice;
	int subsets_per_machine = (int)ceil( (float)num_subsets / num_machines );
	cout << "num_machines: " << num_machines << endl;
	cout << "num_subsets: " << num_subsets << endl;
	cout << "subsets_per_machine: " << subsets_per_machine << endl;
	vector<pid_t> pids;
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
			string exec_path = "/DATA/aws-data/" + path_prefix_stream.str();
			cout << "executing on: " << this_desc.address << ", GPU: " << this_desc.gpu_num << ", path: " << exec_path << endl;

			// create tcr command
			stringstream tcr_command_stream;
			tcr_command_stream << "./atomic-tcr ";
			tcr_command_stream << config.input_file << " ";
			tcr_command_stream << config.alpha << " ";
			tcr_command_stream << config.beta << " ";
			tcr_command_stream << config.beta_sq << " ";
			tcr_command_stream << config.step_size << " ";
			tcr_command_stream << config.iterations << " ";
			tcr_command_stream << config.use_gpu << " ";
			tcr_command_stream << config.use_gpu << " ";
			//cout << tcr_command_stream.str() << endl;


			// generate payload
			stringstream payload_stream;
			payload_stream << "ssh " << this_desc.address << " mkdir -p " << exec_path << "/;" << endl;
			payload_stream << "scp aws-exec/* " << this_desc.address << ":" << exec_path << "/;" << endl;
			payload_stream << "scp " <<  config.input_file << " "<< this_desc.address << ":" << exec_path << "/;" << endl;
			payload_stream << "ssh " << this_desc.address << " \"(cd " << exec_path << "/; " << tcr_command_stream.str() << "&> atomic-tcr.out)\";" << endl;
			cout << "payload: " << endl << "----------" << endl << payload_stream.str() << endl;

			// execute payload
			system( payload_stream.str().c_str() );

			// create path
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

	// done
	cout << "all forks done, exiting..." << endl;
	return EXIT_SUCCESS;
}
