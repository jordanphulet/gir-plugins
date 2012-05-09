#include <cstdlib>
#include <iostream>
#include <MRIData.h>
#include <FileCommunicator.h>
#include <FilterTool.h>
#include <JTimer.h>
#include <sys/wait.h>

using namespace std;
int main( int argc, char** argv )
{
	JTimer my_timer;

	vector<pid_t> pids;
	for( int i = 0; i < 20; i++ )
	{
		// child
		pid_t pid = fork();
		// parent
		if( pid != 0 )
			pids.push_back( pid );
		// child
		else
		{
			sleep( 20 );
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

	float time_elapsed = my_timer.Stop();
	cout << "time elapsed: " << time_elapsed << endl;

	return EXIT_SUCCESS;
}
