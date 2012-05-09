#include <JTimer.h>
#include <iostream>

using namespace std;

JTimer::JTimer()
{
	Start();
}

void JTimer::Start()
{
	//clock_gettime( CLOCK_MONOTONIC, &start_ts );
	start_sec = time(0);
}

float JTimer::Stop()
{
	//clock_gettime( CLOCK_MONOTONIC, &end_ts );
	//int sec_diff = end_ts.tv_sec - start_ts.tv_sec;
	//int nsec_diff = end_ts.tv_nsec - start_ts.tv_nsec;
	//return (float)sec_diff + (float)nsec_diff / (float)1e6;
	end_sec = time(0);
	return (float)(end_sec - start_sec);
}
