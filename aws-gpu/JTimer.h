#ifndef JTIMER_H
#define JTIMER_H

#include <time.h>

class JTimer
{
	public:
	JTimer();

	void Start();
	float Stop();

	private:
	//timespec start_ts;
	//timespec end_ts;
	int start_sec;
	int end_sec;
};

#endif
