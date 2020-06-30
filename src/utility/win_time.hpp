#ifndef WIN_TIME_HPP
#define WIN_TIME_HPP

//#include "stdafx.h"
#include <time.h>
#include <ctime>
#include <windows.h>

//using namespace System;
using namespace std;
 
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif
namespace bias 
{

	class GetTime
	{

	    public:                 

	        long long unsigned int secs;
		    long long unsigned int usec;
		    time_t curr_time;
		    timeval tv;


		    GetTime(long long unsigned int secs, long long unsigned int usec);
			

		    struct timezone
		    {
		        int  tz_minuteswest; /* minutes W of Greenwich */
	            int  tz_dsttime;     /* type of dst correction */
	        };

	       // Definition of a gettimeofday function
		   int getdaytime(struct timeval *tv, struct timezone *tz);

	};

}
#endif


//Example usage to for interval timing:

// Test routine for calculating a time difference
/*int main()
{
  struct timeval timediff;
  char dummychar;
 
// Do some interval timing
  gettimeofday(&timediff, NULL);
  double t1=timediff.tv_sec+(timediff.tv_usec/1000000.0);

// Waste a bit of time here
  cout << "Give me a keystroke and press Enter." << endl;
  cin >> dummychar;
  cout << "Thanks." << endl;
 
//Now that you have measured the user's reaction time, display the results in seconds
  gettimeofday(&timediff, NULL);
  double t2=timediff.tv_sec+(timediff.tv_usec/1000000.0);
  cout << t2-t1 << " seconds have elapsed" << endl;
  return 0;
}*/
    


