#ifndef WIN_TIME_HPP
#define WIN_TIME_HPP

//#include "stdafx.h"
#include <time.h>
#include <ctime> //defines localtime 
#include <chrono>
#include "stamped_image.hpp"

#include <fstream>
//#include <iomanip>

#ifdef WIN32
#include<windows.h>
#endif

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

         unsigned long long secs;
         unsigned long long usec;
         time_t curr_time;
         //timeval tv;

         GetTime(long long unsigned int secs, long long unsigned int usec);
			
         struct timezone
         {
             int  tz_minuteswest; /* minutes W of Greenwich */
             int  tz_dsttime;     /* type of dst correction */
         };

         // Definition of a gettimeofday function
         //int getdaytime(struct timeval *tv, struct timezone *tz);
         std::chrono::system_clock::duration duration_since_midnight();
         TimeStamp getPCtime();

         // this is a hack to avoid linker errors in VS2017

         template <typename T>
         void write_time(std::string filename, int framenum, std::vector<std::vector<T>> timeVec)
         {

             std::ofstream x_out;
             x_out.open(filename.c_str(), std::ios_base::app);
         
             for (int frame_id = 0; frame_id < framenum-1; frame_id++)
             {

                 x_out << timeVec[frame_id][0] <<  "," << timeVec[frame_id][1] << "\n";
             }

             x_out.close();

         }


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
    


