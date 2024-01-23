#ifndef TIMERCLASS_HPP
#define TIMERCLASS_HPP

#include "win_time.hpp"
#include "NIDAQUTILS.hpp"
#include "rtn_status.hpp"


namespace bias {

    class TimerClass
    {
        public:

            TimerClass();

            bool timerNIDAQFlag = false; // false implies pc time
                                         // true implies use nidaq for timing

            std::shared_ptr<Lockable<GetTime>> pcTimerptr = nullptr;
            std::shared_ptr<Lockable<NIDAQUtils>> nidaqTimerptr = nullptr;
           
            RtnStatus allocateTimers(unsigned int cameraNumber, QVariantMap& configMap);
            void getTimeNow();

    };
}
#endif