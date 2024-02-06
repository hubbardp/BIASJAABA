#ifndef TIMERCLASS_HPP
#define TIMERCLASS_HPP

#include "rtn_status.hpp"
#include "lockable.hpp"
#include "win_time.hpp"
#include "NIDAQUTILS.hpp"

#include <QVariantMap>



namespace bias {

    class TimerClass
    {
        public:

            TimerClass();

            bool timerNIDAQFlag; // false implies pc time
                                         // true implies use nidaq for timing
            bool cameraMode; // false implies internal trigger mode for camera
                                     // true implies external trigger mode for camera
            bool isExternalTrig; // is trigger on or off
            bool* isExternalTrigPtr; 

            std::shared_ptr<Lockable<NIDAQUtils>>nidaqTimerptr;
            std::shared_ptr<Lockable<GetTime>>pcTimerptr;

            /*virtual RtnStatus configureTimer(unsigned int cameraNumber, QVariantMap& configMap);
            virtual bool isTriggered();
            virtual bool isTaskStarted();
            virtual void startTimerTasks();
            virtual void startTimerTrigger();
            virtual void stopTimerTrigger();
            virtual void stopTimerTasks();
            virtual void clearTimerTasks();*/

            uint64_t getTimeNow();
            void setExternalTriggerFlag();

    };
}
#endif