#include "timerClass.hpp"
#include <iostream>

namespace bias {

    TimerClass::TimerClass()
    {
       
        
    }


    uint64_t TimerClass::getTimeNow()
    {
        uint64_t time_now = 0;
        uInt32 readOndemand = 0;

        if (timerNIDAQFlag && cameraMode) 
        {
            if (nidaqTimerptr != nullptr) {
                nidaqTimerptr->getNidaqTimeNow(readOndemand);
                time_now = static_cast<uint64_t>(readOndemand);
            }
            else {
                std::cout << "** nidaq timer is  NULL ** " << std::endl;
            }
        }
        else {
            if (pcTimerptr != nullptr)
            {
                time_now = pcTimerptr->getPCtime();
            }
            else {
                std::cout << "** pc timer  is  NULL **" << std::endl;
            }
        }

        return time_now;

    }
    

    /*RtnStatus TimerClass::configureTimer(unsigned int cameraNumber,QVariantMap& configMap)
    {
        RtnStatus rtnStatus;

        rtnStatus.success = true;
        rtnStatus.message = QString("");
        return rtnStatus;
    }

    bool TimerClass::isTriggered() { return true; }

    bool TimerClass::isTaskStarted() { return true; }

    void TimerClass::startTimerTasks() {}

    void TimerClass::startTimerTrigger() {}

    void TimerClass::stopTimerTasks() {}

    void TimerClass::stopTimerTrigger() {}

    void TimerClass::clearTimerTasks() {}
*/

}