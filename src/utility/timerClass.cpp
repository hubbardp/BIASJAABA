#include "timerClass.hpp"

namespace bias {

    TimerClass::TimerClass()
    {
       
        
    }
    
    RtnStatus TimerClass::allocateTimers(unsigned int cameraNumber,QVariantMap& configMap)
    {
        RtnStatus rtnStatus;

        if (timerNIDAQFlag)
        {
            if (cameraNumber == 0) {

                nidaqTimerptr = make_shared<Lockable<NIDAQUtils>>();
                if(nidaqTimerptr!= nullptr && !configMap.isEmpty())
                {
                    rtnStatus = nidaqTimerptr->setNIDAQConfigFromMap(configMap);
                    if (!rtnStatus.success) {

                        rtnStatus.message = QString("NIDAQ Config error");
                        return rtnStatus;

                    }
                }
            }
            else 
            {
                nidaqTimerptr = nullptr;
            }
        }
        else 
        {
            pcTimerptr = make_shared<Lockable<GetTime>>();
        }

        rtnStatus.success = true;
        rtnStatus.message = QString("");
        return rtnStatus;
    }

    void TimerClass::getTimeNow()
    {

    }

}