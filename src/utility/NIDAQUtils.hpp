#ifndef NIDAQUTILS_HPP
#define NIDAQUTILS_HPP
#include "stdio.h"
#include "NIDAQmx.h"

static void NIDAQError(int err);
#define DAQmxErrChk(functionCall) {int error = 0; if (DAQmxFailed(error = (functionCall))) NIDAQError(error);}

static void NIDAQError(int error)
{
    char errBuff[2048] = { '\0' };
    if (DAQmxFailed(error))
        DAQmxGetExtendedErrorInfo(errBuff, 2048);
    if (DAQmxFailed(error))
        printf("DAQmx Error: %s\n", errBuff);
}

namespace bias
{
   
    class NIDAQUtils {
        
    public:

        float64  dataf_high[1000], dataf_low[1000], datas_high[1000], datas_low[1000];
        uInt8 data[1000];

        TaskHandle taskHandle_fout = 0;
        TaskHandle taskHandle_sampout = 0;
        TaskHandle taskHandle_trigger_in = 0;
        TaskHandle taskHandle_grab_in = 0;
        TaskHandle taskHandle_start_signal = 0;

        NIDAQUtils();   
        void initialize();
        void configureNIDAQ();
        void startTasks();
        void start_trigger_signal();
        void Cleanup();
        ~NIDAQUtils();

        
    };

  
}
#endif
