#ifndef NIDAQUTILS_HPP
#define NIDAQUTILS_HPP
#include "stdio.h"
#include "NIDAQmx.h"
#include <vector>
#include "lockable.hpp"
#include "rtn_status.hpp"
#include <string>

#include<QVariantMap>

using namespace std;

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

    static QVariantMap createNIDAQConfigMap()
    {
        QVariantMap nidaqconfigmap;
        nidaqconfigmap.insert("device_name", "");
        nidaqconfigmap.insert("fast_outchannel_counter", "");
        nidaqconfigmap.insert("sample_outchannel_counter", "");
        nidaqconfigmap.insert("frametrig_inchannel_counter", "");
        nidaqconfigmap.insert("framegrab_inchannel_counter", "");
        nidaqconfigmap.insert("frametrig_sampleclk", "");
        nidaqconfigmap.insert("frametrig_datachannel", "");
        nidaqconfigmap.insert("digital_trigger_signalout", "");
        nidaqconfigmap.insert("digital_trigger_sourcein", "");
        nidaqconfigmap.insert("fast_counter_rate", "");
        nidaqconfigmap.insert("sample_counter_rate", "");
        nidaqconfigmap.insert("numSamplesPerChan", "");
        return nidaqconfigmap;
    }

    /*class NIDAQConfig 
    {
    public:
        const char* device_name;
        const char* fast_outchannel_counter;
        const char* sample_outchannel_counter;
        const char* frametrig_inchannel_counter;
        const char* framegrab_inchannel_counter ;
        const char* frametrig_sampleclk;
        const char* frametrig_datachannel;
        const char* framegrab_datachannel;
        const char* digital_trigger_signalout;
        const char* digital_trigger_sourcein;
        float64 fast_counter_rate;
        float64 sample_counter_rate;
        uInt64 numsamplesPerChan;

        NIDAQConfig();
        void initialize();
        QVariantMap NIDAQConfigMap;
        RtnStatus fromMap();
        QVariantMap toMap();
    };*/
   
    class NIDAQUtils: public Lockable<Empty>
    {
   
      public:

        float64  *dataf_high, *dataf_low, *datas_high, *datas_low;
        uInt8 *data;
        std::vector<uInt32>cam_trigger;
        uInt32 read_buffer;
        uInt32 read_ondemand;
        bool start_tasks = false;
        bool istrig = false;
        //NIDAQConfig nidaq_config;

        // nidaq config variables
        string device_name;
        string fast_outchannel_counter;
        string sample_outchannel_counter;
        string frametrig_inchannel_counter;
        string framegrab_inchannel_counter;
        string frametrig_sampleclk;
        string frametrig_datachannel;
        string framegrab_datachannel;
        string digital_trigger_signalout;
        string digital_trigger_sourcein;
        float64 fast_counter_rate;
        float64 sample_counter_rate;
        uInt64 numsamplesPerChan;

        float64 half_period_cycle_fastclk; // divide the period of
        float64 half_period_cycle_sampleclk; // cycle by 2 to get half cycle time
        uint64_t bufsize_samp;
        uint64_t bufsize_fst;

        TaskHandle taskHandle_fout = 0;
        TaskHandle taskHandle_sampout = 0;
        TaskHandle taskHandle_trigger_in = 0;
        TaskHandle taskHandle_grab_in = 0;
        TaskHandle taskHandle_start_signal = 0;

        NIDAQUtils();   
        void initialize();
        RtnStatus setNIDAQConfigFromMap(QVariantMap& nidaqconfigMap);
        void configureNIDAQ();
        void startTasks();
        void stopTasks();
        void start_trigger_signal();
        void stop_trigger_signal();
        void Cleanup();
        void getCamtrig(unsigned int frameCount);
        void getNidaqTimeNow(uInt32& read_ondemand);
        bool getisTrue();
        ~NIDAQUtils();

    /*signals:

        void nidaqIstrig(uInt32 read_buffer);

    private slots: 

        void newFrametrig(uInt32 read_buffer);*/

        
    };

  
}
#endif
