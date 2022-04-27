#include "NIDAQUtils.hpp"

namespace bias {

    NIDAQUtils::NIDAQUtils()
    {
        initialize();
        configureNIDAQ();
       
    }

    void NIDAQUtils::initialize()
    {
        // initialize data buffers 
        for (int i = 0; i < 1000; ++i) {

            dataf_high[i] = 0.00001;
            dataf_low[i] = 0.00001;
            datas_high[i] = 0.00125;
            datas_low[i] = 0.00125;
            data[i] = (uInt8)(i % 2);
        }
    }

    void NIDAQUtils::configureNIDAQ() {

        // NIDAQ Configuration
        
        //fast source channel
        DAQmxErrChk(DAQmxCreateTask("", &taskHandle_fout));
        DAQmxErrChk(DAQmxCreateCOPulseChanTime(taskHandle_fout, "Dev1/ctr3", "", DAQmx_Val_Seconds, DAQmx_Val_Low, 0.0, 0.00001, 0.00001));
        DAQmxErrChk(DAQmxCfgImplicitTiming(taskHandle_fout, DAQmx_Val_ContSamps, 1000));
        DAQmxErrChk(DAQmxWriteCtrTime(taskHandle_fout, 1000, 0, 1.0, DAQmx_Val_GroupByChannel, dataf_high, dataf_low, NULL, NULL));

        // sample clock
        DAQmxErrChk(DAQmxCreateTask("", &taskHandle_sampout));
        DAQmxErrChk(DAQmxCreateCOPulseChanTime(taskHandle_sampout, "Dev1/ctr2", "", DAQmx_Val_Seconds, DAQmx_Val_Low, 0.0, 0.00125, 0.00125));
        DAQmxErrChk(DAQmxCfgImplicitTiming(taskHandle_sampout, DAQmx_Val_ContSamps, 1000));
        DAQmxErrChk(DAQmxWriteCtrTime(taskHandle_sampout, 1000, 0, 1.0, DAQmx_Val_GroupByChannel, datas_high, datas_low, NULL, NULL));

        //Measure frame trigger Channel
        DAQmxErrChk(DAQmxCreateTask("", &taskHandle_trigger_in));
        DAQmxErrChk(DAQmxCreateCICountEdgesChan(taskHandle_trigger_in, "Dev1/ctr0", "", DAQmx_Val_Rising, 0, DAQmx_Val_CountUp));
        DAQmxErrChk(DAQmxCfgSampClkTiming(taskHandle_trigger_in, "/Dev1/PFI14", 1000000.0, DAQmx_Val_Rising, DAQmx_Val_ContSamps, 1000000.0));
        DAQmxErrChk(DAQmxSetCICountEdgesTerm(taskHandle_trigger_in, "Dev1/ctr0", "/Dev1/PFI15"));

        // Measure frame grab Channel
        DAQmxErrChk(DAQmxCreateTask("", &taskHandle_grab_in));
        DAQmxErrChk(DAQmxCreateCICountEdgesChan(taskHandle_grab_in, "Dev1/ctr1", "", DAQmx_Val_Rising, 0, DAQmx_Val_CountUp));
        DAQmxErrChk(DAQmxSetCICountEdgesTerm(taskHandle_grab_in, "Dev1/ctr1", "/Dev1/PFI15"));

        // start trigger signal
        DAQmxErrChk(DAQmxCreateTask("", &taskHandle_start_signal));
        DAQmxErrChk(DAQmxCreateDOChan(taskHandle_start_signal, "/Dev1/port0/line0", "", DAQmx_Val_ChanPerLine));
        DAQmxErrChk(DAQmxCfgSampClkTiming(taskHandle_start_signal, "", 1000.0, DAQmx_Val_Rising, DAQmx_Val_ContSamps, 1000));
        DAQmxErrChk(DAQmxWriteDigitalLines(taskHandle_start_signal, 1000, 0, 10.0, DAQmx_Val_GroupByChannel, data, NULL, NULL));

        // Print application build information
        //printf("Application build date: %s %s \n\n", __DATE__, __TIME__);
        DAQmxErrChk(DAQmxCfgDigEdgeStartTrig(taskHandle_fout, "/Dev1/PFI0", DAQmx_Val_Rising));
        DAQmxErrChk(DAQmxCfgDigEdgeStartTrig(taskHandle_sampout, "/Dev1/PFI0", DAQmx_Val_Rising));

    }

    void NIDAQUtils::startTasks()
    {
        DAQmxErrChk(DAQmxStartTask(taskHandle_grab_in));
        DAQmxErrChk(DAQmxStartTask(taskHandle_trigger_in));
        DAQmxErrChk(DAQmxStartTask(taskHandle_fout));
        DAQmxErrChk(DAQmxStartTask(taskHandle_sampout));
        
    }

    void NIDAQUtils::start_trigger_signal() {

        DAQmxErrChk(DAQmxStartTask(taskHandle_start_signal));
    }

    void NIDAQUtils::Cleanup() {

        if (taskHandle_fout != 0)
        {
            /*********************************************/
            // DAQmx Stop Code
            /*********************************************/
            DAQmxStopTask(taskHandle_trigger_in);
            DAQmxClearTask(taskHandle_trigger_in);
            DAQmxStopTask(taskHandle_grab_in);
            DAQmxClearTask(taskHandle_grab_in);
            DAQmxStopTask(taskHandle_sampout);
            DAQmxClearTask(taskHandle_sampout);
            DAQmxStopTask(taskHandle_fout);
            DAQmxClearTask(taskHandle_fout);
            taskHandle_fout = 0;
            taskHandle_sampout = 0;
            taskHandle_trigger_in = 0;
            taskHandle_grab_in = 0;
            taskHandle_start_signal = 0;
        }

    }

    void NIDAQUtils::getCamtrig(unsigned int frameCount)
    {
        //Fills cam_trigger vector depending on whichever camera gets to it first. 
        if(cam_trigger[frameCount] == 0)
        {
            DAQmxErrChk(DAQmxReadCounterScalarU32(taskHandle_trigger_in, 10.0, &read_buffer, NULL));
            cam_trigger[frameCount] = read_buffer;
        }
        
    }

    /*void NIDAQUtils::newFrametrig(uInt32 read_buffer) {

        istrig = true;
        read_buffer = read_buffer;
    }*/

    NIDAQUtils::~NIDAQUtils() {

        Cleanup();
    }

    

}

