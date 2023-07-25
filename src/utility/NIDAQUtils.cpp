#include "NIDAQUtils.hpp"
#include <stdio.h>

namespace bias {

    NIDAQUtils::NIDAQUtils()
    {
        //initialize();
        //configureNIDAQ();
    }

    void NIDAQUtils::initialize()
    {
        istrig = false;
        float64 half_period_cycle_fastclk = (float64)1.0 / (float64)(fast_counter_rate*2.0); // divide the period of
        float64 half_period_cycle_sampleclk = (float64) 1.0 / (float64)(sample_counter_rate*2.0); // cycle by 2 to get half cycle time
        uint64_t bufSize = numsamplesPerChan;

        printf("half cycle time fast clock %f\n", half_period_cycle_fastclk);
        printf("half cycle time fast clock %f\n", half_period_cycle_sampleclk);

        // initialize data buffers 
        for (int i = 0; i < bufSize; ++i) {

            dataf_high[i] = half_period_cycle_fastclk;
            dataf_low[i] = half_period_cycle_fastclk;
            datas_high[i] = half_period_cycle_sampleclk;
            datas_low[i] = half_period_cycle_sampleclk;
            data[i] = (uInt8)(i % 2);

        }
    }

    RtnStatus NIDAQUtils::setNIDAQConfigFromMap(QVariantMap& nidaqconfigMap)
    {
        RtnStatus rtnStatus;
        //device name
        if (!nidaqconfigMap.contains("device_name"))
        {
            QString errMsgText("device name counter not present");
            rtnStatus.success = false;
            rtnStatus.message = errMsgText;
            return rtnStatus;
        }
        else {
            if (!nidaqconfigMap["device_name"].canConvert<QString>()) {
                QString errMsgText("device name cannot convert to QString");
                rtnStatus.success = false;
                rtnStatus.message = errMsgText;
                return rtnStatus;
            }
            device_name =
                nidaqconfigMap["device_name"].toString().toLocal8Bit().data();
        }
        printf("device name %s\n ", device_name);

        //set fast channel output counter
        if (!nidaqconfigMap.contains("fast_outchannel_counter"))
        {
            QString errMsgText("fast channel counter not present");
            rtnStatus.success = false;
            rtnStatus.message = errMsgText;
            return rtnStatus;
        }
        else {
            if (!nidaqconfigMap["fast_outchannel_counter"].canConvert<QString>()) {
                QString errMsgText("fast channel counter cannot convert to QString");
                rtnStatus.success = false;
                rtnStatus.message = errMsgText;
                return rtnStatus;
            }
            fast_outchannel_counter = 
                nidaqconfigMap["fast_outchannel_counter"].toString().toLocal8Bit().data();
        }
        printf("Fast channel name %s\n ", fast_outchannel_counter);

        //set sample framerate output counter
        if (!nidaqconfigMap.contains("sample_outchannel_counter"))
        {
            QString errMsgText("sample outchannel channel counter not present");
            rtnStatus.success = false;
            rtnStatus.message = errMsgText;
            return rtnStatus;
        }
        else {
            if (!nidaqconfigMap["sample_outchannel_counter"].canConvert<QString>()) {
                QString errMsgText("sample channel counter cannot convert to QString");
                rtnStatus.success = false;
                rtnStatus.message = errMsgText;
                return rtnStatus;
            }
            sample_outchannel_counter =
                nidaqconfigMap["sample_outchannel_counter"].toString().toLocal8Bit().data();
        }
        printf("sample framerate output counter name %s\n ", sample_outchannel_counter);

        //set frame trigger input counter
        if (!nidaqconfigMap.contains("frametrig_inchannel_counter"))
        {
            QString errMsgText("frame trigger counter not present");
            rtnStatus.success = false;
            rtnStatus.message = errMsgText;
            return rtnStatus;
        }
        else {
            if (!nidaqconfigMap["frametrig_inchannel_counter"].canConvert<QString>()) {
                QString errMsgText("frame trigger counter cannot convert to QString");
                rtnStatus.success = false;
                rtnStatus.message = errMsgText;
                return rtnStatus;
            }
            frametrig_inchannel_counter =
                nidaqconfigMap["frametrig_inchannel_counter"].toString().toLocal8Bit().data();
        }
        printf("frame trigger input counter name %s\n ", frametrig_inchannel_counter);

        //set frame trig sample clock
        if (!nidaqconfigMap.contains("frametrig_sampleclk"))
        {
            QString errMsgText("frame trigger sample clock not present");
            rtnStatus.success = false;
            rtnStatus.message = errMsgText;
            return rtnStatus;
        }
        else {
            if (!nidaqconfigMap["frametrig_sampleclk"].canConvert<QString>()) {
                QString errMsgText("frame trigger counter cannot convert to QString");
                rtnStatus.success = false;
                rtnStatus.message = errMsgText;
                return rtnStatus;
            }
            frametrig_sampleclk =
                nidaqconfigMap["frametrig_sampleclk"].toString().toLocal8Bit().data();
        }
        printf("frame trigger sample clock  %s\n ", frametrig_sampleclk);

        //set frame trig data channel
        if (!nidaqconfigMap.contains("frametrig_datachannel"))
        {
            QString errMsgText("frame trigger data channel not present");
            rtnStatus.success = false;
            rtnStatus.message = errMsgText;
            return rtnStatus;
        }
        else {
            if (!nidaqconfigMap["frametrig_datachannel"].canConvert<QString>()) {
                QString errMsgText("frame trigger data channel cannot convert to QString");
                rtnStatus.success = false;
                rtnStatus.message = errMsgText;
                return rtnStatus;
            }
            frametrig_datachannel =
                nidaqconfigMap["frametrig_datachannel"].toString().toLocal8Bit().data();
        }
        printf("frame trigger sample data/src channel  %s\n ", frametrig_datachannel);

        // set framegrab input counter
        if (!nidaqconfigMap.contains("framegrab_inchannel_counter"))
        {
            QString errMsgText("frame grab counter not present");
            rtnStatus.success = false;
            rtnStatus.message = errMsgText;
            return rtnStatus;
        }
        else {
            if (!nidaqconfigMap["framegrab_inchannel_counter"].canConvert<QString>()) {
                QString errMsgText("frame grab counter cannot convert to QString");
                rtnStatus.success = false;
                rtnStatus.message = errMsgText;
                return rtnStatus;
            }
            framegrab_inchannel_counter =
                nidaqconfigMap["framegrab_inchannel_counter"].toString().toLocal8Bit().data();
        }
        printf("frame grab input counter name %s\n ", framegrab_inchannel_counter);

        //set didgital trigger source in
        if (!nidaqconfigMap.contains("digital_trigger_sourcein"))
        {
            QString errMsgText("digital trigger signal out not present");
            rtnStatus.success = false;
            rtnStatus.message = errMsgText;
            return rtnStatus;
        }
        else {
            if (!nidaqconfigMap["digital_trigger_sourcein"].canConvert<QString>()) {
                QString errMsgText("frame grab counter cannot convert to QString");
                rtnStatus.success = false;
                rtnStatus.message = errMsgText;
                return rtnStatus;
            }
            digital_trigger_sourcein =
                nidaqconfigMap["digital_trigger_sourcein"].toString().toLocal8Bit().data();
        }
        printf("digital trigger source in name %s\n ", digital_trigger_sourcein);

        //set didgital trigger signal out
        if (!nidaqconfigMap.contains("digital_trigger_signalout"))
        {
            QString errMsgText("digital trigger signal out not present");
            rtnStatus.success = false;
            rtnStatus.message = errMsgText;
            return rtnStatus;
        }
        else {

            if (!nidaqconfigMap["digital_trigger_signalout"].canConvert<QString>()) {
                QString errMsgText("frame grab counter cannot convert to QString");
                rtnStatus.success = false;
                rtnStatus.message = errMsgText;
                return rtnStatus;
            }
            digital_trigger_signalout =
                nidaqconfigMap["digital_trigger_signalout"].toString().toLocal8Bit().data();
        }
        printf("digital trigger signal out name %s\n ", digital_trigger_signalout);

        // fast counter clock rate
        if (!nidaqconfigMap.contains("fast_counter_rate"))
        {
            QString errMsgText("fast counter clock rate not present");
            rtnStatus.success = false;
            rtnStatus.message = errMsgText;
            return rtnStatus;
        }
        else {

            if (!nidaqconfigMap["fast_counter_rate"].canConvert<float64>()) {
                QString errMsgText("fast counter rate cannot convert to float64");
                rtnStatus.success = false;
                rtnStatus.message = errMsgText;
                return rtnStatus;
            }
            fast_counter_rate =
                nidaqconfigMap["fast_counter_rate"].toDouble();
        }
        printf("fast counter rate %f\n ", fast_counter_rate);
        
        //sample counter clock rate
        if (!nidaqconfigMap.contains("sample_counter_rate"))
        {
            QString errMsgText("sample counter clock rate not present");
            rtnStatus.success = false;
            rtnStatus.message = errMsgText;
            return rtnStatus;
        }
        else {

            if (!nidaqconfigMap["sample_counter_rate"].canConvert<float64>()) {
                QString errMsgText("sample counter rate cannot convert to float64");
                rtnStatus.success = false;
                rtnStatus.message = errMsgText;
                return rtnStatus;
            }
            sample_counter_rate =
                nidaqconfigMap["sample_counter_rate"].toDouble();
        }
        printf("sample counter rate %f\n ", sample_counter_rate);


        //number of sample per channel -size of buffer
        if (!nidaqconfigMap.contains("numSamplesPerChan"))
        {
            QString errMsgText("number of samples per channel not present");
            rtnStatus.success = false;
            rtnStatus.message = errMsgText;
            return rtnStatus;
        }
        else {

            if (!nidaqconfigMap["numSamplesPerChan"].canConvert<uInt64>()) {
                QString errMsgText("number of samples per channel cannot convert to float64");
                rtnStatus.success = false;
                rtnStatus.message = errMsgText;
                return rtnStatus;
            }
            numsamplesPerChan =
                nidaqconfigMap["numSamplesPerChan"].toULongLong();
        }
        printf("number of samples per channel %d\n ", numsamplesPerChan);

        initialize();
        configureNIDAQ();

        return rtnStatus;
    }


    void NIDAQUtils::configureNIDAQ() {

        // NIDAQ Configuration
        
        //fast source channel
        DAQmxErrChk(DAQmxCreateTask("", &taskHandle_fout));
        DAQmxErrChk(DAQmxCreateCOPulseChanTime(taskHandle_fout, (device_name + fast_outchannel_counter).c_str(),
             "", DAQmx_Val_Seconds, DAQmx_Val_Low, 0.0, 0.00001, 0.00001));
        DAQmxErrChk(DAQmxCfgImplicitTiming(taskHandle_fout, DAQmx_Val_ContSamps, numsamplesPerChan));
        DAQmxErrChk(DAQmxWriteCtrTime(taskHandle_fout, numsamplesPerChan, 0, 1.0, DAQmx_Val_GroupByChannel, dataf_high, dataf_low, NULL, NULL));

        // sample clock - framerate
        DAQmxErrChk(DAQmxCreateTask("", &taskHandle_sampout));
        DAQmxErrChk(DAQmxCreateCOPulseChanTime(taskHandle_sampout, (device_name + sample_outchannel_counter).c_str(), "", DAQmx_Val_Seconds, DAQmx_Val_Low, 0.0, 0.00125, 0.00125));
        DAQmxErrChk(DAQmxCfgImplicitTiming(taskHandle_sampout, DAQmx_Val_ContSamps, numsamplesPerChan));
        DAQmxErrChk(DAQmxWriteCtrTime(taskHandle_sampout, numsamplesPerChan, 0, 1.0, DAQmx_Val_GroupByChannel, datas_high, datas_low, NULL, NULL));

        //Measure frame trigger Channel
        DAQmxErrChk(DAQmxCreateTask("", &taskHandle_trigger_in));
        DAQmxErrChk(DAQmxCreateCICountEdgesChan(taskHandle_trigger_in, (device_name + frametrig_inchannel_counter).c_str(), "", DAQmx_Val_Rising, 0, DAQmx_Val_CountUp));
        DAQmxErrChk(DAQmxCfgSampClkTiming(taskHandle_trigger_in, ("/" + device_name + frametrig_sampleclk).c_str() , 1000.0, DAQmx_Val_Rising, DAQmx_Val_ContSamps, numsamplesPerChan));
        DAQmxErrChk(DAQmxSetCICountEdgesTerm(taskHandle_trigger_in, (device_name + frametrig_inchannel_counter).c_str(),
                    ("/" + device_name + frametrig_datachannel).c_str()));

        // Measure frame grab Channel
        DAQmxErrChk(DAQmxCreateTask("", &taskHandle_grab_in));
        DAQmxErrChk(DAQmxCreateCICountEdgesChan(taskHandle_grab_in, (device_name + framegrab_inchannel_counter).c_str(), "", DAQmx_Val_Rising, 0, DAQmx_Val_CountUp));
        DAQmxErrChk(DAQmxSetCICountEdgesTerm(taskHandle_grab_in, (device_name + framegrab_inchannel_counter).c_str(), 
                    ("/" + device_name + frametrig_datachannel).c_str()));

        // start trigger signal
        DAQmxErrChk(DAQmxCreateTask("", &taskHandle_start_signal));
        DAQmxErrChk(DAQmxCreateDOChan(taskHandle_start_signal, ("/" + device_name + digital_trigger_signalout).c_str(), "", DAQmx_Val_ChanPerLine));
        DAQmxErrChk(DAQmxCfgSampClkTiming(taskHandle_start_signal, "", 1000.0, DAQmx_Val_Rising, DAQmx_Val_ContSamps, numsamplesPerChan));
        DAQmxErrChk(DAQmxWriteDigitalLines(taskHandle_start_signal, numsamplesPerChan, 0, 10.0, DAQmx_Val_GroupByChannel, data, NULL, NULL));

        DAQmxErrChk(DAQmxCfgDigEdgeStartTrig(taskHandle_fout, ("/" + device_name + digital_trigger_sourcein).c_str(), DAQmx_Val_Rising));
        DAQmxErrChk(DAQmxCfgDigEdgeStartTrig(taskHandle_sampout, ("/" + device_name + digital_trigger_sourcein).c_str(), DAQmx_Val_Rising));

    }

    void NIDAQUtils::startTasks()
    {
        acquireLock();
        DAQmxErrChk(DAQmxStartTask(taskHandle_grab_in));
        DAQmxErrChk(DAQmxStartTask(taskHandle_trigger_in));
        DAQmxErrChk(DAQmxStartTask(taskHandle_fout));
        DAQmxErrChk(DAQmxStartTask(taskHandle_sampout));
        start_tasks = true;
        releaseLock();
        //printf("***** tasks started ***** \n ");
    }

    void NIDAQUtils::stopTasks()
    {
        acquireLock();
        DAQmxErrChk(DAQmxStopTask(taskHandle_trigger_in));
        DAQmxErrChk(DAQmxStopTask(taskHandle_grab_in));
        DAQmxErrChk(DAQmxStopTask(taskHandle_sampout));
        DAQmxErrChk(DAQmxStopTask(taskHandle_fout));    
        start_tasks = false;
        releaseLock();
        //printf("***** tasks stopped *****\n ");
    }

    void NIDAQUtils::start_trigger_signal() {

        //printf("***** Start trigger Entred *****");
        acquireLock();
        DAQmxErrChk(DAQmxStartTask(taskHandle_start_signal));
        istrig = true;
        releaseLock(); 
        //printf("***** Trig Started *****\n  ", istrig); 

    }

    void NIDAQUtils::stop_trigger_signal() {

        acquireLock();
        DAQmxErrChk(DAQmxStopTask(taskHandle_start_signal));
        istrig = false;
        std::fill(cam_trigger.begin(), cam_trigger.end(), 0);
        releaseLock();
       //printf("***** Trig stopped *****\n ", istrig);

    }

    void NIDAQUtils::Cleanup() {

        if (taskHandle_fout != 0)
        {
            /*********************************************/
            // DAQmx Stop Code
            /*********************************************/
            DAQmxClearTask(taskHandle_trigger_in);
            DAQmxClearTask(taskHandle_grab_in);
            DAQmxClearTask(taskHandle_sampout);
            DAQmxClearTask(taskHandle_fout);
            DAQmxClearTask(taskHandle_start_signal);
            taskHandle_fout = 0;
            taskHandle_sampout = 0;
            taskHandle_trigger_in = 0;
            taskHandle_grab_in = 0;
            taskHandle_start_signal = 0;
            istrig = false;
            start_tasks = false;
        }

    }

    void NIDAQUtils::getCamtrig(unsigned int frameCount)
    {
        //Fills cam_trigger vector depending on whichever camera gets to it first. 
        float64 timeout_seconds = 10.0;
        acquireLock();
        if(cam_trigger[frameCount] == 0)
        {
            DAQmxErrChk(DAQmxReadCounterScalarU32(taskHandle_trigger_in, timeout_seconds, &read_buffer, NULL));
            cam_trigger[frameCount] = read_buffer;
        }
        releaseLock();
        
    }

    void NIDAQUtils::getNidaqTimeNow(uInt32& read_ondemand)
    {
        float64 timeout_seconds = 10.0;
        acquireLock();
        DAQmxErrChk(DAQmxReadCounterScalarU32(taskHandle_grab_in, timeout_seconds, &read_ondemand, NULL));
        releaseLock();
    }

    /*void NIDAQUtils::newFrametrig(uInt32 read_buffer) {

        istrig = true;
        read_buffer = read_buffer;
    }*/

    NIDAQUtils::~NIDAQUtils() {

        Cleanup();
    }  

}

