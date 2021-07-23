#include "signal_slot_demo_plugin.hpp"
#include "image_label.hpp"
#include "camera_window.hpp"
#include <QtDebug>
#include <QTimer>
#include <QMessageBox>
#include <QThread>
//#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <iostream>

namespace bias
{
    // Public static variables
    // ------------------------------------------------------------------------
    const QString SignalSlotDemoPlugin::PLUGIN_NAME = QString("signalSlotDemo");
    const QString SignalSlotDemoPlugin::PLUGIN_DISPLAY_NAME = QString("Signal Slot Demo");

    // Public Methods
    // ------------------------------------------------------------------------
    SignalSlotDemoPlugin::SignalSlotDemoPlugin(ImageLabel *imageLabelPtr, 
                     std::shared_ptr<Lockable<GetTime>> gettime, QWidget *parentPtr) : BiasPlugin(parentPtr)
    {
        imageLabelPtr_ = imageLabelPtr;
        gettime_ = gettime;
        nidaq_task_ = nullptr;
        cam_delay1.resize(500000);
        cam_delay2.resize(500000);
        setupUi(this);
        connectWidgets();
        initialize();
    }

    void SignalSlotDemoPlugin::finalSetup()
    {
        QPointer<CameraWindow> partnerCameraWindowPtr = getPartnerCameraWindowPtr();
        if (partnerCameraWindowPtr)
        {
            QPointer<BiasPlugin> partnerPluginPtr = partnerCameraWindowPtr -> getPluginByName("signalSlotDemo");
            qRegisterMetaType<FrameData>("FrameData");
            connect(partnerPluginPtr, SIGNAL(newFrameData(FrameData)), this, SLOT(onNewFrameData(FrameData)));
        }
        updateMessageLabels();
    }

    void SignalSlotDemoPlugin::reset()
    {
    }


    void SignalSlotDemoPlugin::stop()
    {

    }


    //void SignalSlotDemoPlugin::processFrames(QList<StampedImage> frameList)
    void SignalSlotDemoPlugin::processFrames()
    {
        // -------------------------------------------------------------------
        // NOTE: called in separate thread. Use lock to access data shared
        // with other class methods.
        // -------------------------------------------------------------------

        uInt32 read_buffer, read_ondemand;

        pluginImageQueuePtr_ -> acquireLock();
        pluginImageQueuePtr_ -> waitIfEmpty();


        if (pluginImageQueuePtr_ -> empty() )
        {

            pluginImageQueuePtr_ -> releaseLock();
            return;

        }

        if(!pluginImageQueuePtr_ -> empty())
        {

            StampedImage latestFrame = pluginImageQueuePtr_ ->front();//frameList.back();
            //frameList.clear();

            //-------------------DEVEL----------------------------------------------------//

            // get camera times wrt to stamped image times
            int64_t pc_ts1;
            pc_ts1 = gettime_->getPCtime();
            cam_delay2[latestFrame.frameCount] = pc_ts1;


            if(latestFrame.frameCount == 499999){
                std::string filecam = "signal_slot_f2f" + std::to_string(cameraNumber_) + ".csv"; 
                gettime_->write_time_1d<int64_t>(filecam , 500000, cam_delay2);
            }
            //---------------------------------------------------------------------------//
            
            /*acquireLock();
            currentImage_ = latestFrame.image;
            timeStamp_ = latestFrame.timeStamp;
            frameCount_ = latestFrame.frameCount;
            releaseLock();

            FrameData frameData;
            frameData.count = frameCount_;
            frameData.image = currentImage_;
            emit newFrameData(frameData);
            numMessageSent_++;*/
            
            // the updateMesageLabels() causes 
            //the plugin to crash sometimes when doing latency measurements
            //this call is made in the onNewFrameData anyways when the 
            //frame is emiited. Safe to comment??

            //updateMessageLabels();
            
            if (nidaq_task_ != nullptr) {
                nidaq_task_->acquireLock();
                DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_grab_in, 10.0, &read_ondemand, NULL));
                cam_delay1[latestFrame.frameCount] = read_ondemand;
                nidaq_task_->releaseLock();
                if (latestFrame.frameCount == 499999) {
                    std::string filename = "signal_slot_time_cam" + std::to_string(cameraNumber_) + ".csv";
                    gettime_->write_time_1d<uInt32>(filename, 500000, cam_delay1);
                }
            }
                      
            pluginImageQueuePtr_->pop();

        }

        pluginImageQueuePtr_->releaseLock();

    }


    cv::Mat SignalSlotDemoPlugin::getCurrentImage()
    {
        acquireLock();
        cv::Mat currentImageCopy = currentImage_.clone();
        releaseLock();
        return currentImageCopy;
    }


    QString SignalSlotDemoPlugin::getName()
    {
        return PLUGIN_NAME;
    }


    QString SignalSlotDemoPlugin::getDisplayName()
    {
        return PLUGIN_DISPLAY_NAME;
    }


    /*TimeStamp SignalSlotDemoPlugin::getPCtime()
    {

        unsigned long long int secs;
        unsigned int usec;
        time_t curr_time;
        timeval tv;

        //get computer local time since midnight
        curr_time = time(NULL);
        tm *tm_local = localtime(&curr_time);
        gettimeofday(&tv, NULL);
        secs = (tm_local->tm_hour*3600) + tm_local->tm_min*60 + tm_local->tm_sec;
        usec = (unsigned int)tv.tv_usec;
        TimeStamp ts = {secs,usec};

        return ts;

    }


    TimeStamp SignalSlotDemoPlugin::cameraOffsetTime(std::shared_ptr<Lockable<Camera>> cameraPtr)
    {

        TimeStamp cam_ofs = {0,0};
        TimeStamp pc_ts, cam_ts;
        double pc_s, cam_s, offset_s;
        std::vector<double> timeofs;

        for(int ind=0;ind < 10;ind++)
        {

            //get computer local time since midnight
            pc_ts = getPCtime();
            pc_s = (double)((pc_ts.seconds*1e6) + (pc_ts.microSeconds))*1e-6;

            //calculate camera time
            if(cameraPtr!=nullptr){
                cam_ts = cameraPtr->getDeviceTimeStamp();
                cam_s = (double)((cam_ts.seconds*1e6) + (cam_ts.microSeconds))*1e-6;
            }else{

                std::cout << " No camera found " << std::endl;
            }

            timeofs.push_back(pc_s-cam_s);
            //printf("%0.06f \n" ,pc_s-cam_s);
            //printf("%0.06f  %0.06f pc_s-cam_us\n ", pc_s ,cam_s);

        }

        //write_time("offset.csv",20,timeofs);

        //calculate mean
        offset_s = accumulate(timeofs.begin(),timeofs.end(),0.0)/timeofs.size();
        cam_ofs.seconds = int(offset_s);
        cam_ofs.microSeconds = (offset_s - cam_ofs.seconds)*1e6;
        ofs_isSet = false;


        //calculate std dev
        double std_sum=0;
        for(int k=0;k < timeofs.size() ;k++)
        {
           std_sum += (timeofs[k] - offset_s) * (timeofs[k] - offset_s);
        }

        std_sum = std_sum/timeofs.size();
        std_sum = sqrt(std_sum);

        //printf("%0.06f average offset \n" ,offset_s);
        printf("%0.06f std deviation \n ",std_sum);
        //printf("%ld seconds %d microseconds", cam_ofs.seconds, cam_ofs.microSeconds);

        return cam_ofs;

    }*/

    // Protected Methods
    // ------------------------------------------------------------------------

    void SignalSlotDemoPlugin::connectWidgets()
    {

    }

    unsigned int SignalSlotDemoPlugin::getPartnerCameraNumber()
    {
        // Returns camera number of partner camera. For this example
        // we just use camera 0 and 1. In another setting you might do
        // this by GUID or something else.
        if (cameraNumber_ == 0)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

    QPointer<CameraWindow> SignalSlotDemoPlugin::getPartnerCameraWindowPtr()
    {
        QPointer<CameraWindow> partnerCameraWindowPtr = nullptr;
        if ((cameraWindowPtrList_ -> size()) > 1)
        {
            for (auto cameraWindowPtr : *cameraWindowPtrList_)
            {
                if ((cameraWindowPtr -> getCameraNumber()) == partnerCameraNumber_)
                {
                    partnerCameraWindowPtr = cameraWindowPtr;
                }
            }
        }
        return partnerCameraWindowPtr;
    }


    void SignalSlotDemoPlugin::initialize()
    {

        QPointer<CameraWindow> cameraWindowPtr = getCameraWindow();
        cameraNumber_ = cameraWindowPtr -> getCameraNumber();
        cameraGuidString_ = cameraWindowPtr ->  getCameraGuidString();
        cameraWindowPtrList_ = cameraWindowPtr -> getCameraWindowPtrList();
        cameraPtr_ = cameraWindowPtr->getCameraPtr();

        QString labelStr = QString("Camera #: %0,     GUID: %2").arg(cameraNumber_).arg(cameraGuidString_);
        cameraNumberLabelPtr -> setText(labelStr);

        partnerCameraNumber_ = getPartnerCameraNumber();
        connectedToPartner_ = false;

        numMessageSent_ = 0;
        numMessageReceived_ = 0;

    }


    void SignalSlotDemoPlugin::updateMessageLabels()
    {
        
        QString sentLabelText = QString("messages sent: %1").arg(numMessageSent_);
        messageSentLabelPtr -> setText(sentLabelText);

        QString recvLabelText = QString("messages recv: %1").arg(numMessageReceived_);
        messageReceivedLabelPtr -> setText(recvLabelText);
        
    }


    // Private Slots
    // ------------------------------------------------------------------------

    void SignalSlotDemoPlugin::onNewFrameData(FrameData data)
    {
        
        numMessageReceived_++;
        updateMessageLabels();

    }

    void SignalSlotDemoPlugin::setupNIDAQ(std::shared_ptr <Lockable<NIDAQUtils>> nidaq_task) {

        nidaq_task_ = nidaq_task;
    }


}
