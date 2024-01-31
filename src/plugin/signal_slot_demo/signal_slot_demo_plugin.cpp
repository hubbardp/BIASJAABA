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
                     bool testConfigEnabled,
                     string trial_info,
                     std::shared_ptr<TestConfig> testConfig,
                     QWidget *parentPtr) : BiasPlugin(parentPtr)
    {
        imageLabelPtr_ = imageLabelPtr;
        nidaq_task_ = nullptr;
        testConfigEnabled_ = testConfigEnabled;
        testConfig_ = testConfig;
        trial_num_ = trial_info;

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
        int64_t pc_time;
        
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
            
            //---------------------------------------------------------------------------//
            
            acquireLock();
            currentImage_ = latestFrame.image;
            timeStamp_ = latestFrame.timeStamp;
            frameCount_ = latestFrame.frameCount;
            releaseLock();

            FrameData frameData;
            frameData.count = frameCount_;
            frameData.image = currentImage_;
            emit newFrameData(frameData);
            numMessageSent_++;
            
            // the updateMesageLabels() causes 
            //the plugin to crash sometimes when doing latency measurements
            //this call is made in the onNewFrameData anyways when the 
            //frame is emiited. Safe to comment??
            //updateMessageLabels();
            
            //if (testConfigEnabled_ && !testConfig_->imagegrab_prefix.empty()
            //    && testConfig_->plugin_prefix == "signal_slot")
            //{
            //    
            //    if (nidaq_task_ != nullptr) {

            //        /*if (cameraNumber_ == 0
            //            && frameCount_ <= unsigned long(testConfig_->numFrames)) {

            //            nidaq_task_->acquireLock();
            //            DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_trigger_in, 10.0, &read_buffer, NULL));
            //            nidaq_task_->releaseLock();

            //        }*/

            //        if (frameCount_ <= testConfig_->numFrames) {

            //            nidaq_task_->acquireLock();
            //            DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_grab_in, 10.0, &read_ondemand, NULL));
            //            nidaq_task_->releaseLock();

            //        }

            //    }

            //    if (!testConfig_->nidaq_prefix.empty()) {

            //        if (cameraNumber_ == 0)
            //            time_stamps3[frameCount_][0] = nidaq_task_->cam_trigger[frameCount_];
            //        else
            //            time_stamps3[frameCount_][0] = 0;

            //        time_stamps3[frameCount_][1] = read_ondemand;
            //    }

            //    if (!testConfig_->f2f_prefix.empty()) {

            //        pc_time = gettime_->getPCtime();

            //        if (frameCount_ <= testConfig_->numFrames)
            //            time_stamps2[frameCount_] = pc_time;
            //    }

            //    if (!testConfig_->queue_prefix.empty()) {

            //        if (frameCount_ <= testConfig_->numFrames)
            //            queue_size[frameCount_] = pluginImageQueuePtr_->size();

            //    }

            //    if (frameCount_ == testConfig_->numFrames-1
            //        && !testConfig_->f2f_prefix.empty())
            //    {
            //        
            //        std::string filename = testConfig_->dir_list[0] + "/"
            //            + testConfig_->f2f_prefix + "/" + testConfig_->cam_dir
            //            + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
            //            + testConfig_->plugin_prefix
            //            + "_" + testConfig_->f2f_prefix + "cam"
            //            + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";
            //        std::cout << filename << std::endl;
            //        gettime_->write_time_1d<int64_t>(filename, testConfig_->numFrames, time_stamps2);

            //    }

            //    if (frameCount_ == testConfig_->numFrames-1
            //        && !testConfig_->nidaq_prefix.empty())
            //    {

            //        std::string filename = testConfig_->dir_list[0] + "/"
            //            + testConfig_->nidaq_prefix + "/" + testConfig_->cam_dir
            //            + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
            //            + testConfig_->plugin_prefix
            //            + "_" + testConfig_->nidaq_prefix + "cam"
            //            + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";

            //        gettime_->write_time_2d<uInt32>(filename, testConfig_->numFrames, time_stamps3);

            //    }

            //    if (frameCount_ == testConfig_->numFrames-1
            //        && !testConfig_->queue_prefix.empty()) {

            //        string filename = testConfig_->dir_list[0] + "/"
            //            + testConfig_->queue_prefix + "/" + testConfig_->cam_dir
            //            + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
            //            + testConfig_->plugin_prefix
            //            + "_" + testConfig_->queue_prefix + "cam"
            //            + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";

            //        gettime_->write_time_1d<unsigned int>(filename, testConfig_->numFrames, queue_size);

            //    }

            //}
                      
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

    void SignalSlotDemoPlugin::allocate_testVec() {

        if (testConfigEnabled_) {

            if (!testConfig_->f2f_prefix.empty()) {

                time_stamps2.resize(testConfig_->numFrames);
            }

            if (!testConfig_->nidaq_prefix.empty()) {
               
                time_stamps3.resize(testConfig_->numFrames, std::vector<uInt32>(2, 0));
            }

            if (!testConfig_->queue_prefix.empty()) {

                queue_size.resize(testConfig_->numFrames);
            }

            
        }

    }

    // Private Slots
    // ------------------------------------------------------------------------

    void SignalSlotDemoPlugin::onNewFrameData(FrameData data)
    {
        
        numMessageReceived_++;
        updateMessageLabels();

    }

    void SignalSlotDemoPlugin::setupNIDAQ(std::shared_ptr <Lockable<NIDAQUtils>> nidaq_task,
                                          bool testConfigEnabled, string trial_info,
                                          std::shared_ptr<TestConfig> testConfig) {
        
        nidaq_task_ = nidaq_task;
        testConfig_ = testConfig;
        testConfigEnabled_ = testConfigEnabled;
        trial_num_ = trial_info;

        if (testConfigEnabled_)
            allocate_testVec();

    }


}
