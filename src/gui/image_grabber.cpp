#include "image_grabber.hpp"
#include "exception.hpp"
#include "camera.hpp"
#include "stamped_image.hpp"
#include "affinity.hpp"
#include <iostream>
#include <QTime>
#include <QThread>
#include <QFileInfo>
#include <opencv2/core/core.hpp>


// TEMPOERARY
// ----------------------------------------
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "win_time.hpp"
#include "camera_device.hpp"
// ----------------------------------------

#define DEBUG 0 

namespace bias {

    unsigned int ImageGrabber::DEFAULT_NUM_STARTUP_SKIP = 2;
    unsigned int ImageGrabber::MIN_STARTUP_SKIP = 2;
    unsigned int ImageGrabber::MAX_ERROR_COUNT = 500;

    ImageGrabber::ImageGrabber(QObject *parent) : QObject(parent) 
    {
        
        initialize(0,NULL,NULL,NULL,false,"",NULL,NULL,NULL);
    }

    ImageGrabber::ImageGrabber (
            unsigned int cameraNumber,
            std::shared_ptr<Lockable<Camera>> cameraPtr,
            std::shared_ptr<LockableQueue<StampedImage>> newImageQueuePtr,
            QPointer<QThreadPool> threadPoolPtr,
            bool testConfigEnabled,
            string trial_info,
            std::shared_ptr<TestConfig> testConfig,
            std::shared_ptr<Lockable<GetTime>> gettime,
            std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task,
            QObject *parent
            ) : QObject(parent)
    {
        initialize(cameraNumber, cameraPtr, newImageQueuePtr, threadPoolPtr,
                   testConfigEnabled, trial_info, testConfig, gettime, nidaq_task);
    }

    void ImageGrabber::initialize(
        unsigned int cameraNumber,
        std::shared_ptr<Lockable<Camera>> cameraPtr,
        std::shared_ptr<LockableQueue<StampedImage>> newImageQueuePtr,
        QPointer<QThreadPool> threadPoolPtr,
        bool testConfigEnabled,
        string trial_info,
        std::shared_ptr<TestConfig> testConfig,
        std::shared_ptr<Lockable<GetTime>> gettime,
        std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task
        ) 
    {
        capturing_ = false;
        stopped_ = true;
        cameraPtr_ = cameraPtr;
        threadPoolPtr_ = threadPoolPtr;
        newImageQueuePtr_ = newImageQueuePtr;
        numStartUpSkip_ = DEFAULT_NUM_STARTUP_SKIP;
        cameraNumber_ = cameraNumber;
        testConfigEnabled_ = testConfigEnabled;
        testConfig_ = testConfig;
        trial_num = trial_info;
        fstfrmtStampRef_ = 0.0;
        process_frame_time = 1;
        
        QPointer<CameraWindow> cameraWindowPtr = getCameraWindow();
        cameraWindowPtrList_ = cameraWindowPtr->getCameraWindowPtrList();
        partnerCameraWindowPtr = getPartnerCameraWindowPtr();
      
        if ((cameraPtr_ != NULL) && (newImageQueuePtr_ != NULL))
        {
            ready_ = true;
        }
        else
        {
            ready_ = false;
        }
        errorCountEnabled_ = true;

        nidaq_task_ = nidaq_task;
        // needs to be allocated here outside the testConfig.Intend to record nidaq
        // camera trigger timestamps for other threads even if imagegrab is 
        // turned off in testConfig suite
        if (nidaq_task_ != nullptr) {
            nidaq_task_->cam_trigger.resize(testConfig_->numFrames);
        }

        gettime_ = gettime;
#if DEBUG
        if (testConfigEnabled_ && !testConfig_->imagegrab_prefix.empty()) {

            if (!testConfig_->f2f_prefix.empty()) {

                ts_pc.resize(testConfig_->numFrames);
            }

            if (!testConfig_->nidaq_prefix.empty()) {

                ts_nidaq.resize(testConfig_->numFrames, std::vector<uInt32>(2, 0));
                ts_nidaqThres.resize(testConfig_->numFrames);
            }
            
            if (!testConfig_->queue_prefix.empty()) {

                queue_size.resize(testConfig_->numFrames);
            }

            if (process_frame_time)
            {
                ts_process.resize(testConfig_->numFrames, 0);
            }
        }
#endif
    }

    void ImageGrabber::stop()
    {
        stopped_ = true;   
    }


    void ImageGrabber::enableErrorCount()
    {
        errorCountEnabled_ = true;
    }
   
    void ImageGrabber::disableErrorCount()
    {
        errorCountEnabled_ = false;
    }

    void ImageGrabber::run()
    { 

        bool isFirst = true;
        bool istriggered = false;
        bool done = false;
        bool error = false;
        bool errorEmitted = false;
        unsigned int errorId = 0;
        unsigned int errorCount = 0;
        unsigned long frameCount = 0;
        unsigned long startUpCount = 0;
        double dtEstimate = 0.0;
        TriggerType trig;

        TimeStamp timeStamp;
        TimeStamp timeStampInit; 

        double timeStampDbl = 0.0;
        double timeStampDblLast = 0.0;

        int64_t pc_time, start_process, end_process;

        QString errorMsg("no message");

        if (!ready_) 
        { 
            return; 
        }

        // Set thread priority to "time critical" and assign cpu affinity
        QThread *thisThread = QThread::currentThread();
        thisThread -> setPriority(QThread::TimeCriticalPriority);
        ThreadAffinityService::assignThreadAffinity(true,cameraNumber_);

        trig = cameraPtr_->getTriggerType();
        
        // Start image capture
        cameraPtr_ -> acquireLock();
        try
        {
            cameraPtr_ -> startCapture();
            if (nidaq_task_ != nullptr) {
                
                cameraPtr_->setupNIDAQ(nidaq_task_, testConfigEnabled_,
                                       trial_num, testConfig_,
                                       gettime_, cameraNumber_);
               
            }
            else {
                printf("nidaq not set");      
            }
        }
        catch (RuntimeError &runtimeError)
        {
            error = true;
            errorId = runtimeError.id();
            errorMsg = QString::fromStdString(runtimeError.what());
        }
        cameraPtr_ -> releaseLock();

        if (error)
        {
            emit startCaptureError(errorId, errorMsg);
            errorEmitted = true;
            return;

        } 

        acquireLock();
        stopped_ = false;
        releaseLock();

        //// TEMPORARY - for mouse grab detector testing
        //// ------------------------------------------------------------------------------

        //// Check for existence of movie file
        //QString grabTestMovieFileName("bias_test.avi");
        //cv::VideoCapture fileCapture;
        //unsigned int numFrames = 0;
        //int fourcc = 0;
        //bool haveGrabTestMovie = false;

        //if (QFileInfo(grabTestMovieFileName).exists())
        //{
        //    fileCapture.open(grabTestMovieFileName.toStdString());
        //    if ( fileCapture.isOpened() )
        //    {
        //        numFrames = (unsigned int)(fileCapture.get(CV_CAP_PROP_FRAME_COUNT));
        //        fourcc = int(fileCapture.get(CV_CAP_PROP_FOURCC));
        //        haveGrabTestMovie = true;
        //    }
        //}
        //// -------------------------------------------------------------------------------
                
        // Grab images from camera until the done signal is given
        while (!done)
        {

            acquireLock();
            done = stopped_;
            releaseLock();
          
            if (!istriggered && nidaq_task_ != nullptr && cameraNumber_ == 0) {

                nidaq_task_->startTasks();
                istriggered = true;
            }

            start_process = gettime_->getPCtime();

            // Grab an image
            cameraPtr_->acquireLock();
            try
            {
                stampImg.image = cameraPtr_->grabImage();
                stampImg.isSpike = false;
                timeStamp = cameraPtr_->getImageTimeStamp();
                error = false;
            }
            catch (RuntimeError &runtimeError)
            {
                std::cout << "Frame grab error: id = ";
                std::cout << runtimeError.id() << ", what = ";
                std::cout << runtimeError.what() << std::endl;
                error = true;
            }
            cameraPtr_->releaseLock();

            // grabImage is nonblocking - returned frame is empty is a new frame is not available.
            if (stampImg.image.empty())
            {
                QThread::yieldCurrentThread();
                continue;
            }

            // Push image into new image queue
            if (!error)
            {
                errorCount = 0;                  // Reset error count 
                timeStampDblLast = timeStampDbl; // Save last timestamp

                // Set initial time stamp for fps estimate
                if ((startUpCount == 0) && (numStartUpSkip_ > 0))
                {
                    timeStampInit = timeStamp;
                }
                timeStampDbl = convertTimeStampToDouble(timeStamp, timeStampInit);

                // Skip some number of frames on startup - recommened by Point Grey. 
                // During this time compute running avg to get estimate of frame interval
                if (startUpCount < numStartUpSkip_)
                {
                    double dt = timeStampDbl - timeStampDblLast;
                    if (startUpCount == MIN_STARTUP_SKIP)
                    {
                        dtEstimate = dt;
                    }
                    else if (startUpCount > MIN_STARTUP_SKIP)
                    {
                        double c0 = double(startUpCount - 1) / double(startUpCount);
                        double c1 = double(1.0) / double(startUpCount);
                        dtEstimate = c0 * dtEstimate + c1 * dt;
                    }
                    startUpCount++;

                    if (cameraNumber_ == 0 && nidaq_task_ != nullptr) {

                        nidaq_task_->acquireLock();
                        DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_trigger_in, 10.0, &read_buffer_, NULL));
                        nidaq_task_->releaseLock();
                    }

                    continue;
                }
            

                //std::cout << "dt grabber: " << timeStampDbl - timeStampDblLast << std::endl;

                // Reset initial time stamp for image acquisition
                if ((isFirst) && (startUpCount >= numStartUpSkip_))
                {
                    timeStampInit = timeStamp;
                    timeStampDblLast = 0.0;
                    isFirst = false;
                    timeStampDbl = convertTimeStampToDouble(timeStamp, timeStampInit);
                    emit startTimer();
                }
                //

                //// TEMPORARY - for mouse grab detector testing
                //// --------------------------------------------------------------------- 
                //cv::Mat fileMat;
                //StampedImage fileImg;
                //if (haveGrabTestMovie)
                //{
                //    fileCapture >> fileMat; 
                //    if (fileMat.empty())
                //    {
                //        fileCapture.set(CV_CAP_PROP_POS_FRAMES,0);
                //        continue;
                //    }

                //    cv::Mat  fileMatMono = cv::Mat(fileMat.size(), CV_8UC1);
                //    cvtColor(fileMat, fileMatMono, CV_RGB2GRAY);
                //    
                //    cv::Mat camSizeImage = cv::Mat(stampImg.image.size(), CV_8UC1);
                //    int padx = camSizeImage.rows - fileMatMono.rows;
                //    int pady = camSizeImage.cols - fileMatMono.cols;

                //    cv::Scalar padColor = cv::Scalar(0);
                //    cv::copyMakeBorder(fileMatMono, camSizeImage, 0, pady, 0, padx, cv::BORDER_CONSTANT, cv::Scalar(0));
                //    stampImg.image = camSizeImage;
                //}

                // ---------------------------------------------------------------------
                // Test Configuration
                //------------------------------------------------------------------------
                if (testConfigEnabled_ && frameCount < testConfig_->numFrames) {
                    
                    if (!testConfig_->imagegrab_prefix.empty()
                        && nidaq_task_ != nullptr) {

                        if (cameraNumber_ == 0)
                        {
                            nidaq_task_->acquireLock();
                            DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_grab_in, 10.0, &read_ondemand_, NULL));
                            nidaq_task_->getCamtrig(frameCount);
                            nidaq_task_->releaseLock();
                        }

                        if (cameraNumber_ == 1)
                        {
                            nidaq_task_->acquireLock();
                            DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_grab_in, 10.0, &read_ondemand_, NULL));
                            nidaq_task_->getCamtrig(frameCount);
                            nidaq_task_->releaseLock();
                        }

                        if (!testConfig_->nidaq_prefix.empty())
                        {

                            nidaq_task_->acquireLock();
                            ts_nidaq[frameCount][0] = nidaq_task_->cam_trigger[frameCount];
                            nidaq_task_->releaseLock();
                            ts_nidaq[frameCount][1] = read_ondemand_;
#if DEBUG
                            spikeDetected(frameCount);
#endif

                        }

                    }
                }
                //-------------------------------------------------------------------------
                if (nidaq_task_ != nullptr && frameCount == 0) {

                    acquireLock();
                    //pc_time = gettime_->getPCtime();
                    //fstfrmtStampRef_ = static_cast<uint64_t>(pc_time);
                    fstfrmtStampRef_ = nidaq_task_->cam_trigger[frameCount];
                    releaseLock();
                    
                }

                // Set image data timestamp, framecount and frame interval estimate
                stampImg.timeStamp = timeStampDbl;
                stampImg.timeStampInit = timeStampInit;
                stampImg.timeStampVal = timeStamp;
                stampImg.frameCount = frameCount;
                stampImg.dtEstimate = dtEstimate;
                stampImg.fstfrmtStampRef = fstfrmtStampRef_;

                newImageQueuePtr_->acquireLock();
                newImageQueuePtr_->push(stampImg);
                newImageQueuePtr_->signalNotEmpty();
                newImageQueuePtr_->releaseLock();

                frameCount++;
                end_process = gettime_->getPCtime();

                ///---------------------------------------------------------------
#if DEBUG
                if (testConfigEnabled_ && ((frameCount-1) < testConfig_->numFrames)) {

                    if (!testConfig_->imagegrab_prefix.empty()){ 

                        if (!testConfig_->f2f_prefix.empty()) {

                            pc_time = gettime_->getPCtime();
                            if (frameCount <= unsigned long(testConfig_->numFrames))
                                ts_pc[frameCount - 1] = pc_time;
                        }

                        if (!testConfig_->queue_prefix.empty()) {

                            if (frameCount <= unsigned long(testConfig_->numFrames))
                                queue_size[frameCount - 1] = newImageQueuePtr_->size();

                        }

                        if (process_frame_time)
                        {
                            if (frameCount <= unsigned long(testConfig_->numFrames))
                                ts_process[frameCount - 1] = end_process - start_process;
                        }

                        if (frameCount == testConfig_->numFrames
                            && !testConfig_->f2f_prefix.empty())
                        {

                            std::string filename = testConfig_->dir_list[0] + "/"
                                + testConfig_->f2f_prefix + "/" + testConfig_->cam_dir
                                + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                                + testConfig_->imagegrab_prefix
                                + "_" + testConfig_->f2f_prefix + "cam"
                                + std::to_string(cameraNumber_) + "_" + trial_num + ".csv";
                            
                            gettime_->write_time_1d<int64_t>(filename, testConfig_->numFrames, ts_pc);
                            
                        }

                        if (frameCount == testConfig_->numFrames
                            && !testConfig_->nidaq_prefix.empty())
                        {

                            std::string filename = testConfig_->dir_list[0] + "/"
                                + testConfig_->nidaq_prefix + "/" + testConfig_->cam_dir
                                + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                                + testConfig_->imagegrab_prefix
                                + "_" + testConfig_->nidaq_prefix + "cam"
                                + std::to_string(cameraNumber_) + "_" + trial_num + ".csv";

                            std::string filename1 = testConfig_->dir_list[0] + "/"
                                + testConfig_->nidaq_prefix + "/" + testConfig_->cam_dir
                                + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                                + testConfig_->imagegrab_prefix
                                + "_" + testConfig_->nidaq_prefix + "_thres" + "cam"
                                + std::to_string(cameraNumber_) + "_" + trial_num + ".csv";

                            gettime_->write_time_2d<uInt32>(filename, testConfig_->numFrames, ts_nidaq);
                            gettime_->write_time_1d<float>(filename1, testConfig_->numFrames, ts_nidaqThres);
                        }

                        if (frameCount == testConfig_->numFrames
                            && !testConfig_->queue_prefix.empty()) {

                            string filename = testConfig_->dir_list[0] + "/"
                                + testConfig_->queue_prefix + "/" + testConfig_->cam_dir
                                + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                                + testConfig_->imagegrab_prefix
                                + "_" + testConfig_->queue_prefix + "cam"
                                + std::to_string(cameraNumber_) + "_" + trial_num + ".csv";

                            gettime_->write_time_1d<unsigned int>(filename, testConfig_->numFrames, queue_size);

                        }

                        if (frameCount == testConfig_->numFrames
                            && process_frame_time) {

                            string filename = testConfig_->dir_list[0] + "/"
                                + testConfig_->nidaq_prefix + "/" + testConfig_->cam_dir
                                + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                                + testConfig_->imagegrab_prefix
                                + "_" + "process_time" + "cam"
                                + std::to_string(cameraNumber_) + "_" + trial_num + ".csv";
                            std::cout << filename << std::endl;
                            gettime_->write_time_1d<int64_t>(filename, testConfig_->numFrames, ts_process);

                        }
                    }
                }
#endif
            }else {

                if (errorCountEnabled_ ) 
                {
                    errorCount++;
                    if (errorCount > MAX_ERROR_COUNT)
                    {
                        errorId = ERROR_CAPTURE_MAX_ERROR_COUNT;
                        errorMsg = QString("Maximum allowed capture error count reached");
                        if (!errorEmitted) 
                        {
                            emit captureError(errorId, errorMsg);
                            errorEmitted = true;
                        }
                    }
                }
            }

        } // while (!done) 

        // Stop image capture
        error = false;
        cameraPtr_ -> acquireLock();
        try
        {
            cameraPtr_ -> stopCapture();
        }
        catch (RuntimeError &runtimeError)
        {
            error = true;
            errorId = runtimeError.id();
            errorMsg = QString::fromStdString(runtimeError.what());
        }
        cameraPtr_ -> releaseLock();

        if ((error) && (!errorEmitted))
        { 
            emit stopCaptureError(errorId, errorMsg);
        }

    }


    double ImageGrabber::convertTimeStampToDouble(TimeStamp curr, TimeStamp init)
    {
        double timeStampDbl = 0; 
        timeStampDbl  = double(curr.seconds);
        timeStampDbl -= double(init.seconds);
        timeStampDbl += (1.0e-6)*double(curr.microSeconds);
        timeStampDbl -= (1.0e-6)*double(init.microSeconds);
        return timeStampDbl;
    }

    void ImageGrabber::spikeDetected(unsigned int frameCount) {

        float imgGrab_time = (ts_nidaq[frameCount][1] - ts_nidaq[frameCount][0]) * 0.02;
        if (imgGrab_time > testConfig_->latency_threshold)
        {
            //cameraPtr_->skipDetected(stampImg);
            acquireLock();
            stampImg.isSpike = true;
            releaseLock();
            assert(stampImg.frameCount == frameCount);
            ts_nidaqThres[frameCount] = imgGrab_time;
        }

    }

    void ImageGrabber::connectSlots() 
    {
        
        QPointer<ImageGrabber>partnerImageGrabberPtr_ = partnerCameraWindowPtr->getImageGrabberPtr();
        qRegisterMetaType<uInt32>("uInt32");
        connect(this, SIGNAL(triggerSignal(uInt32, int)),
            partnerImageGrabberPtr_, SLOT(partnerTriggerSignal(uInt32, int)));

    }

    void ImageGrabber::partnerTriggerSignal(uInt32 read_buffer, int frameCount)
    {
 
        if (cameraNumber_ == 1) {

            nidaq_task_->acquireLock();
            DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_grab_in, 10.0, &read_ondemand_, NULL));
            nidaq_task_->releaseLock();

            ts_nidaq[frameCount - 1][0] = read_buffer;
            ts_nidaq[frameCount - 1][1] = read_ondemand_;

            /*if (((read_ondemand_ - read_buffer_) * 0.02) > testConfig_->latency_threshold)
            {
                cameraPtr_->skipDetected(stampImg);
            }*/

        }
       
    }


    unsigned int ImageGrabber::getPartnerCameraNumber()
    {
        // Returns camera number of partner camera. For this example
        // we just use camera 0 and 1. In another setting you might do
        // this by GUID or something else.
        if (cameraNumber_ == 0)
        {
            return 1;

        }else {

            return 0;

        }
    }

    QPointer<CameraWindow> ImageGrabber::getCameraWindow()
    {
        QPointer<CameraWindow> cameraWindowPtr = (CameraWindow*)(parent());
        return cameraWindowPtr;
    }

    QPointer<CameraWindow> ImageGrabber::getPartnerCameraWindowPtr()
    {
        QPointer<CameraWindow> partnerCameraWindowPtr = nullptr;
        if ((cameraWindowPtrList_->size()) > 1)
        {
            for (auto cameraWindowPtr : *cameraWindowPtrList_)
            {
                partnerCameraNumber_ = getPartnerCameraNumber();
               
                if ((cameraWindowPtr->getCameraNumber()) == partnerCameraNumber_)
                {
                   
                    partnerCameraWindowPtr = cameraWindowPtr;
                }
            }
        }
        return partnerCameraWindowPtr;
    }
   
} // namespace bias


