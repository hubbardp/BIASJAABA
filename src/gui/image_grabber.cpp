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

        if (testConfigEnabled) {

            if (!testConfig_->f2f_prefix.empty()) {

                gettime_ = gettime;
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

        std:cout << "trial num "  << trial_num << std::endl;
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

        StampedImage stampImg;

        TimeStamp timeStamp;
        TimeStamp timeStampInit; 

        double timeStampDbl = 0.0;
        double timeStampDblLast = 0.0;

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
                
                cameraPtr_->setupNIDAQ(nidaq_task_, gettime_, cameraNumber_);
               
            }
            else {
                //printf("nidaq not set");
                    
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

        
        int64_t pc_time;
        int64_t pc1, pc2;
        uInt32 read_buffer = 0, read_ondemand = 0;
     
        // Grab images from camera until the done signal is given
        while (!done)
        {

            acquireLock();
            done = stopped_;
            releaseLock();

            // Grab an image
            if (!istriggered && nidaq_task_ != nullptr && cameraNumber_ == 0) {

                nidaq_task_->startTasks();
                istriggered = true;

            }

            cameraPtr_->acquireLock();
            try
            {

                pc1 = 0, pc2 = 0;
                //if(startUpCount >= numStartUpSkip_)
                //    pc1 = gettime_->getPCtime();
                stampImg.image = cameraPtr_->grabImage();
                //if(startUpCount >= numStartUpSkip_)
                //    pc2 = gettime_->getPCtime();
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
                        DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_trigger_in, 10.0, &read_buffer, NULL));
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
                //// ---------------------------------------------------------------------

                // Set image data timestamp, framecount and frame interval estimate
                stampImg.timeStamp = timeStampDbl;
                stampImg.timeStampInit = timeStampInit;
                stampImg.timeStampVal = timeStamp;
                stampImg.frameCount = frameCount;
                stampImg.dtEstimate = dtEstimate;

                frameCount++;

                newImageQueuePtr_->acquireLock();
                newImageQueuePtr_->push(stampImg);
                newImageQueuePtr_->signalNotEmpty();
                newImageQueuePtr_->releaseLock();

                ///---------------------------------------------------------------
                if (testConfigEnabled_) {

                    if (nidaq_task_ != nullptr) {

                        if (cameraNumber_ == 0
                            && frameCount <= testConfig_->numFrames) {

                            nidaq_task_->acquireLock();
                            DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_trigger_in, 10.0, &read_buffer, NULL));
                            nidaq_task_->releaseLock();

                        }

                        if (frameCount <= testConfig_->numFrames) {

                            nidaq_task_->acquireLock();
                            DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_grab_in, 10.0, &read_ondemand, NULL));
                            nidaq_task_->releaseLock();

                        }

                    }

                    if (!testConfig_->nidaq_prefix.empty()) {

                        if (cameraNumber_ == 0)
                            time_stamps3[frameCount - 1][0] = read_buffer;
                        else
                            time_stamps3[frameCount - 1][0] = 0;

                        time_stamps3[frameCount - 1][1] = read_ondemand;
                    }

                    if (!testConfig_->f2f_prefix.empty()) {

                        pc_time = gettime_->getPCtime();
                        
                        if (frameCount <= testConfig_->numFrames)
                            time_stamps2[frameCount - 1] = pc_time;
                    }

                    if (!testConfig_->queue_prefix.empty()) {

                        if (frameCount <= testConfig_->numFrames)
                            queue_size[frameCount - 1] = newImageQueuePtr_->size();

                    }

                    if (frameCount == testConfig_->numFrames
                        && !testConfig_->f2f_prefix.empty())
                    {
              
                        std::string filename = testConfig_->dir_list[0] + "/"
                            + testConfig_->f2f_prefix + "/" + testConfig_->cam_dir 
                            +  "/" + testConfig_->imagegrab_prefix 
                            + "_" + testConfig_->f2f_prefix + "cam"
                            + std::to_string(cameraNumber_) + "_" + trial_num + ".csv";
                        gettime_->write_time_1d<int64_t>(filename, testConfig_->numFrames, time_stamps2);
                    }

                    if (frameCount == testConfig_->numFrames
                        && !testConfig_->nidaq_prefix.empty())
                    {

                        std::string filename = testConfig_->dir_list[0] + "/"
                            + testConfig_->nidaq_prefix + "/" + testConfig_->cam_dir 
                            + "/" + testConfig_->imagegrab_prefix
                            + "_" + testConfig_->nidaq_prefix + "cam"
                            + std::to_string(cameraNumber_) + "_" + trial_num + ".csv";
                        gettime_->write_time_2d<uInt32>(filename, testConfig_->numFrames, time_stamps3);

                    }

                    if (frameCount == testConfig_->numFrames
                        && !testConfig_->queue_prefix.empty()) {

                        string filename = testConfig_->dir_list[0] + "/"
                            + testConfig_->queue_prefix + "/" + testConfig_->cam_dir 
                            + "/" + testConfig_->imagegrab_prefix
                            + "_" + testConfig_->queue_prefix + "cam"
                            + std::to_string(cameraNumber_) + "_" + trial_num + ".csv";
                        
                        gettime_->write_time_1d<unsigned int>(filename, testConfig_->numFrames, queue_size);

                    }
                }

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

} // namespace bias


