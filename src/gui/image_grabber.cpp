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
        initialize(0,NULL,NULL, NULL);
    }

    ImageGrabber::ImageGrabber (
            unsigned int cameraNumber,
            std::shared_ptr<Lockable<Camera>> cameraPtr,
            std::shared_ptr<LockableQueue<StampedImage>> newImageQueuePtr,
            GetTime *gettime,
            QObject *parent
            ) : QObject(parent)
    {
        initialize(cameraNumber, cameraPtr, newImageQueuePtr, gettime);
    }

    void ImageGrabber::initialize( 
            unsigned int cameraNumber,
            std::shared_ptr<Lockable<Camera>> cameraPtr,
            std::shared_ptr<LockableQueue<StampedImage>> newImageQueuePtr,
            GetTime *gettime
            ) 
    {
        capturing_ = false;
        stopped_ = true;
        cameraPtr_ = cameraPtr;
        newImageQueuePtr_ = newImageQueuePtr;
        numStartUpSkip_ = DEFAULT_NUM_STARTUP_SKIP;
        cameraNumber_ = cameraNumber;
        if ((cameraPtr_ != NULL) && (newImageQueuePtr_ != NULL))
        {
            ready_ = true;
        }
        else
        {
            ready_ = false;
        }
        errorCountEnabled_ = true;

        gettime_ = gettime;
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
        bool done = false;
        bool error = false;
        bool errorEmitted = false;
        unsigned int errorId = 0;
        unsigned int errorCount = 0;
        unsigned long frameCount = 0;
        unsigned long startUpCount = 0;
        double dtEstimate = 0.0;

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

        // Start image capture
        cameraPtr_ -> acquireLock();
        try
        {
            cameraPtr_ -> startCapture();
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

        
        TimeStamp pc_time, pc_1, pc_2;
        int64_t pc_ts1, pc_ts2, cam_ts1, cam_ts2;
        cameraPtr_ -> cameraOffsetTime();
        
        // Grab images from camera until the done signal is given
        while (!done)
        {
            acquireLock();
            done = stopped_;
            releaseLock();

            // Grab an image
            pc_1 = gettime_->getPCtime();
            cameraPtr_ -> acquireLock();
            try
            {
          
                stampImg.image = cameraPtr_ -> grabImage();
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
            cameraPtr_ -> releaseLock();
            

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
                        double c0 = double(startUpCount-1)/double(startUpCount);
                        double c1 = double(1.0)/double(startUpCount);
                        dtEstimate = c0*dtEstimate + c1*dt;
                    }
                    startUpCount++;
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

                newImageQueuePtr_ -> acquireLock();
                newImageQueuePtr_ -> push(stampImg);
                newImageQueuePtr_ -> signalNotEmpty(); 
                newImageQueuePtr_ -> releaseLock();

                //pc_1 = cameraPtr_->getCPUtime();
                pc_2 = gettime_->getPCtime();
                pc_ts2 = (pc_2.seconds*1e6 + pc_2.microSeconds )- (cameraPtr_->cam_ofs.seconds*1e6
                + cameraPtr_->cam_ofs.microSeconds);
                pc_ts1 = (pc_2.seconds*1e6 + pc_2.microSeconds) - (pc_1.seconds*1e6 + pc_1.microSeconds);
                cam_ts2 = timeStamp.seconds * 1e6 + timeStamp.microSeconds;
                time_stamps1.push_back({cam_ts2, pc_ts2});
                //time_stamps2.push_back({ cam_ts2, pc_ts2 - cam_ts2 });

                if (time_stamps1.size() == 1500000)
                {
                    std::string filename = "imagegrab_f2f" + std::to_string(cameraNumber_) + ".csv";
                    gettime_->write_time<int64_t>(filename, 1500000, time_stamps1);
                }

            }
            else
            {
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


