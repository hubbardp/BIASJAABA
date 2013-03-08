#include "image_grabber.hpp"
#include "camera.hpp"
#include "exception.hpp"
#include "stamped_image.hpp"
#include "affinity.hpp"
#include <iostream>
#include <QTime>
#include <QThread>
#include <opencv2/core/core.hpp>

namespace bias {

    unsigned int ImageGrabber::NUM_STARTUP_SKIP = 15;
    unsigned int ImageGrabber::MAX_ERROR_COUNT = 500;

    ImageGrabber::ImageGrabber(QObject *parent) : QObject(parent) 
    {
        initialize(NULL,NULL);
    }

    ImageGrabber::ImageGrabber (
            std::shared_ptr<Lockable<Camera>> cameraPtr,
            std::shared_ptr<LockableQueue<StampedImage>> newImageQueuePtr, 
            QObject *parent
            ) : QObject(parent)
    {
        initialize(cameraPtr, newImageQueuePtr);
    }

    void ImageGrabber::initialize( 
            std::shared_ptr<Lockable<Camera>> cameraPtr,
            std::shared_ptr<LockableQueue<StampedImage>> newImageQueuePtr 
            ) 
    {
        capturing_ = false;
        stopped_ = true;
        cameraPtr_ = cameraPtr;
        newImageQueuePtr_ = newImageQueuePtr;
        if ((cameraPtr_ != NULL) && (newImageQueuePtr_ != NULL))
        {
            ready_ = true;
        }
        else
        {
            ready_ = false;
        }
    }

    void ImageGrabber::stop()
    {
        stopped_ = true;
    }

    void ImageGrabber::run()
    { 
        //double tLast = 0.0;

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
        assignThreadAffinity(true,1,0);


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

        // Grab images from camera until the done signal is given
        while (!done)
        {
            acquireLock();
            done = stopped_;
            releaseLock();

            // Grab an image
            cameraPtr_ -> acquireLock();
            try
            {
                stampImg.image = cameraPtr_ -> grabImage();
                timeStamp = cameraPtr_ -> getImageTimeStamp();
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

            // Push image into new image queue
            if (!error) 
            {
                //errorCount = 0;                  // Reset error count
                errorCount++;
                timeStampDblLast = timeStampDbl; // Save last timestamp
                
                // Set initial time stamp for fps estimate
                if (startUpCount == 0)
                {
                    timeStampInit = timeStamp;
                }

                // Reset initial time stamp for image acquisition
                if ((isFirst) && (startUpCount >= NUM_STARTUP_SKIP))
                {
                    timeStampInit = timeStamp;
                    timeStampDblLast = 0.0;
                    isFirst = false;
                }

                timeStampDbl  = double(timeStamp.seconds);
                timeStampDbl -= double(timeStampInit.seconds);
                timeStampDbl += (1.0e-6)*double(timeStamp.microSeconds);
                timeStampDbl -= (1.0e-6)*double(timeStampInit.microSeconds);

                // Skip some number of frames on startup - recommened by Point Grey. 
                // During this time compute running avg to get estimate of frame interval
                if (startUpCount < NUM_STARTUP_SKIP)
                {
                    double dt = timeStampDbl - timeStampDblLast;
                    if (startUpCount == 1)
                    {
                        dtEstimate = dt;

                    }
                    else if (startUpCount > 1)
                    {
                        double c0 = double(startUpCount-1)/double(startUpCount);
                        double c1 = double(1.0)/double(startUpCount);
                        dtEstimate = c0*dtEstimate + c1*dt;
                    }
                    startUpCount++;
                    continue;
                }

                // Set image data timestamp, framecount and frame interval estimate
                stampImg.timeStamp = timeStampDbl;
                stampImg.frameCount = frameCount;
                stampImg.dtEstimate = dtEstimate;
                frameCount++;

                newImageQueuePtr_ -> acquireLock();
                newImageQueuePtr_ -> push(stampImg);
                newImageQueuePtr_ -> signalNotEmpty(); 
                newImageQueuePtr_ -> releaseLock();

            }
            else
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

} // namespace bias


