#include "image_dispatcher.hpp"
#include "stamped_image.hpp"
#include "affinity.hpp"
#include <iostream>
#include <QThread>

// DEVEL
// ----------------------------------------------------------------------------
#include "camera_window.hpp"
#include <QDir>
#include <QFileInfo>
#include <fstream>
#include <QtDebug>
// ----------------------------------------------------------------------------

namespace bias
{

    ImageDispatcher::ImageDispatcher(QObject *parent) : QObject(parent)
    {
        initialize(false,false,0,NULL,NULL,NULL,NULL,NULL);
    }

    ImageDispatcher::ImageDispatcher( 
            bool logging,
            bool pluginEnabled,
            unsigned int cameraNumber,
            std::shared_ptr<Lockable<Camera>> cameraPtr,
            std::shared_ptr<LockableQueue<StampedImage>> newImageQueuePtr, 
            std::shared_ptr<LockableQueue<StampedImage>> logImageQueuePtr, 
            std::shared_ptr<LockableQueue<StampedImage>> pluginImageQueuePtr,
            std::shared_ptr<Lockable<GetTime>> gettime,
            QObject *parent
            ) : QObject(parent)
    {
        initialize(
                logging,
                pluginEnabled,
                cameraNumber,
                cameraPtr,
                newImageQueuePtr,
                logImageQueuePtr,
                pluginImageQueuePtr,
                gettime
                );
    }

    void ImageDispatcher::initialize(
            bool logging,
            bool pluginEnabled,
            unsigned int cameraNumber,
            std::shared_ptr<Lockable<Camera>> cameraPtr,
            std::shared_ptr<LockableQueue<StampedImage>> newImageQueuePtr,
            std::shared_ptr<LockableQueue<StampedImage>> logImageQueuePtr,
            std::shared_ptr<LockableQueue<StampedImage>> pluginImageQueuePtr,
            std::shared_ptr<Lockable<GetTime>> gettime
            ) 
    {
        newImageQueuePtr_ = newImageQueuePtr;
        logImageQueuePtr_ = logImageQueuePtr;
        pluginImageQueuePtr_ = pluginImageQueuePtr;
        if (
                (newImageQueuePtr_     != NULL) && 
                (logImageQueuePtr_     != NULL) &&
                (pluginImageQueuePtr_  != NULL)
           )

        {
            ready_ = true;
        }
        else 
        {
            ready_ = false;
        }

        stopped_ = true;
        logging_ = logging;
        cameraNumber_ = cameraNumber;
        cameraPtr_ = cameraPtr;
        pluginEnabled_ = pluginEnabled;

        frameCount_ = 0;
        currentTimeStamp_ = 0.0;

        //DEVEL
        gettime_ = gettime;
        //time_stamps1.resize(500000);
        //queue_size.resize(500000);
        
    }

    cv::Mat ImageDispatcher::getImage() const
    {
        cv::Mat currentImageCopy = currentImage_.clone();
        return currentImageCopy;
    }

    double ImageDispatcher::getTimeStamp() const
    {
        return currentTimeStamp_;
    }

    double ImageDispatcher::getFPS() const
    {
        return fpsEstimator_.getValue();
    }

    unsigned long ImageDispatcher::getFrameCount() const
    {
        return frameCount_;
    }

    void ImageDispatcher::stop()
    {
        stopped_ = true;

    }


    void ImageDispatcher::run()
    {
        bool done = false; 
        StampedImage newStampImage;
        int64_t pc_time;


        if (!ready_) 
        { 
            return; 
        }

        // Set thread priority to normal and assign cpu affinity
        QThread *thisThread = QThread::currentThread();
        thisThread -> setPriority(QThread::TimeCriticalPriority);
        ThreadAffinityService::assignThreadAffinity(false,cameraNumber_);

        // Initiaiize values
        acquireLock();
        frameCount_ = 0;
        stopped_ = false;
        fpsEstimator_.reset();
        releaseLock();

        // DEVEL - make this non development. Need to pass video file dir as argument
        // ---------------------------------------------------------------------------
        /*CameraWindow* cameraWindowPtr = qobject_cast<CameraWindow *>(parent());
        QDir videoFileDir = cameraWindowPtr -> getVideoFileDir();
        QString stampLogName = QString("stamp_log_cam%1.txt").arg(cameraNumber_);
        QFileInfo stampFileInfo = QFileInfo(videoFileDir, stampLogName);
        std::string stampFileName = stampFileInfo.absoluteFilePath().toStdString();
        std::ofstream stampOutStream;
        stampOutStream.open(stampFileName);*/
        // ---------------------------------------------------------------------------
        
 
        while (!done) 
        {

            newImageQueuePtr_ -> acquireLock();
            newImageQueuePtr_ -> waitIfEmpty();
            if (newImageQueuePtr_ -> empty())
            {
                newImageQueuePtr_ -> releaseLock();
                break;
            }
            newStampImage = newImageQueuePtr_ -> front();
            newImageQueuePtr_ -> pop();
            //queue_size[frameCount_] = newImageQueuePtr_->size();
            newImageQueuePtr_ -> releaseLock();
        
            /*if (frameCount_ == 499999) {
                gettime_->acquireLock();
                string filename = "imagegrab_queue_" + std::to_string(cameraNumber_) + ".csv";
                gettime_->write_time_1d<unsigned int>(filename, 500000, queue_size);
                gettime_->releaseLock();
                Sleep(10);
            }*/
            
            if (logging_ )
            {
                logImageQueuePtr_ -> acquireLock();
                logImageQueuePtr_ -> push(newStampImage);
                logImageQueuePtr_ -> signalNotEmpty();
                logImageQueuePtr_ -> releaseLock();
            }

            if (pluginEnabled_)
            {
                pluginImageQueuePtr_ -> acquireLock();
                pluginImageQueuePtr_ -> push(newStampImage);
                pluginImageQueuePtr_ -> signalNotEmpty();
                pluginImageQueuePtr_ -> releaseLock();
            }

            acquireLock();
            currentImage_ = newStampImage.image;
            currentTimeStamp_ = newStampImage.timeStamp;
            frameCount_ = newStampImage.frameCount;
            fpsEstimator_.update(newStampImage.timeStamp);
            done = stopped_;
            releaseLock();

            /************************************************************************************************************/
            /*gettime_->acquireLock();
            pc_time = gettime_->getPCtime();
            gettime_->releaseLock();

            if (frameCount_ < 500000)
                time_stamps1[frameCount_] = pc_time;*/

            /*if (frameCount_ == 500000)
            {

                std::string filename = "imagedispatch_f2f_" + std::to_string(cameraNumber_) + ".csv";
                gettime_->write_time_1d<int64_t>(filename, 500000, time_stamps1);
                Sleep(100);
            }*/


            /************************************************************************************************************/
            // DEVEL
            // ----------------------------------------------------------------
            //stampOutStream << QString::number(currentTimeStamp_,'g',15).toStdString(); 
            //stampOutStream << std::endl;
            // ----------------------------------------------------------------

        }

        // DEVEL
        // --------------------------------------------------------------------
        //stampOutStream.close();
        // --------------------------------------------------------------------
    }

} // namespace bias


