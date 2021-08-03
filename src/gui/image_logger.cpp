#include "image_logger.hpp"
#include "basic_types.hpp"
#include "exception.hpp"
#include "stamped_image.hpp"
#include "video_writer.hpp"
#include "affinity.hpp"
#include <QThread>
#include <queue>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

namespace bias
{
    const unsigned int MAX_LOG_QUEUE_SIZE = 1000;

    ImageLogger::ImageLogger(QObject *parent) : QObject(parent) 
    {
        initialize(0, NULL,NULL,NULL,NULL);
    }

    ImageLogger::ImageLogger (
            unsigned int cameraNumber,
            std::shared_ptr<VideoWriter> videoWriterPtr,
            std::shared_ptr<LockableQueue<StampedImage>> logImageQueuePtr,
            std::shared_ptr<Lockable<GetTime>> gettime,
            std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task,
            QObject *parent
            ) : QObject(parent)
    {
        initialize(cameraNumber, videoWriterPtr, logImageQueuePtr, gettime,nidaq_task);
    }

    void ImageLogger::initialize( 
            unsigned int cameraNumber,
            std::shared_ptr<VideoWriter> videoWriterPtr,
            std::shared_ptr<LockableQueue<StampedImage>> logImageQueuePtr,
            std::shared_ptr<Lockable<GetTime>> gettime,
            std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task
            ) 
    {
        frameCount_ = 0;
        stopped_ = true;
        cameraNumber_ = cameraNumber;
        videoWriterPtr_ = videoWriterPtr;
        logImageQueuePtr_ = logImageQueuePtr;
        logQueueSize_ = 0;
        if ((logImageQueuePtr_ != NULL) && (videoWriterPtr_ != NULL))
        {
            ready_ = true;
        }
        else
        {
            ready_ = false;
        }
        gettime_ = gettime;
        nidaq_task_ = nidaq_task;
        //queue_size.resize(500000);
        time_stamps2.resize(500000);
        //time_stamps1.resize(500000);
    }

    void ImageLogger::stop()
    {
        stopped_ = true;
    }

    unsigned int ImageLogger::getLogQueueSize()
    {
        return logQueueSize_;
    }

    void ImageLogger::run()
    {
        bool done = false;
        bool errorFlag = false;
        StampedImage newStampedImage;
        size_t logQueueSize;
        uInt32 read_ondemand = 0;
        int64_t pc_time;
        int64_t pc1, pc2;

        if (!ready_) 
        { 
            return; 
        }

        // Set thread priority to normal
        QThread *thisThread = QThread::currentThread();
        thisThread -> setPriority(QThread::NormalPriority);
        ThreadAffinityService::assignThreadAffinity(false,cameraNumber_);

        acquireLock();
        stopped_ = false;
        frameCount_ = 0;
        int startUpCount = 0;
        releaseLock();
        cv::Mat tmp_img;

        if (nidaq_task_ != nullptr) {

        }
        else {
            printf("nidaq not set-%d", cameraNumber_);
        }


        while (!done)
        {
            pc1 = 0, pc2 = 0;
            
            //pc1 = gettime_->getPCtime();
            logImageQueuePtr_ -> acquireLock();
            logImageQueuePtr_ -> waitIfEmpty();
            if (logImageQueuePtr_ -> empty())
            {
                logImageQueuePtr_ -> releaseLock();
                std::cout << "inside......." << std::endl;
                break;

            }
            newStampedImage = logImageQueuePtr_ -> front();
            frameCount_ = newStampedImage.frameCount;
            logImageQueuePtr_ -> pop();
            logQueueSize =  logImageQueuePtr_ -> size();
            logImageQueuePtr_ -> releaseLock();
            //pc2 = gettime_->getPCtime();

            /*if(frameCount_ < 500000)
                queue_size[frameCount_] = logQueueSize;

            if (frameCount_ == 499999) {

                gettime_ -> acquireLock();
                string filename = "log_queue_" + std::to_string(cameraNumber_) + ".csv";
                gettime_->write_time_1d<unsigned int>(filename, 500000, queue_size);
                gettime_ -> releaseLock();
                
            }*/

            //frameCount_++;
            
            if (!errorFlag) 
            {
                // Check if log queue has grown too large - if so signal an error
                if (logQueueSize > MAX_LOG_QUEUE_SIZE)
                {
                    unsigned int errorId = ERROR_IMAGE_LOGGER_MAX_QUEUE_SIZE;
                    QString errorMsg("logger image queue has exceeded the maximum allowed size");
                    emit imageLoggingError(errorId, errorMsg);
                    errorFlag = true;
                }
                //std::cout << "cam: " << cameraNumber_ << ", queue size: " << logQueueSize;
                //std::cout << "/" << MAX_LOG_QUEUE_SIZE << std::endl;

                // Add frame to video writer
                pc1 = gettime_->getPCtime();
                try 
                {
                    videoWriterPtr_ -> addFrame(newStampedImage);
                    //tmp_img = newStampedImage.image;
                    //imwrite("./out_feat/imgtest_" + std::to_string(newStampedImage.frameCount) + ".jpg",newStampedImage.image);
                    //newStampedImage.image.convertTo(tmp_img, CV_32FC1);
                    //tmp_img = tmp_img / 255;
                    //write_output("./out_feat/img_" + std::to_string(newStampedImage.frameCount) + ".csv", tmp_img.ptr<float>(0),
                    //          tmp_img.cols, tmp_img.rows);

                }
                catch (RuntimeError &runtimeError)
                {
                    unsigned int errorId = runtimeError.id();;
                    QString errorMsg = QString::fromStdString(runtimeError.what());
                    emit imageLoggingError(errorId, errorMsg);
                }
                pc2 = gettime_->getPCtime();
                
            }

            /*if (nidaq_task_ != nullptr) {
                nidaq_task_->acquireLock();
                DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_grab_in, 10.0, &read_ondemand, NULL));
                time_stamps1[frameCount_-1] = read_ondemand;
                nidaq_task_->releaseLock();

                if (frameCount_ == 500000) {

                    gettime_->acquireLock();
                    std::string filename1 = "logger_process_time_cam2sys" + to_string(cameraNumber_) + ".csv";
                    gettime_->write_time_1d<uInt32>(filename1, 500000, time_stamps1);
                    gettime_->releaseLock();
                    
                }
            }*/

            /*gettime_->acquireLock();
            pc_time = gettime_->getPCtime();
            gettime_->releaseLock();*/
   
            if (frameCount_ >= 0 && frameCount_ < 500000)
                time_stamps2[frameCount_] = (pc2-pc1)*1e-3;

            if (frameCount_ == 499999)
            {
                std::string filename = "imagelogger_vid_f2f_" + std::to_string(cameraNumber_) + ".csv";
                gettime_->write_time_1d<float>(filename, 500000, time_stamps2);
                
            }

            acquireLock();
            done = stopped_;
            logQueueSize_ = static_cast<unsigned int>(logQueueSize);
            releaseLock();

        } // while (!done)

        try
        {
            videoWriterPtr_ -> finish();
        }
        catch (RuntimeError &runtimeError)
        {
            unsigned int errorId = runtimeError.id();;
            QString errorMsg = QString::fromStdString(runtimeError.what());
            emit imageLoggingError(errorId, errorMsg);
        }
    
    }  // void ImageLogger::run()


} // namespace bias

