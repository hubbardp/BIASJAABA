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
        initialize(0,NULL,NULL,false,"", NULL, NULL, NULL);
    }

    ImageLogger::ImageLogger (
            unsigned int cameraNumber,
            std::shared_ptr<VideoWriter> videoWriterPtr,
            std::shared_ptr<LockableQueue<StampedImage>> logImageQueuePtr,
            bool testConfigEnabled,
            string trial_info,
            std::shared_ptr<TestConfig> testConfig,
            std::shared_ptr<Lockable<GetTime>> gettime,
            std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task,
            QObject *parent
            ) : QObject(parent)
    {
        initialize(cameraNumber, videoWriterPtr, logImageQueuePtr, 
                   testConfigEnabled, trial_info, testConfig, gettime, nidaq_task);
    }

    void ImageLogger::initialize( 
            unsigned int cameraNumber,
            std::shared_ptr<VideoWriter> videoWriterPtr,
            std::shared_ptr<LockableQueue<StampedImage>> logImageQueuePtr,
            bool testConfigEnabled,
            string trial_info,
            std::shared_ptr<TestConfig> testConfig,
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
        testConfigEnabled_ = testConfigEnabled;
        testConfig_ = testConfig;
        trial_num_ = trial_info;

        if (testConfigEnabled) {

            if (!testConfig_->f2f_prefix.empty()) {

                gettime_ = gettime;
                time_stamps2.resize(testConfig_->numFrames);
            }

            if (!testConfig_->nidaq_prefix.empty()) {

                time_stamps3.resize(testConfig_->numFrames);
            }

            if (!testConfig_->queue_prefix.empty()) {

                time_stamps1.resize(testConfig_->numFrames);
            }
        }
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
        //ThreadAffinityService::assignThreadAffinity(false,cameraNumber_);

        acquireLock();
        stopped_ = false;
        frameCount_ = 0;
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
                break;

            }
            newStampedImage = logImageQueuePtr_ -> front();
            frameCount_ = newStampedImage.frameCount;
            logImageQueuePtr_ -> pop();
            logQueueSize =  logImageQueuePtr_ -> size();
            logImageQueuePtr_ -> releaseLock();
            //pc2 = gettime_->getPCtime();

            frameCount_++;
            
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
                //pc1 = gettime_->getPCtime();
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
                //pc2 = gettime_->getPCtime();
                
            }

            if (testConfigEnabled_) {

                if (nidaq_task_ != nullptr) {

                    if (frameCount_ <= testConfig_->numFrames) {

                        nidaq_task_->acquireLock();
                        DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_grab_in, 10.0, &read_ondemand, NULL));
                        nidaq_task_->releaseLock();

                    }

                }

                if (!testConfig_->nidaq_prefix.empty()) {

                    time_stamps3[frameCount_ - 1] = read_ondemand;
                }

                if (!testConfig_->f2f_prefix.empty()) {

                    pc_time = gettime_->getPCtime();

                    if (frameCount_ <= testConfig_->numFrames)
                        time_stamps2[frameCount_ - 1] = pc_time;
                }

                if (!testConfig_->queue_prefix.empty()) {

                    if (frameCount_ <= testConfig_->numFrames)
                        queue_size[frameCount_ - 1] = logImageQueuePtr_->size();

                }

                if (frameCount_ == testConfig_->numFrames
                    && !testConfig_->f2f_prefix.empty())
                {

                    std::string filename = testConfig_->dir_list[0] + "/"
                        + testConfig_->f2f_prefix + "/" + testConfig_->cam_dir
                        + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                        + testConfig_->logging_prefix
                        + "_" + testConfig_->f2f_prefix + "cam"
                        + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";

                    gettime_->write_time_1d<int64_t>(filename, testConfig_->numFrames, time_stamps2);

                }

                if (frameCount_ == testConfig_->numFrames
                    && !testConfig_->nidaq_prefix.empty())
                {

                    std::string filename = testConfig_->dir_list[0] + "/"
                        + testConfig_->nidaq_prefix + "/" + testConfig_->cam_dir
                        + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                        + testConfig_->logging_prefix
                        + "_" + testConfig_->nidaq_prefix + "cam"
                        + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";

                    gettime_->write_time_1d<uInt32>(filename, testConfig_->numFrames, time_stamps3);

                }

                if (frameCount_ == testConfig_->numFrames
                    && !testConfig_->queue_prefix.empty()) {

                    string filename = testConfig_->dir_list[0] + "/"
                        + testConfig_->queue_prefix + "/" + testConfig_->cam_dir
                        + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                        + testConfig_->logging_prefix
                        + "_" + testConfig_->queue_prefix + "cam"
                        + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";

                    gettime_->write_time_1d<unsigned int>(filename, testConfig_->numFrames, queue_size);

                }

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
        std::cout << "DEBUG: Image Logger run method exited " << std::endl;
    
    }  // void ImageLogger::run()

    void ImageLogger::resetImageLoggerParams()
    {
        stopped_ = false;
        frameCount_ = 0;

        videoWriterPtr_->resetVideoWriterParams();

    }


} // namespace bias

