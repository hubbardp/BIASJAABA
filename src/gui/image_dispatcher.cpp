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

#define DEBUG 0

namespace bias
{

    ImageDispatcher::ImageDispatcher(QObject *parent) : QObject(parent)
    {
        initialize(false,false,0,NULL,NULL,NULL,NULL,false,"",NULL,NULL,NULL);
    }

    ImageDispatcher::ImageDispatcher( 
            bool logging,
            bool pluginEnabled,
            unsigned int cameraNumber,
            std::shared_ptr<Lockable<Camera>> cameraPtr,
            std::shared_ptr<LockableQueue<StampedImage>> newImageQueuePtr, 
            std::shared_ptr<LockableQueue<StampedImage>> logImageQueuePtr, 
            std::shared_ptr<LockableQueue<StampedImage>> pluginImageQueuePtr,
            bool testConfigEnabled,
            string trial_info,
            std::shared_ptr<TestConfig> testConfig,
            std::shared_ptr<Lockable<GetTime>> gettime,
            std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task,
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
                testConfigEnabled,
                trial_info,
                testConfig,
                gettime,
                nidaq_task
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
            bool testConfigEnabled,
            string trial_info,
            std::shared_ptr<TestConfig> testConfig,
            std::shared_ptr<Lockable<GetTime>> gettime,
            std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task
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
        testConfigEnabled_ = testConfigEnabled;
        testConfig_ = testConfig;
        trial_num = trial_info;
        nidaq_task_ = nidaq_task;
        process_frame_time = 1;

#if DEBUG
        if (testConfigEnabled_ && !testConfig_->queue_prefix.empty()) {
            
            queue_size.resize(testConfig_->numFrames,0);
            
        }

        if (testConfigEnabled_ && !testConfig_->nidaq_prefix.empty()) {

            ts_nidaq.resize(testConfig_->numFrames, std::vector<uInt32>(2, 0));
            ts_nidaqThres.resize(testConfig_->numFrames, 0.0);
        }

        if (testConfigEnabled_ && !testConfig_->f2f_prefix.empty()) {

            ts_pc.resize(testConfig_->numFrames, 0);
        }

        if (testConfigEnabled_ && process_frame_time)
        {
            ts_process.resize(testConfig_->numFrames, 0);
        }
#endif
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
        int64_t pc_time, start_process, end_process;
        float imgDispatchTime;
        process_frame_time = 1;

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

            start_process = gettime_->getPCtime();
            newImageQueuePtr_->acquireLock();
            newImageQueuePtr_->waitIfEmpty();
            if (newImageQueuePtr_->empty())
            {
                newImageQueuePtr_->releaseLock();
                break;
            }

            newStampImage = newImageQueuePtr_->front();
            newImageQueuePtr_->pop();
            newImageQueuePtr_->releaseLock();
            
            if (logging_)
            {
                logImageQueuePtr_->acquireLock();
                logImageQueuePtr_->push(newStampImage);
                logImageQueuePtr_->signalNotEmpty();
                logImageQueuePtr_->releaseLock();
            }

            if (pluginEnabled_)
            {
                pluginImageQueuePtr_->acquireLock();
                frameCount_ = newStampImage.frameCount;
                pluginImageQueuePtr_->push(newStampImage);
                pluginImageQueuePtr_->signalNotEmpty();
                pluginImageQueuePtr_->releaseLock();

            }

            acquireLock();
            currentImage_ = newStampImage.image;
            currentTimeStamp_ = newStampImage.timeStamp;
            frameCount_ = newStampImage.frameCount;
            fpsEstimator_.update(newStampImage.timeStamp);
            done = stopped_;
            releaseLock();
            end_process = gettime_->getPCtime();
            
#if DEBUG
            if (testConfigEnabled_) {
                
                if (frameCount_ < testConfig_->numFrames) {

                    if (nidaq_task_ != nullptr) {

                        nidaq_task_->acquireLock();
                        DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_grab_in, 10.0, &read_ondemand, NULL));
                        nidaq_task_->releaseLock();

                    }

                    if (!testConfig_->nidaq_prefix.empty()) {

                        imgDispatchTime = (read_ondemand - nidaq_task_->cam_trigger[frameCount_])*0.02;
                        ts_nidaq[frameCount_][0] = nidaq_task_->cam_trigger[frameCount_];
                        ts_nidaq[frameCount_][1] = read_ondemand;
                        if (imgDispatchTime > 4.0)
                        {
                            ts_nidaqThres[frameCount_] = imgDispatchTime;
                        }
                    }

                    if (!testConfig_->f2f_prefix.empty()) {

                        acquireLock();
                        ts_pc[frameCount_] = gettime_->getPCtime();
                        releaseLock();
                    }

                    if (process_frame_time)
                    {
                        ts_process[frameCount_] = end_process - start_process;
                    }
                }

                if (frameCount_ == testConfig_->numFrames - 1
                    && !testConfig_->queue_prefix.empty()) {

                    string filename = testConfig_->dir_list[0] + "/"
                        + testConfig_->queue_prefix + "/" + testConfig_->cam_dir
                        + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                        + "imagedispatch"
                        + "_" + testConfig_->queue_prefix + "cam"
                        + std::to_string(cameraNumber_) + "_" + trial_num + ".csv";

                    gettime_->write_time_1d<unsigned int>(filename, testConfig_->numFrames, queue_size);
                }

                if (frameCount_ == testConfig_->numFrames - 1
                    && !testConfig_->nidaq_prefix.empty()) {

                    string filename = testConfig_->dir_list[0] + "/"
                        + testConfig_->nidaq_prefix + "/" + testConfig_->cam_dir
                        + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                        + "imagedispatch_"
                        + testConfig_->nidaq_prefix + "cam"
                        + std::to_string(cameraNumber_) + "_" + trial_num + ".csv";


                    string filename1 = testConfig_->dir_list[0] + "/"
                        + testConfig_->nidaq_prefix + "/" + testConfig_->cam_dir
                        + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                        + "imagedispatch"
                        + "_" + "nidaq_thres" + "cam"
                        + std::to_string(cameraNumber_) + "_" + trial_num + ".csv";

                    gettime_->write_time_2d<uInt32>(filename, testConfig_->numFrames, ts_nidaq);
                    gettime_->write_time_1d<float>(filename1, testConfig_->numFrames, ts_nidaqThres);
                }

                if (frameCount_ == testConfig_->numFrames - 1
                    && !testConfig_->f2f_prefix.empty()) {

                    string filename = testConfig_->dir_list[0] + "/"
                        + testConfig_->f2f_prefix + "/" + testConfig_->cam_dir
                        + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                        + "imagedispatch"
                        + "_" + testConfig_->f2f_prefix + "cam"
                        + std::to_string(cameraNumber_) + "_" + trial_num + ".csv";

                    gettime_->write_time_1d<int64_t>(filename, testConfig_->numFrames, ts_pc);
                }

                if (frameCount_ == testConfig_->numFrames - 1
                    && process_frame_time) {

                    string filename = testConfig_->dir_list[0] + "/"
                        + testConfig_->nidaq_prefix + "/" + testConfig_->cam_dir
                        + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                        + "imagedispatch"
                        + "_" + "process_time" + "cam"
                        + std::to_string(cameraNumber_) + "_" + trial_num + ".csv";
                    
                    gettime_->write_time_1d<int64_t>(filename, testConfig_->numFrames, ts_process);
                }
            }
#endif
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


