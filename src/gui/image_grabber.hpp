#ifndef BIAS_IMAGE_GRABBER_HPP  
#define BIAS_IMAGE_GRABBER_HPP

#include <memory>
#include <QMutex>
#include <QObject>
#include <QRunnable>
#include <QPointer>
#include <QThreadPool>

#include "basic_types.hpp"
#include "camera_fwd.hpp"
#include "lockable.hpp"
#include "camera_window.hpp"

#include "win_time.hpp"
#include "NIDAQUtils.hpp"
#include "test_config.hpp"
#include "video_utils.hpp"

// TEMPOERARY
// ----------------------------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "win_time.hpp"
#include "camera_device.hpp"
// ----------------------------------------

namespace bias
{

    class StampedImage;

    class ImageGrabber : public QObject, public QRunnable, public Lockable<Empty>
    {
        Q_OBJECT

        public:
            ImageGrabber(QObject *parent=0);

            ImageGrabber(
                    unsigned int cameraNumber,
                    std::shared_ptr<Lockable<Camera>> cameraPtr,
                    std::shared_ptr<LockableQueue<StampedImage>> newImageQueuePtr,
                    QPointer<QThreadPool> threadPoolPtr,
                    bool testConfigEnabled,
                    string trial_info,
                    std::shared_ptr<TestConfig> testConfig,
                    std::shared_ptr<Lockable<GetTime>> gettime,
                    std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task,
                    QObject *parent=0
                    );

            void initialize(
                    unsigned int cameraNumber,
                    std::shared_ptr<Lockable<Camera>> cameraPtr,
                    std::shared_ptr<LockableQueue<StampedImage>> newImageQueuePtr,
                    QPointer<QThreadPool> threadPoolPtr,
                    bool testConfigEnabled,
                    string trial_info,
                    std::shared_ptr<TestConfig> testConfig,
                    std::shared_ptr<Lockable<GetTime>> gettime,
                    std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task
                    );

            void stop();
            void enableErrorCount();
            void disableErrorCount();

            static unsigned int DEFAULT_NUM_STARTUP_SKIP;
            static unsigned int MIN_STARTUP_SKIP;
            static unsigned int MAX_ERROR_COUNT;

            void connectSlots();

        signals:
            void triggerSignal(uInt32 read_buffer, int frameCount);
            void startTimer();
            void startCaptureError(unsigned int errorId, QString errorMsg);
            void stopCaptureError(unsigned int errorId, QString errorMsg);
            void captureError(unsigned int errorId, QString errorMsg);

        private slots:
            void partnerTriggerSignal(uInt32 read_buffer, int frameCount);

        private:
            bool ready_;
            bool stopped_;
            bool capturing_;
            bool errorCountEnabled_;
            unsigned int numStartUpSkip_;
            unsigned int cameraNumber_;
            bool testConfigEnabled_;
            string trial_num;
            uint64_t fstfrmtStampRef_;

            //test
            bool process_frame_time_;

            unsigned int partnerCameraNumber_;
            uInt32 read_buffer_ = 0, read_ondemand_ = 0;
           
            std::shared_ptr<Lockable<Camera>> cameraPtr_;
            std::shared_ptr<LockableQueue<StampedImage>> newImageQueuePtr_;
            QSharedPointer<QList<QPointer<CameraWindow>>> cameraWindowPtrList_;
            QPointer<CameraWindow> partnerCameraWindowPtr;

            void run();
            double convertTimeStampToDouble(TimeStamp curr, TimeStamp init);
            QPointer<QThreadPool> threadPoolPtr_;

            std::shared_ptr<Lockable<GetTime>> gettime_;
            std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task_;
            std::shared_ptr<TestConfig>testConfig_;
            
            //test
            std::vector<float> ts_nidaqThres; //nidaq thres
            std::vector<int64_t> ts_pc; // pc timings
            std::vector<std::vector<uInt32>>ts_nidaq; // nidaq timings
            std::vector<unsigned int> queue_size; // queue size
            std::vector<int64_t>ts_process;

            priority_queue<int, vector<int>, greater<int>>delayFrames_view1; // side skips
            priority_queue<int, vector<int>, greater<int>>delayFrames_view2; // front skips
            vector<int> delay_view1;
            vector<int> delay_view2;

            videoBackend* vid_obj_;
            cv::VideoCapture cap_obj_;

            //void spikeDetected(unsigned int frameCount);
            void initializeVidBackend();
            void initiateVidFrameDelay(priority_queue<int, vector<int>, greater<int>>& skip_frames,
                                       vector<int>& delay_frames);
            unsigned int getPartnerCameraNumber();
            QPointer<CameraWindow> getCameraWindow();
            QPointer<CameraWindow> getPartnerCameraWindowPtr();
    };


    //class TSleepThread : QThread
    //{
    //    public:
    //        static void sleep(unsigned long secs) { QThread::sleep(secs); };
    //        static void msleep(unsigned long msecs) { QThread::msleep(msecs); };
    //        static void usleep(unsigned long usecs) { QThread::usleep(usecs); };
    //};


} // namespace bias

#endif // #ifndef BIAS_IMAGE_GRABBER_HPP
