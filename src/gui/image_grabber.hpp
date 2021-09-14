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

#include "win_time.hpp"
#include "NIDAQUtils.hpp"
#include "test_config.hpp"


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

        signals:
            void startTimer();
            void startCaptureError(unsigned int errorId, QString errorMsg);
            void stopCaptureError(unsigned int errorId, QString errorMsg);
            void captureError(unsigned int errorId, QString errorMsg);

        private:
            bool ready_;
            bool stopped_;
            bool capturing_;
            bool errorCountEnabled_;
            unsigned int numStartUpSkip_;
            unsigned int cameraNumber_;
            bool testConfigEnabled_;

            std::shared_ptr<Lockable<Camera>> cameraPtr_;
            std::shared_ptr<LockableQueue<StampedImage>> newImageQueuePtr_;

            void run();
            double convertTimeStampToDouble(TimeStamp curr, TimeStamp init);
            QPointer<QThreadPool> threadPoolPtr_;

            std::shared_ptr<Lockable<GetTime>> gettime_;
            std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task_;
            std::shared_ptr<TestConfig>testConfig_;

            //test
            std::vector<float> time_stamps1;
            std::vector<int64_t> time_stamps2;
            std::vector<std::vector<uInt32>>time_stamps3;
            std::vector<unsigned int> queue_size;
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
