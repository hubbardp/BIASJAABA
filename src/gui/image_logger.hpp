#ifndef BIAS_IMAGE_LOGGER_HPP
#define BIAS_IMAGE_LOGGER_HPP

#include <memory>
#include <QMutex>
#include <QObject>
#include <QRunnable>
#include "camera_fwd.hpp"
#include "lockable.hpp"

#include "win_time.hpp"
#include "NIDAQUtils.hpp"
#include "timerClass.hpp"
#include "test_config.hpp"


// Debugging -------------------
//#include <opencv2/core/core.hpp>
// -----------------------------

class QString;


namespace bias
{

    class VideoWriter;

    class StampedImage;

    class ImageLogger : public QObject, public QRunnable, public Lockable<Empty>
    {
        Q_OBJECT

        public:
            ImageLogger(QObject *parent=0);

            ImageLogger(
                    unsigned int cameraNumber,
                    std::shared_ptr<VideoWriter> videoWriterPtr,
                    std::shared_ptr<LockableQueue<StampedImage>> logImageQueuePtr,
                    bool testConfigEnabled,
                    string trial_info,
                    std::shared_ptr<TestConfig> testConfig,
                    std::shared_ptr<Lockable<TimerClass>> timerClass,
                    QObject *parent=0
                    );

            void initialize(
                    unsigned int cameraNumber,
                    std::shared_ptr<VideoWriter> videoWriterPtr,
                    std::shared_ptr<LockableQueue<StampedImage>> logImageQueuePtr,
                    bool testConfigEnabled,
                    string trial_info,
                    std::shared_ptr<TestConfig> testConfig,
                    std::shared_ptr<Lockable<TimerClass>> timerClass
                    );

            void stop();

            unsigned int getLogQueueSize();
            void resetImageLoggerParams(QString videoFileFullPath);
            

            // Debugging --------------------------
            //cv::Mat getBackgroundMembershipImage();
            // ------------------------------------

        signals:
            void imageLoggingError(unsigned int errorId, QString errorMsg);

        private:
            bool ready_;
            bool stopped_;
            unsigned long frameCount_;
            unsigned int cameraNumber_;
            unsigned int logQueueSize_;
            bool testConfigEnabled_;
            string trial_num_;

            std::shared_ptr<VideoWriter> videoWriterPtr_;
            std::shared_ptr<LockableQueue<StampedImage>> logImageQueuePtr_;


            std::vector<float> time_stamps1;
            std::vector<int64_t> time_stamps2;
            std::vector<uInt32>time_stamps3;
            std::vector<unsigned int> queue_size;
            
            void run();

            std::shared_ptr<Lockable<GetTime>> gettime_;
            std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task_;
            std::shared_ptr <Lockable<TimerClass>> timerClass_;
            std::shared_ptr<TestConfig>testConfig_;
    };

} // namespace bias

#endif // #ifndef BIAS_IMAGE_LOGGER_HPP
