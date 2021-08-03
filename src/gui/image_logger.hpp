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
                    std::shared_ptr<Lockable<GetTime>> gettime,
                    std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task,
                    QObject *parent=0
                    );

            void initialize(
                    unsigned int cameraNumber,
                    std::shared_ptr<VideoWriter> videoWriterPtr,
                    std::shared_ptr<LockableQueue<StampedImage>> logImageQueuePtr,
                    std::shared_ptr<Lockable<GetTime>> gettime,
                    std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task
                    );

            void stop();

            unsigned int getLogQueueSize();


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

            std::shared_ptr<VideoWriter> videoWriterPtr_;
            std::shared_ptr<LockableQueue<StampedImage>> logImageQueuePtr_;

            std::vector<unsigned int> queue_size;
            std::vector<uInt32> time_stamps1;
            std::vector<float>time_stamps2;
            void run();

            std::shared_ptr<Lockable<GetTime>> gettime_;
            std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task_;
    };

} // namespace bias

#endif // #ifndef BIAS_IMAGE_LOGGER_HPP
