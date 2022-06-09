#ifndef BIAS_IMAGE_PROCESSOR_HPP
#define BIAS_IMAGE_PROCESSOR_HPP

#include <memory>
#include <QMutex>
#include <QObject>
#include <QRunnable>
#include <opencv2/core/core.hpp>
#include "fps_estimator.hpp"
#include "lockable.hpp"
#include "camera_fwd.hpp"
#include "NIDAQUtils.hpp"


// DEVEL 
#include "win_time.hpp"
#include "test_config.hpp"

namespace bias
{

    class StampedImage;

    class ImageDispatcher : public QObject, public QRunnable, public Lockable<Empty>
    {
        Q_OBJECT

        public:
            ImageDispatcher(QObject *parent=0);

            ImageDispatcher( 
                    bool logging,
                    bool pluginEnabled,
                    unsigned int cameraNumber,
                    std::shared_ptr<Lockable<Camera>> cameraPtr,
                    std::shared_ptr<LockableQueue<StampedImage>> newImageQueuePtr, 
                    std::shared_ptr<LockableQueue<StampedImage>> logImageQueuePtr, 
                    std::shared_ptr<LockableQueue<StampedImage>> pluginImageQueuePtr,
                    std::shared_ptr<LockableQueue<unsigned int>> skippedFramesPluginPtr,
                    bool testConfigEnabled,
                    string trial_info,
                    std::shared_ptr<TestConfig> testConfig,
                    std::shared_ptr<Lockable<GetTime>> gettime,
                    std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task,
                    QObject *parent = 0
                    );

            void initialize( 
                    bool logging,
                    bool pluginEnabled,
                    unsigned int cameraNumber,
                    std::shared_ptr<Lockable<Camera>> cameraPtr,
                    std::shared_ptr<LockableQueue<StampedImage>> newImageQueuePtr,
                    std::shared_ptr<LockableQueue<StampedImage>> logImageQueuePtr ,
                    std::shared_ptr<LockableQueue<StampedImage>> pluginImageQueuePtr,
                    std::shared_ptr<LockableQueue<unsigned int>> skippedFramesPluginPtr,
                    bool testConfigEnabled,
                    string trial_info,
                    std::shared_ptr<TestConfig> testConfig,
                    std::shared_ptr<Lockable<GetTime>> gettime,
                    std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task
                    );

            // Use lock when calling these methods
            // ----------------------------------
            void stop();
            cv::Mat getImage() const;     // Note, might want to change so that we return 
            double getTimeStamp() const;  // the stampedImage.
            double getFPS() const;
            unsigned long getFrameCount() const;
            // -----------------------------------

        private:
            bool ready_;
            bool logging_;
            bool pluginEnabled_;
            unsigned int cameraNumber_;
            bool testConfigEnabled_;
            string trial_num;
            bool process_frame_time;
            uInt32 read_buffer = 0, read_ondemand = 0;

            std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task_;
            std::shared_ptr<Lockable<Camera>> cameraPtr_;
            std::shared_ptr<LockableQueue<StampedImage>> newImageQueuePtr_;
            std::shared_ptr<LockableQueue<StampedImage>> logImageQueuePtr_;
            std::shared_ptr<LockableQueue<StampedImage>> pluginImageQueuePtr_;
            std::shared_ptr<LockableQueue<unsigned int>> skippedFramesPluginPtr_;
            
            // use lock when setting these values
            // -----------------------------------
            bool stopped_;
            cv::Mat currentImage_;        // Note, might want to change so that we store
            double currentTimeStamp_;     // the stampedImage. 
            FPS_Estimator fpsEstimator_;
            unsigned long frameCount_;
            // ------------------------------------

            void run();

            std::shared_ptr<TestConfig>testConfig_;
            std::shared_ptr<Lockable<GetTime>> gettime_;
            std::vector<unsigned int>queue_size;
            std::vector<std::vector<uInt32>> ts_nidaq;
            std::vector<float>ts_nidaqThres;
            std::vector<int64_t>ts_pc;
            std::vector<int64_t>ts_process;
            
    };

} // namespace bias

#endif // #ifndef BIAS_IMAGE_PROCESSOR_HPP
