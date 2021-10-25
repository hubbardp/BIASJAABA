#ifndef SIGNAL_SLOT_DEMO_HPP
#define SIGNAL_SLOT_DEMO_HPP
#include "ui_signal_slot_demo_plugin.h"
#include "bias_plugin.hpp"
#include "frame_data.hpp"
#include <QPointer>
#include <QVector>
#include <QList>

#include "test_config.hpp"

namespace cv
{
    class Mat;
}


namespace bias
{

    class ImageLabel;
    class CameraWindow;


    class SignalSlotDemoPlugin : public BiasPlugin, public Ui::SignalSlotDemoPluginDialog
    {
        Q_OBJECT

        public:

            static const QString PLUGIN_NAME;
            static const QString PLUGIN_DISPLAY_NAME;

            SignalSlotDemoPlugin(ImageLabel *imageLabelPtr, 
                std::shared_ptr<Lockable<GetTime>> gettime,
                bool testConfigEnabled,
                string trial_info,
                std::shared_ptr<TestConfig> testConfig,
                QWidget *parentPtr=0);

            virtual void finalSetup();
            virtual void reset();
            virtual void stop();

            //virtual void processFrames(QList<StampedImage> frameList);
            virtual void processFrames();
            virtual cv::Mat getCurrentImage();

            virtual QString getName();
            virtual QString getDisplayName();

            //virtual TimeStamp getPCtime();
            //virtual TimeStamp cameraOffsetTime(std::shared_ptr<Lockable<Camera>> cameraPtr);
            virtual void setupNIDAQ(std::shared_ptr <Lockable<NIDAQUtils>> nidaq_task,
                                    bool testConfigEnabled, string trial_info,
                                    std::shared_ptr<TestConfig> testConfig);

        signals:

            void newFrameData(FrameData data); 

        protected:

            unsigned int cameraNumber_;
            unsigned int partnerCameraNumber_ ;
            QString cameraGuidString_;
            bool connectedToPartner_;

            unsigned long numMessageSent_;
            unsigned long numMessageReceived_;

            bool testConfigEnabled_;
            string trial_num_;

            //test
            std::vector<float> time_stamps1;
            std::vector<int64_t> time_stamps2;
            std::vector<std::vector<uInt32>>time_stamps3;
            std::vector<unsigned int> queue_size;
             
            std::shared_ptr<TestConfig>testConfig_;
            std::shared_ptr <Lockable<NIDAQUtils>> nidaq_task_;
            std::shared_ptr<Lockable<GetTime>> gettime_;
            QPointer<ImageLabel> imageLabelPtr_;
            QSharedPointer<QList<QPointer<CameraWindow>>> cameraWindowPtrList_;
            std::shared_ptr<Lockable<Camera>> cameraPtr_;
            
            void connectWidgets();
            void initialize();
            
            unsigned int getPartnerCameraNumber();
            QPointer<CameraWindow> getPartnerCameraWindowPtr();

            void updateMessageLabels();
            
            void allocate_testVec();

        private slots:

            void onNewFrameData(FrameData data);


    };
}
#endif
