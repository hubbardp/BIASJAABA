#ifndef SIGNAL_SLOT_DEMO_HPP
#define SIGNAL_SLOT_DEMO_HPP
#include "ui_signal_slot_demo_plugin.h"
#include "bias_plugin.hpp"
#include "frame_data.hpp"
#include <QPointer>
#include <QVector>
#include <QList>


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
                                 GetTime* gettime, QWidget *parentPtr=0);

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
            virtual void setupNIDAQ(std::shared_ptr <Lockable<NIDAQUtils>> nidaq_task);

        signals:

            void newFrameData(FrameData data); 

        protected:

            unsigned int cameraNumber_;
            unsigned int partnerCameraNumber_ ;
            QString cameraGuidString_;
            bool connectedToPartner_;

            unsigned long numMessageSent_;
            unsigned long numMessageReceived_;


            std::vector<uInt32>cam_delay1;
            std::vector<int64_t>cam_delay2;
            std::vector<float>time_lat;
            TimeStamp cam_ofs={0,0};
            
            std::shared_ptr <Lockable<NIDAQUtils>> nidaq_task_;
            GetTime* gettime_;
            QPointer<ImageLabel> imageLabelPtr_;
            QSharedPointer<QList<QPointer<CameraWindow>>> cameraWindowPtrList_;
            std::shared_ptr<Lockable<Camera>> cameraPtr_;
            
            void connectWidgets();
            void initialize();
            
            unsigned int getPartnerCameraNumber();
            QPointer<CameraWindow> getPartnerCameraWindowPtr();

            void updateMessageLabels();

        private slots:

            void onNewFrameData(FrameData data);


    };
}
#endif
