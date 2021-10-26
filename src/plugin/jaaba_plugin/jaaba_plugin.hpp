#ifndef JAABA_PLUGIN_WINDOW_HPP
#define JAABA_PLUGIN_WINDOW_HPP

#include "ui_jaaba_plugin.h"
#include "camera_window.hpp"
//#include "bias_plugin.hpp"

#include "rtn_status.hpp"
#include "frame_data.hpp"
#include "camera_facade.hpp"
#include "HOGHOF.hpp"
#include "beh_class.hpp"
#include "process_scores.hpp"
#include "vis_plots.hpp"
#include "image_label.hpp"
#include "shape_data.hpp"

#include <QMainWindow>
#include <QMessageBox>
#include <QPointer>
#include <QThreadPool>
#include <QThread>

#include<ctime>
#include "win_time.hpp"


#ifdef WITH_SPIN
#include "camera_device_spin.hpp"
#endif

//test development
//#include <fstream>
//#include <string>
//#include "H5Cpp.h"
//#include "video_utils.hpp"
//#include <opencv2/highgui/highgui.hpp>

namespace bias
{
    class HOGHOF;
    class beh_class;
    class CameraWindow;

    class JaabaPlugin : public BiasPlugin, public Ui::JaabaPluginDialog
    {
        Q_OBJECT

        public:

            static const QString PLUGIN_NAME;
            static const QString PLUGIN_DISPLAY_NAME;
            int lastProcessedFrameCount = 0;

            JaabaPlugin(int numberOfCameras, QPointer<QThreadPool> threadPoolPtr,
                std::shared_ptr<Lockable<GetTime>> gettime, QWidget *parent=0);

            void resetTrigger();

            virtual void finalSetup();
            virtual QString getName();
            virtual QString getDisplayName();
            //virtual void processFrames(QList<StampedImage> frameList);
            virtual void processFrames();
            virtual void reset();
            virtual void stop();
            cv::Mat getCurrentImage();
            int getLaserTrigger();
            virtual void setupNIDAQ(std::shared_ptr <Lockable<NIDAQUtils>> nidaq_task,
                                    bool testConfigEnabled, string trial_info,
                                    std::shared_ptr<TestConfig> testConfig);

        protected:
 
            bool laserOn; 
            unsigned int cameraNumber_;
            unsigned int partnerCameraNumber_;
 
            QPointer<ProcessScores> processScoresPtr_side;
            QPointer<ProcessScores> processScoresPtr_front;
            QPointer<beh_class> classifier;
            QPointer<VisPlots> visplots;
            QPointer<QThreadPool> threadPoolPtr_;
            std::shared_ptr<Lockable<GetTime>> gettime_;
            QThread thread_vis;
            std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task_;
            std::shared_ptr<TestConfig>testConfig_;

            QQueue<FrameData> sendImageQueue;
            QQueue<FrameData> receiveImageQueue;

            QSharedPointer<QList<QPointer<CameraWindow>>> cameraWindowPtrList_;
            std::shared_ptr<Lockable<Camera>> cameraPtr_;
            QPointer<CameraWindow> getPartnerCameraWindowPtr();
            std::shared_ptr<LockableQueue<StampedImage>> partnerPluginImageQueuePtr_;
            unsigned int getPartnerCameraNumber();

            void updateTrigStateInfo();
            RtnStatus connectTriggerDev();

            void allocate_testVec();

        private:

            int nviews_;
            int nDevices_;
            bool save;
            bool stop_save;
            bool detectStarted = false;
            unsigned int numskippedFrames_=0;
            int image_height=0;
            int image_width=0;
            float threshold_runtime = static_cast<float>(3000);
            double tStamp=0.0;
            int64_t process_time=0;
            TimeStamp cam_ofs={0,0};
           
 
            unsigned long numMessageSent_;
            unsigned long numMessageReceived_;
                         
            // Trigger parameters
            bool triggerEnabled;
            bool triggerArmedState;

            bool testConfigEnabled_;
            string trial_num_;

            //test
            std::vector<float> time_stamps1;
            std::vector<int64_t> time_stamps2;
            std::vector<std::vector<uInt32>>time_stamps3;
            std::vector<unsigned int> queue_size;

            std::vector<float>classifier_score;
            std::vector<float>laserRead = { 0,0,0,0,0,0 };
            std::vector<double>timeofs;
            std::vector<double>timestd;
          
            bool pluginReady();
            bool isSender();
            bool isReceiver(); 
            void triggerLaser();   
            int getNumberofViews();
            int getNumberOfDevices();
            void updateWidgetsOnLoad();
            void initialize();
            void setupHOGHOF();
            void setupClassifier();
            void connectWidgets();
            void detectEnabled();
            double convertTimeStampToDouble(TimeStamp curr, TimeStamp init);
            void getFormatSettings();            
            void gpuInit();
            //void cameraOffsetTime();
            //TimeStamp getPCtime();
 
        signals:

            void partnerImageQueue(std::shared_ptr<LockableQueue<StampedImage>> partnerPluginImageQueuePtr);
            void processSide(bool side);
            void processFront(bool front);

        private slots:

            void SideViewCheckBoxChanged(int state);
            void FrontViewCheckBoxChanged(int state);
            void reloadButtonPressed();
            void detectClicked();
            void saveClicked();
            void onPartnerPlugin(std::shared_ptr<LockableQueue<StampedImage>> partnerPluginImageQueuePtr);
            void trigResetPushButtonClicked();
            void trigEnabledCheckBoxStateChanged(int state);


    };

}
#endif
