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

#ifdef linux
#include <sys/time.h>
#endif

#ifdef WIN32
#include "win_time.hpp"
#endif


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

            JaabaPlugin(int numberOfCameras,QWidget *parent=0);

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

        protected:
 
            bool laserOn; 
            unsigned int cameraNumber_;
            unsigned int partnerCameraNumber_;
 
            QPointer<ProcessScores> processScoresPtr_side;
            QPointer<ProcessScores> processScoresPtr_front;
            QPointer<beh_class> classifier;
            QPointer<VisPlots> visplots;
            QPointer<QThreadPool> threadPoolPtr;
            QThread thread_vis;

            QQueue<FrameData> sendImageQueue;
            QQueue<FrameData> receiveImageQueue;

            QSharedPointer<QList<QPointer<CameraWindow>>> cameraWindowPtrList_;
            std::shared_ptr<Lockable<Camera>> cameraPtr_;
            QPointer<CameraWindow> getPartnerCameraWindowPtr();
            std::shared_ptr<LockableQueue<StampedImage>> partnerPluginImageQueuePtr_;
            unsigned int getPartnerCameraNumber();

            void updateTrigStateInfo();
            RtnStatus connectTriggerDev();

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
            TimeStamp cam_ofs={0,0};
           
 
            unsigned long numMessageSent_;
            unsigned long numMessageReceived_;

                         
            // Trigger parameters
            bool triggerEnabled;
            bool triggerArmedState;


            std::vector<float>classifier_score;
            std::vector<float>laserRead = {0,0,0,0,0,0};
            std::vector<double>timeofs;
            std::vector<double>timestd;
			//DEVEL
            std::vector<int64_t> time_seconds;
            std::vector<int64_t> time_useconds;
            std::vector<int64_t> tcam0;
            std::vector<int64_t> tcam1;

          
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
           
            // Test
            std::vector<double>gpuSide;
            std::vector<double>gpuFront;
            std::vector<double>gpuOverall;
            std::vector<double>timediff;
            std::vector<double>timeStamp1; 
            void write_output(std::string file,float* out_img, unsigned w, unsigned h);

 
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
