#ifndef JAABA_PLUGIN_WINDOW_HPP

#define JAABA_PLUGIN_WINDOW_HPP

#include "ui_jaaba_plugin.h"
#include "bias_plugin.hpp"
#include "rtn_status.hpp"
#include "frame_data.hpp"
#include "HOGHOF.hpp"
#include "beh_class.hpp"
#include "process_scores.hpp"
#include "shape_data.hpp"
#include <QMainWindow>
#include <QPointer>
#include <QThreadPool>
#include <QThread>

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
            int frameCount = 0;

            JaabaPlugin(int numberOfCameras,QWidget *parent=0);

            virtual void finalSetup();
            virtual QString getName();
            virtual QString getDisplayName();
            virtual void processFrames(QList<StampedImage> frameList);
            virtual void reset();
            virtual void stop();
            cv::Mat getCurrentImage();
            virtual QObject getObject();

        protected:
  
            unsigned int cameraNumber_;
            unsigned int partnerCameraNumber_ ;
 
            QPointer<ProcessScores> processScoresPtr_;
            QPointer<QThreadPool> threadPoolPtr_;

            QQueue<FrameData> sendImageQueue;
            QQueue<FrameData> receiveImageQueue;

            QSharedPointer<QList<QPointer<CameraWindow>>> cameraWindowPtrList_;
            QPointer<CameraWindow> getPartnerCameraWindowPtr();
            unsigned int getPartnerCameraNumber();

        private:

            int nviews_;
            bool save;
            bool stop_save;
            bool detectStarted = false;
            
            unsigned long numMessageSent_;
            unsigned long numMessageReceived_;
         
            videoBackend* vid_sde;
            videoBackend* vid_front;
            cv::VideoCapture capture_sde;
            cv::VideoCapture capture_front;
        
            bool pluginReady();
            bool isSender();
            bool isReceiver(); 
            void initialize();
            void setupHOGHOF();
            void setupClassifier();
            void connectWidgets();
            int getNumberofViews();
            void updateWidgetsOnLoad();
            //void checkviews();
            void detectEnabled();

 
        signals:

            void newFrameData(FrameData data);
            void newShapeData(ShapeData data);

        private slots:

            void SideViewCheckBoxChanged(int state);
            void FrontViewCheckBoxChanged(int state);
            void reloadButtonPressed();
            void detectClicked();
            void saveClicked();
            void onNewFrameData(FrameData data);
            void onNewShapeData(ShapeData data);
    
    };

}
#endif
