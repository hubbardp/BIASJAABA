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
            int lastProcessedFrameCount = 0;
            int frameCount=0;

            JaabaPlugin(int numberOfCameras,QWidget *parent=0);

            virtual void finalSetup();
            virtual QString getName();
            virtual QString getDisplayName();
            //virtual void processFrames(QList<StampedImage> frameList);
            virtual void processFrames();
            virtual void reset();
            virtual void stop();
            cv::Mat getCurrentImage();

        protected:
  
            unsigned int cameraNumber_;
            unsigned int partnerCameraNumber_ ;
 
            QPointer<ProcessScores> processScoresPtr_;
            QPointer<QThreadPool> threadPoolPtr_;

            QQueue<FrameData> sendImageQueue;
            QQueue<FrameData> receiveImageQueue;

            QSharedPointer<QList<QPointer<CameraWindow>>> cameraWindowPtrList_;
            QPointer<CameraWindow> getPartnerCameraWindowPtr();
            std::shared_ptr<LockableQueue<StampedImage>> partnerPluginImageQueuePtr_;
            unsigned int getPartnerCameraNumber();

        private:

            int nviews_;
            bool save;
            bool stop_save;
            bool detectStarted = false;
            
            unsigned long numMessageSent_;
            unsigned long numMessageReceived_;
          
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

            // Test
            std::vector<float>gpuSide;
            std::vector<float>gpuFront;
            std::vector<float>gpuOverall;
            void write_output(std::string file,float* out_img, unsigned w, unsigned h);

 
        signals:

            void newFrameData(FrameData data);
            void newShapeData(ShapeData data);
            void partnerImageQueue(std::shared_ptr<LockableQueue<StampedImage>> partnerPluginImageQueuePtr);

        private slots:

            void SideViewCheckBoxChanged(int state);
            void FrontViewCheckBoxChanged(int state);
            void reloadButtonPressed();
            void detectClicked();
            void saveClicked();
            void onNewFrameData(FrameData data);
            void onNewShapeData(ShapeData data);
            void onPartnerPlugin(std::shared_ptr<LockableQueue<StampedImage>> partnerPluginImageQueuePtr);
    
    };

}
#endif
