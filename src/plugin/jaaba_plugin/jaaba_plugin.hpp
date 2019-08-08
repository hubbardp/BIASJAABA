#ifndef JAABA_PLUGIN_WINDOW_HPP
#define JAABA_PLUGIN_WINDOW_HPP

#include "ui_jaaba_plugin.h"
#include "bias_plugin.hpp"
#include "rtn_status.hpp"
#include "frame_data.hpp"
#include "HOGHOF.hpp"
#include "beh_class.hpp"
#include <QMainWindow>
#include <QPointer>
#include <QThreadPool>
#include <QThread>
#include <QSharedMemory>
//#include <QBuffer>
//test development
#include <fstream>
#include <string>
#include "H5Cpp.h"
#include "video_utils.hpp"
#include <opencv2/highgui/highgui.hpp>

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
            cv::Mat getCurrentImage();

        protected:
  
            unsigned int cameraNumber_;
            unsigned int partnerCameraNumber_ ;
            //FrameData frameData;
 
            QPointer<HOGHOF> HOGHOF_side;
            QPointer<HOGHOF> HOGHOF_front;
            QPointer<beh_class> classifier;

            QSharedPointer<QList<QPointer<CameraWindow>>> cameraWindowPtrList_;
            QPointer<CameraWindow> getPartnerCameraWindowPtr();
            unsigned int getPartnerCameraNumber();

        private:

            int nviews_;
            bool detectStarted;
            bool save;
            bool stop_save;
            
            unsigned long numMessageSent_;
            unsigned long numMessageReceived_;
            FrameData frameData;
            LockableQueue<FrameData> senderImageQueue_;
            LockableQueue<StampedImage> receiverImageQueue_;
         
            //HOGShape tmp_sideshape;
            //HOGShape tmp_frontshape;
            videoBackend* vid_sde;
            videoBackend* vid_front;
            cv::VideoCapture capture_sde;
            cv::VideoCapture capture_front;
        
            bool pluginReady();
            bool isSender();
            bool isReceiver(); 
            void initialize();
            void initHOGHOF(QPointer<HOGHOF> hoghof);
            void genFeatures(QPointer<HOGHOF> hoghof, int frameCount);
            void setupHOGHOF();
            void setupClassifier();
            void connectWidgets();
            int getNumberofViews();
            void updateWidgetsOnLoad();
            void checkviews();
            void detectEnabled();
            void startProcessthread();

            //test development
            void copy_features1d(int frame_num, int num_elements, 
                            std::vector<float> &vec_feat, float* array_feat);
            int createh5(std::string filename, int frame_num,
                          int num_frames, int hog_elements, int hof_elements,
                          std::vector<float> hog, std::vector<float> hof);
            void create_dataset(H5::H5File& file, std::string key,
                          std::vector<float> features, int num_frames, int num_elements);
            void stop();
            //void write_output(std::string file,float* out_img, unsigned w, unsigned h);
            void write_output_shape(std::string filename, std::string view, unsigned x, unsigned y, unsigned bin);
            void write_histoutput(std::string file,float* out_img, unsigned w, unsigned h,unsigned nbins);
            void read_output_shape(std::string filename);      
            void write_score(std::string file, int frame, float score);
            void read_score(std::string file_side, std::string file_front, int framenum);    
 
        signals:

            void newFrameData(FrameData data);

        private slots:

            void SideViewCheckBoxChanged(int state);
            void FrontViewCheckBoxChanged(int state);
            void reloadButtonPressed();
            void detectClicked();
            void saveClicked();
            void onNewFrameData(FrameData data);
            void processData();

    
    };

}
#endif
