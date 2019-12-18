#ifndef PROCESS_SCORES_HPP
#define PROCESS_SCORES_HPP

#include "lockable.hpp"
#include "stamped_image.hpp"
#include "frame_data.hpp"
#include "HOGHOF.hpp"
#include "beh_class.hpp"
#include <opencv2/opencv.hpp>
#include <QThreadPool>
#include <QRunnable>
#include <QPointer>
#include <QQueue>
#include <memory>
#include "bias_plugin.hpp"
#include "video_utils.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include "timer.h"

#include "shape_data.hpp"

namespace bias 
{

    class HOGHOF;
    class beh_class;

    class ProcessScores : public QObject, public QRunnable, public Lockable<Empty>
    {

       Q_OBJECT

       public :

           bool save;
           bool detectStarted_;
           bool isSide;
           bool isFront;
           bool processSide;
           bool processFront;
          
           bool isProcessed_side;
           bool isProcessed_front; 
           bool isHOGHOFInitialised;
           int processedFrameCount;
           //QPointer<HOGHOF> HOGHOF_side;
           //QPointer<HOGHOF> HOGHOF_front;
           QPointer<HOGHOF> HOGHOF_frame;
           QPointer<HOGHOF> HOGHOF_partner;
           //QPointer<beh_class> classifier; 

           ProcessScores(QObject *parent=0);
           void stop();
           //void enqueueFrameDataSender(FrameData frameData);
           //void enqueueFrameDataReceiver(FrameData frameData);
           void enqueueFrameData(FrameData frameData);
           void detectOn();
           void detectOff();
            
           videoBackend* vid_sde;
           videoBackend* vid_front;
           cv::VideoCapture capture_sde;
           cv::VideoCapture capture_front;
           
           cv::Mat curr_frame; 
           cv::Mat grey_frame;

           QWaitCondition wait_to_process_;
           QWaitCondition signal_to_process_;
           QMutex mutex_;

         
           void initHOGHOF(QPointer<HOGHOF> hoghof, int img_height, int img_width);
           void genFeatures(QPointer<HOGHOF> hoghof, int frameCount);
           void write_score(std::string file, int framenum, float score);
           void write_histoutput(std::string file,float* out_img, unsigned w, unsigned h,unsigned nbins);
           void write_time(std::string file, int framenum, std::vector<float> timeVec);


       private :

           bool stopped_;
           bool ready_;
 
           QQueue<FrameData> frameQueue_;
           QPointer<BiasPlugin> partnerPluginPtr_;
           //QQueue<FrameData> senderImageQueue_;
           //QQueue<FrameData> receiverImageQueue_;

           void run();
           //void initHOGHOF(QPointer<HOGHOF> hoghof, int img_height, int img_width);
           //void genFeatures(QPointer<HOGHOF> hoghof, int frameCount);
           //void write_histoutput(std::string file,float* out_img, unsigned w, unsigned h,unsigned nbins);

       signals:

           void newShapeData(ShapeData data);
           void sideProcess(bool side);
           void frontProcess(bool front);
        
       private slots:

            //void onNewShapeData(ShapeData data);




    };

}

#endif


















































