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
           bool isSide;
           bool isFront;
           bool isHOGHOFInitialised;
           //QPointer<HOGHOF> HOGHOF_side;
           //QPointer<HOGHOF> HOGHOF_front;
           QPointer<HOGHOF> HOGHOF_frame;
           QPointer<HOGHOF> HOGHOF_partner;
           QPointer<beh_class> classifier; 

           ProcessScores(QObject *parent=0);
           void stop();
           //void enqueueFrameDataSender(FrameData frameData);
           //void enqueueFrameDataReceiver(FrameData frameData);
           void enqueueFrameData(FrameData frameData);
           void detectOn();
           void detectOff();
            
           videoBackend* vid_;
           cv::VideoCapture capture_;
           cv::Mat curr_frame; 
           cv::Mat grey_frame;
          
           void write_score(std::string file, int framenum, float score);

       private :

           bool stopped_;
           bool ready_;
           bool detectStarted_;
           int frameCount;
 
           QQueue<FrameData> frameQueue_;
           QPointer<BiasPlugin> partnerPluginPtr_;
           //QQueue<FrameData> senderImageQueue_;
           //QQueue<FrameData> receiverImageQueue_;

           void run();
           void initHOGHOF(QPointer<HOGHOF> hoghof, int img_height, int img_width);
           void genFeatures(QPointer<HOGHOF> hoghof, int frameCount);

           void write_histoutput(std::string file,float* out_img, unsigned w, unsigned h,unsigned nbins);

        signals:

            void newShapeData(ShapeData data); 
       
        private slots:

            //void onNewShapeData(ShapeData data);




    };

}

#endif


















































