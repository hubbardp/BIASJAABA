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
#include "video_utils.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

namespace bias 
{

    class HOGHOF;
    class beh_class;

    class ProcessScores : public QObject, public QRunnable, public Lockable<Empty>
    {

       Q_OBJECT

       public :

           bool save;
           QPointer<HOGHOF> HOGHOF_side;
           QPointer<HOGHOF> HOGHOF_front;
           QPointer<beh_class> classifier; 

           ProcessScores(QObject *parent=0);
           void stop();
           void enqueueFrameDataSender(FrameData frameData);
           void enqueueFrameDataReceiver(FrameData frameData);
           void detectOn();
           void detectOff();
            
           videoBackend* vid_sde;
           videoBackend* vid_frt;
           cv::VideoCapture capture_sde;
           cv::VideoCapture capture_frt;
           cv::Mat curr_side; 
           cv::Mat curr_front;
           cv::Mat grey_sde;
           cv::Mat grey_frt;

       private :

           bool stopped_;
           bool ready_;
           bool detectStarted_;
           int frameCount;
 
           QQueue<FrameData> senderImageQueue_;
           QQueue<FrameData> receiverImageQueue_;

           void run();
           void initHOGHOF(QPointer<HOGHOF> hoghof, int img_height, int img_width);
           void genFeatures(QPointer<HOGHOF> hoghof, int frameCount);

           void write_score(std::string file, int framenum, float score);
           void write_histoutput(std::string file,float* out_img, unsigned w, unsigned h,unsigned nbins);


    };

}

#endif


















































