#ifndef PROCESS_SCORES_HPP
#define PROCESS_SCORES_HPP

//#include "lockable.hpp"
#include "stamped_image.hpp"
#include "frame_data.hpp"
#include "HOGHOF.hpp"
#include "beh_class.hpp"
#include "bias_plugin.hpp"
#include "video_utils.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include "timer.h"
#include "shape_data.hpp"
#include "jaaba_utils.hpp"

#include <opencv2/opencv.hpp>
#include <QThreadPool>
#include <QRunnable>
#include <QPointer>
#include <QQueue>
#include <memory>

#include "win_time.hpp"

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
           bool mesPass_;
           int scoreCount;
           bool score_calculated_;
           bool skipFront;
           bool skipSide;
          
           bool isProcessed_side;
           bool isProcessed_front; 
           bool isHOGHOFInitialised;
           int processedFrameCount;
           int frameCount_;
           int partner_frameCount_;
           uint64_t side_read_time_, front_read_time_;

           QPointer<HOGHOF> HOGHOF_frame;
           QPointer<HOGHOF> HOGHOF_partner;
           QPointer<beh_class> classifier;

           PredData predScore;
           PredData predScorePartner;
           PredData predScoreFinal;
           QQueue<PredData> frontScoreQueue;
           QQueue<PredData> sideScoreQueue;
           std::vector<int64_t> frame_read_stamps; // frame read pass timings
           std::vector<int64_t> partner_frame_read_stamps; // partner read pass timings
           std::shared_ptr<Lockable<GetTime>> getTime_;
           ProcessScores(QObject *parent, bool mesPass,
                         std::shared_ptr<Lockable<GetTime>> getTime);
          
           void stop();
           void detectOn();
           void detectOff();
            
           videoBackend* vid_sde;
           videoBackend* vid_front;
           cv::VideoCapture capture_sde;
           cv::VideoCapture capture_front;
           
           cv::Mat curr_frame; 
           cv::Mat grey_frame;

           void onProcessSide();
           void onProcessFront();
           void initHOGHOF(QPointer<HOGHOF> hoghof, int img_height, int img_width);
           void genFeatures(QPointer<HOGHOF> hoghof, int frameCount);
           void write_score(std::string file, int framenum, PredData& score);
           void write_histoutput(std::string file,float* out_img, unsigned w, unsigned h,unsigned nbins);


        private :

           bool stopped_;
           bool ready_;
           float threshold_runtime = static_cast<float>(3000);
 
           QQueue<FrameData> frameQueue_;
           QPointer<BiasPlugin> partnerPluginPtr_;
           //QQueue<FrameData> senderImageQueue_;
           //QQueue<FrameData> receiverImageQueue_;

           void run();

        signals:

           void newShapeData(ShapeData data);
           void sideProcess(bool side);
           void frontProcess(bool front);


    };

}

#endif