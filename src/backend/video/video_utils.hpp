#ifndef VIDEO_UTILS_HPP
#define VIDEO_UTILS_HPP

#include <QString>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "basic_types.hpp"


namespace bias {

    class videoBackend {

    public:

        QString filename;
          
        videoBackend();    
        videoBackend(QString file);
        cv::Mat grabImage();
        void convertImagetoFloat(cv::Mat& img);
        int getImageHeight();
        int getImageWidth();
        float getFPS();
        int getNumFrames();
        int getCurrentFrameNumber();
        TimeStamp getImageTimeStamp();
        void setBufferSize();
        std::string type2str(int type);
        void readVidFrames(std::vector<cv::Mat>& vid_frames);
        void preprocess_vidFrame(cv::Mat& cur_Img);
        void releaseCapObject();
        void checkCapOpen();

    private:

        cv::VideoCapture cap_;
        bool isOpen_;
        double dt_;
   
    };

}

#endif