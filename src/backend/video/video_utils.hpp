#ifdef WITH_VIDEO
#ifndef VIDEO_UTILS_HPP
#define VIDEO_UTILS_HPP

#include <QString>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>


namespace bias {

    class videoBackend {

        public:

        QString filename;
          
        videoBackend();    
        videoBackend(QString file);

        cv::VideoCapture videoCapObject();    
        void releaseCapObject(cv::VideoCapture& cap); 
   
        cv::Mat getImage(cv::VideoCapture& cap);
        void convertImagetoFloat(cv::Mat& img);
        int getImageHeight(cv::VideoCapture& cap);
        int getImageWidth(cv::VideoCapture& cap);
        float getfps(cv::VideoCapture& cap);
        int getNumFrames(cv::VideoCapture& cap);
        int getcurrentFrameNumber(cv::VideoCapture& cap);
        void setBufferSize(cv::VideoCapture& cap);
        std::string type2str(int type);
        void readVidFrames(cv::VideoCapture& capture, std::vector<cv::Mat>& vid_frames);
        void preprocess_vidFrame(cv::Mat& cur_Img);
    };

}

#endif
#endif
