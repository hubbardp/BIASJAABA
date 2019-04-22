#ifdef WITH_VIDEO
#include <QString>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>


namespace bias {

    class videoBackend {

        public:

        QString filename;
    
        videoBackend(QString file);

        cv::VideoCapture videoCapObject(videoBackend& vid);    
        void releaseCapObject(cv::VideoCapture& cap); 
   
        cv::Mat getImage(cv::VideoCapture& cap);
        cv::Mat convertImagetoFloat(cv::Mat& img);
        int getImageHeight(cv::VideoCapture& cap);
        int getImageWidth(cv::VideoCapture& cap);
        float getfps(cv::VideoCapture& cap);
        int getNumFrames(cv::VideoCapture& cap);

    };

}

#endif
