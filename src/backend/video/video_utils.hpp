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

    };

}

#endif
