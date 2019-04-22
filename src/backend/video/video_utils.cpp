#ifdef WITH_VIDEO
#include "video_utils.hpp"
#include <opencv2/videoio.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <QDebug>
#include <iostream>
//#include "stampedImage.hpp"


namespace bias {


    videoBackend::videoBackend(QString file) {

        filename = file;   

    }

    cv::VideoCapture videoBackend::videoCapObject(videoBackend& vid) {

        cv::VideoCapture cap(vid.filename.toStdString().c_str());
        return cap;

    }

    void videoBackend::releaseCapObject(cv::VideoCapture& cap) {

        cap.release();

    }
    
    cv::Mat videoBackend::getImage(cv::VideoCapture& cap) {

        cv::Mat frame;
        cv::Mat grey;

        if(!cap.isOpened()) {
 
            printf("File not found"); 
            exit(-1);   
        }
        
        if(!cap.read(frame)) {

            printf("Unable to Read Frame");
            exit(-1);
        }

        // convert the frame to grayscale
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
        return grey;

    }

    cv::Mat videoBackend::convertImagetoFloat(cv::Mat& img) {

        // convert the frame into float32
        img.convertTo(img, CV_32FC1);
        img = img / 255;
        return img;

    }


    int videoBackend::getImageHeight(cv::VideoCapture& cap) {

        return cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    }

    int videoBackend::getImageWidth(cv::VideoCapture& cap) {

        return cap.get(cv::CAP_PROP_FRAME_WIDTH);

    }


    float videoBackend::getfps(cv::VideoCapture& cap) {

        return cap.get(cv::CAP_PROP_FPS);

    }


    int videoBackend::getNumFrames(cv::VideoCapture& cap) {

        return cap.get(cv::CAP_PROP_FRAME_COUNT);

    }

}

#endif
