#ifdef WITH_VIDEO
#include "video_utils.hpp"
#include <opencv2/videoio.hpp>
#include <iostream>
//#include "stampedImage.hpp"


namespace bias {


    //videoBackend::videoBackend() {}

    videoBackend::videoBackend(QString file) {

        filename = file;   

    }


    cv::VideoCapture videoBackend::videoCapObject() {

        cv::VideoCapture cap(this->filename.toStdString().c_str());
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
        
        cap.read(frame);

        if (frame.empty())
            return frame;

        // convert the frame to grayscale
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
        return grey; 

    }

    void videoBackend::convertImagetoFloat(cv::Mat& img) {

        // convert the frame into float32
        img.convertTo(img, CV_32FC1);
        img = img / 255;

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

    int videoBackend::getcurrentFrameNumber(cv::VideoCapture& cap) {

        return cap.get(cv::CAP_PROP_POS_FRAMES);
    }

    void videoBackend::setBufferSize(cv::VideoCapture& cap) {

        cap.set(cv::CAP_PROP_BUFFERSIZE, 3);

    }

    std::string videoBackend::type2str(int type) {

        std::string r;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);

        switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
        }

        r += "C";
        r += (chans + '0');

        return r;
    }

    void videoBackend::readVidFrames(cv::VideoCapture& capture, std::vector<cv::Mat>& vid_frames) {
        
        int num_frames = getNumFrames(capture);
        int height = getImageHeight(capture);
        int width = getImageWidth(capture);

        vid_frames.resize(num_frames, cv::Mat());

        for (int frm_id = 0; frm_id < num_frames; frm_id++)
        {
            if (capture.isOpened())
                vid_frames[frm_id] = getImage(capture);
            else
                assert("Video capture is closed");
        }
        std::cout << "Finished reading movie" << std::endl;
    }

    void videoBackend::preprocess_vidFrame(cv::Mat& cur_Img)
    {
        cv::Mat greyImg;
        convertImagetoFloat(cur_Img);
        
    }

}

#endif
