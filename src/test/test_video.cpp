#include "video_utils.hpp"
#include <opencv2/highgui/highgui.hpp>

// Temporary includes
#include <QDebug>
#include <iostream>

using namespace bias;

int main(int argc, char *argv[]) {

    
    QString vidFile = "/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_sde.avi" ;
    videoBackend vid(vidFile) ;
    cv::VideoCapture capture = vid.videoCapObject(vid);
    cv::Mat img = vid.getImage(capture);
     
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", img );                   // Sh
    cv::waitKey(0);
    qDebug() << "sdfs " << vid.filename;

    vid.convertImagetoFloat(img);    
    vid.releaseCapObject(capture);
    return 0;

}
