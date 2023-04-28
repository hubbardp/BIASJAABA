#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    cv::VideoWriter writer;
    int codec = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');  // select desired codec (must be available at runtime)
    double fps = 400.0;                          // framerate of the created video stream
    string filename = "C:/Users/27rut/BIAS/build/Release/test.avi";             // name of the output video file
    bool isColor = 0;
    cv::Size size = cv::Size(200, 200);

    bool openOK = writer.open(filename, codec, fps, size, isColor);
    std::cout << "openOK =" << openOK << std::endl;
    while (1);
}