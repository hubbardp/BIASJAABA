#ifndef BIAS_STAMPED_IMAGE_HPP
#define BIAS_STAMPED_IMAGE_HPP 

#include <opencv2/core/core.hpp>
#include "basic_types.hpp"

namespace bias
{
    struct StampedImage
    {
        cv::Mat image;
        double timeStamp;
        TimeStamp timeStampInit;
        TimeStamp timeStampVal;
        double dtEstimate;
        unsigned long frameCount;
    };

}

#endif // #ifndef BIAS_STAMPED_IMAGE_HPP
