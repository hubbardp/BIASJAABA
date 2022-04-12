#ifndef BIAS_STAMPED_IMAGE_HPP
#define BIAS_STAMPED_IMAGE_HPP 

#include <opencv2/core/core.hpp>
#include "basic_types.hpp"

namespace bias
{
    class StampedImage
    {

	public:

        cv::Mat image;
        double timeStamp;
        TimeStamp timeStampInit;
        TimeStamp timeStampVal;
        double dtEstimate;
        unsigned long frameCount;
        bool isSpike;
        float fstfrmtStampRef;
    };

}

#endif // #ifndef BIAS_STAMPED_IMAGE_HPP
