#ifndef FLYTRACK_CONFIG_HPP
#define FLYTRACK_CONFIG_HPP

#include "rtn_status.hpp"
#include <QVariantMap>
#include <QColor>


namespace bias
{

    enum ROIType { CIRCLE, NONE };

    enum FlyVsBgModeType { FLY_DARKER_THAN_BG, FLY_BRIGHTER_THAN_BG, FLY_ANY_DIFFERENCE_BG };

    class FlyTrackConfig
    {

    public:

        // default parameters
        static const int DEFAULT_BACKGROUND_THRESHOLD; // foreground/background threshold, between 0 and 255
        static const int DEFAULT_N_FRAMES_BG_EST; // number of frames used for background estimation, set to 0 to use all frames
        static const int DEFAULT_LAST_FRAME_SAMPLE; // last frame sampled for background estimation, set to 0 to use last frame of video
        static const FlyVsBgModeType DEFAULT_FLY_VS_BG_MODE; // whether the fly is darker than the background
        static const ROIType DEFAULT_ROI_TYPE; // type of ROI
        static const double DEFAULT_ROI_CENTER_X_FRAC; // x-coordinate of ROI center, relative
        static const double DEFAULT_ROI_CENTER_Y_FRAC; // y-coordinate of ROI center, relative
        static const double DEFAULT_ROI_RADIUS_FRAC; // radius of ROI, relative
        static const int DEFAULT_HISTORY_BUFFER_LENGTH; // number of frames to buffer velocity, orientation
        static const double DEFAULT_MIN_VELOCITY_MAGNITUDE; // minimum velocity magnitude in pixels/frame to consider fly moving
        static const double DEFAULT_HEAD_TAIL_WEIGHT_VELOCITY; // weight of velocity dot product in head-tail orientation resolution
        static const double DEFAULT_MIN_VEL_MATCH_DOTPROD; // minimum dot product for velocity matching
        static const bool DEFAULT_DEBUG; // flag for debugging


        // parameters
        int backgroundThreshold; // foreground threshold
        int nFramesBgEst; // number of frames used for background estimation, set to 0 to use all frames
        int lastFrameSample; // last frame sampled for background estimation, set to 0 to use last frame of video
        FlyVsBgModeType flyVsBgMode; // whether the fly is darker than the background
        ROIType roiType; // type of ROI
        double roiCenterX; // x-coordinate of ROI center
        double roiCenterY; // y-coordinate of ROI center
        double roiRadius; // radius of ROI
        int historyBufferLength; // number of frames to buffer velocity, orientation
        double minVelocityMagnitude; // minimum velocity magnitude in pixels/frame to consider fly moving
        double headTailWeightVelocity; // weight of velocity dot product in head-tail orientation resolution
        bool DEBUG; // flag for debugging

        FlyTrackConfig();
        void setImageSize(int width, int height);

        void print();

    protected:
        double roiCenterXFrac_; // x-coordinate of ROI center, relative to image width
        double roiCenterYFrac_; // y-coordinate of ROI center, relative to image height
        double roiRadiusFrac_; // radius of ROI, relative to min(image width, image height)
        int imageWidth_; // image width
        int imageHeight_; // image height

    };
}

#endif