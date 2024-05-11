#ifndef FLYTRACK_CONFIG_HPP
#define FLYTRACK_CONFIG_HPP

#include "rtn_status.hpp"
#include <QVariantMap>
#include <QColor>


namespace bias
{

    const int N_ROI_TYPES = 2;
    enum ROIType { CIRCLE, NONE };

    const int N_FLY_VS_BG_MODES = 3;
    enum FlyVsBgModeType { FLY_DARKER_THAN_BG, FLY_BRIGHTER_THAN_BG, FLY_ANY_DIFFERENCE_BG };

    bool roiTypeToString(ROIType roiType, QString& roiTypeString);
    bool roiTypeFromString(QString roiTypeString, ROIType& roiType);
    bool flyVsBgModeToString(FlyVsBgModeType flyVsBgMode, QString& flyVsBgModeString);
    bool flyVsBgModeFromString(QString flyVsBgModeString, FlyVsBgModeType& flyVsBgMode);

    class FlyTrackConfig
    {

        public:

            // default parameters
            static const QString DEFAULT_BG_VIDEO_FILE_PATH; // video to estimate background from
            static const QString DEFAULT_BG_IMAGE_FILE_PATH; // saved background median estimate
            static const QString DEFAULT_TMP_OUT_DIR; // temporary output directory

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
            QString bgVideoFilePath; // video to estimate background from
            QString bgImageFilePath; // saved background median estimate
            QString tmpOutDir; // temporary output directory
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
            void setBgVideoFilePath(QString bgVideoFilePath);

            RtnStatus setBgEstFromMap(QVariantMap configMap);
            RtnStatus setRoiFromMap(QVariantMap configMap);
            RtnStatus setBgSubFromMap(QVariantMap configMap);
            RtnStatus setHeadTailFromMap(QVariantMap configMap);
            RtnStatus setMiscFromMap(QVariantMap configMap);
            QVariantMap toMap();
            RtnStatus fromMap(QVariantMap configMap);
            RtnStatus fromJson(QByteArray jsonConfigArray);
            QByteArray toJson();
            QString toString();

            void print();
            void setRoiFracAbsFromMap(QVariantMap configMap, RtnStatus& rtnStatus,
                QString fracField, QString absField, double& fracParam, double& absParam, int imgSize);


        protected:
            bool roiLocationSet_; // whether the absolute ROI location has been set
            double roiCenterXFrac_; // x-coordinate of ROI center, relative to image width
            double roiCenterYFrac_; // y-coordinate of ROI center, relative to image height
            double roiRadiusFrac_; // radius of ROI, relative to min(image width, image height)
            int imageWidth_; // image width
            int imageHeight_; // image height

    };
}

#endif