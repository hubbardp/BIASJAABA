#include "flytrack_config.hpp"
#include "json.hpp"
#include <iostream>
#include <QMessageBox>
#include <QtDebug>

namespace bias
{
    const int FlyTrackConfig::DEFAULT_BACKGROUND_THRESHOLD = 75; // foreground/background threshold, between 0 and 255
    const int FlyTrackConfig::DEFAULT_N_FRAMES_BG_EST = 100; // number of frames used for background estimation, set to 0 to use all frames
    const int FlyTrackConfig::DEFAULT_LAST_FRAME_SAMPLE = 0; // last frame sampled for background estimation, set to 0 to use last frame of video
    const FlyVsBgModeType FlyTrackConfig::DEFAULT_FLY_VS_BG_MODE = FLY_DARKER_THAN_BG; // whether the fly is darker than the background
    const ROIType FlyTrackConfig::DEFAULT_ROI_TYPE = CIRCLE; // type of ROI
    const double FlyTrackConfig::DEFAULT_ROI_CENTER_X_FRAC = 0.5; // x-coordinate of ROI center, relative
    const double FlyTrackConfig::DEFAULT_ROI_CENTER_Y_FRAC = 0.5; // y-coordinate of ROI center, relative
    const double FlyTrackConfig::DEFAULT_ROI_RADIUS_FRAC = 0.475; // radius of ROI, relative
    const int FlyTrackConfig::DEFAULT_HISTORY_BUFFER_LENGTH = 5; // number of frames to buffer velocity, orientation
    const double FlyTrackConfig::DEFAULT_MIN_VELOCITY_MAGNITUDE = 1.0; // minimum velocity magnitude in pixels/frame to consider fly moving
    const double FlyTrackConfig::DEFAULT_HEAD_TAIL_WEIGHT_VELOCITY = 3.0; // weight of velocity dot product in head-tail orientation resolution
    const double FlyTrackConfig::DEFAULT_MIN_VEL_MATCH_DOTPROD = 0.25; // minimum dot product for velocity matching
    const bool FlyTrackConfig::DEFAULT_DEBUG = false; // flag for debugging

    FlyTrackConfig::FlyTrackConfig()
    {
		backgroundThreshold = DEFAULT_BACKGROUND_THRESHOLD;
		nFramesBgEst = DEFAULT_N_FRAMES_BG_EST;
		lastFrameSample = DEFAULT_LAST_FRAME_SAMPLE;
		flyVsBgMode = DEFAULT_FLY_VS_BG_MODE;
		roiType = DEFAULT_ROI_TYPE;
		roiCenterXFrac_ = DEFAULT_ROI_CENTER_X_FRAC;
		roiCenterYFrac_ = DEFAULT_ROI_CENTER_Y_FRAC;
		roiRadiusFrac_ = DEFAULT_ROI_RADIUS_FRAC;
		historyBufferLength = DEFAULT_HISTORY_BUFFER_LENGTH;
		minVelocityMagnitude = DEFAULT_MIN_VELOCITY_MAGNITUDE;
		headTailWeightVelocity = DEFAULT_HEAD_TAIL_WEIGHT_VELOCITY;
		DEBUG = DEFAULT_DEBUG;
		roiCenterX = 0;
		roiCenterY = 0;
		roiRadius = 0;
		imageWidth_ = -1;
		imageHeight_ = -1;
	}

	void FlyTrackConfig::setImageSize(int width, int height)
	{
		imageWidth_ = width;
		imageHeight_ = height;
		roiCenterX = imageWidth_*roiCenterXFrac_;
		roiCenterY = imageHeight_*roiCenterYFrac_;
		roiRadius = std::min(imageWidth_, imageHeight_)*roiRadiusFrac_;
	}

    void FlyTrackConfig::print() {
		std::cout << "backgroundThreshold: " << backgroundThreshold << std::endl;
		std::cout << "nFramesBgEst: " << nFramesBgEst << std::endl;
		std::cout << "lastFrameSample: " << lastFrameSample << std::endl;
		std::cout << "flyVsBgMode: " << flyVsBgMode << std::endl;
		std::cout << "roiType: " << roiType << std::endl;
		std::cout << "roiCenterX: " << roiCenterX << std::endl;
		std::cout << "roiCenterY: " << roiCenterY << std::endl;
		std::cout << "roiRadius: " << roiRadius << std::endl;
		std::cout << "historyBufferLength: " << historyBufferLength << std::endl;
		std::cout << "minVelocityMagnitude: " << minVelocityMagnitude << std::endl;
		std::cout << "headTailWeightVelocity: " << headTailWeightVelocity << std::endl;
		std::cout << "DEBUG: " << DEBUG << std::endl;
    }

}