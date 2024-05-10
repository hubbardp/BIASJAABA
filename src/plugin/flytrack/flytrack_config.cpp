#include "flytrack_config.hpp"
#include "json.hpp"
#include <iostream>
#include <QMessageBox>
#include <QtDebug>

namespace bias
{

    // helper functions

    void setRoiFracAbsFromMap(QVariantMap configMap, RtnStatus& rtnStatus,
        QString fracField, QString absField, double& fracParam, double& absParam, int imgSize) {
        bool roiFracSet = false;
        bool roiAbsSet = false;
        if (configMap.contains(fracField)) {
            if (configMap[fracField].canConvert<double>()) {
                fracParam = configMap[fracField].toDouble();
                roiFracSet = true;
            }
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage(QString("unable to convert %1 to double").arg(fracField));
            }
        }
        if (configMap.contains(absField)) {
            if (configMap[absField].canConvert<double>()) {
                absParam = configMap[absField].toDouble();
                roiAbsSet = true;
            }
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage(QString("unable to convert %1 to double").arg(absField));
            }
        }
        if (roiFracSet && !roiAbsSet && imgSize > 0) {
            absField = (double)imgSize * fracParam;
        }
    }

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

    RtnStatus FlyTrackConfig::fromMap(QVariantMap configMap) {
        RtnStatus rtnStatus;
        rtnStatus.success = true;

        QVariantMap oldConfigMap = toMap();
        RtnStatus rtnStatusBgEst = setBgEstFromMap(configMap["bgEst"].toMap());
        RtnStatus rtnStatusRoi = setRoiFromMap(configMap["roi"].toMap());
        RtnStatus rtnStatusBgSub = setBgSubFromMap(configMap["bgSub"].toMap());
        RtnStatus rtnStatusHeadTail = setHeadTailFromMap(configMap["headTail"].toMap());
        RtnStatus rtnStatusMisc = setMiscFromMap(configMap["misc"].toMap());

        rtnStatus.success = rtnStatusBgEst.success && rtnStatusRoi.success && rtnStatusBgSub.success && rtnStatusHeadTail.success;
        rtnStatus.message += rtnStatusBgEst.message + QString(", ");
        rtnStatus.message += rtnStatusRoi.message + QString(", ");
        rtnStatus.message += rtnStatusBgSub.message + QString(", ");
        rtnStatus.message += rtnStatusHeadTail.message + QString(", ");
        rtnStatus.message += rtnStatusMisc.message;

        return rtnStatus;

    }

    RtnStatus FlyTrackConfig::setBgEstFromMap(QVariantMap configMap) {
		RtnStatus rtnStatus;
		rtnStatus.success = true;
        rtnStatus.message = QString("");

        if (configMap.isEmpty())
        {
            rtnStatus.message = QString("flyTrack bgEst config empty");
            return rtnStatus;
        }

        if (configMap.contains("nFramesBgEst")) {
            if(configMap["nFramesBgEst"].canConvert<int>()) 
                nFramesBgEst = configMap["nFramesBgEst"].toInt();
            else {
				rtnStatus.success = false;
				rtnStatus.appendMessage("unable to convert nFramesBgEst to int");
			}
        }
        if (configMap.contains("lastFrameSample")) {
			if(configMap["lastFrameSample"].canConvert<int>()) 
                lastFrameSample = configMap["lastFrameSample"].toInt();
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert lastFrameSample to int");
            }
        }
		return rtnStatus;
	}

    RtnStatus FlyTrackConfig::setRoiFromMap(QVariantMap configMap) {
		RtnStatus rtnStatus;
		rtnStatus.success = true;
		rtnStatus.message = QString("");

        if (configMap.isEmpty())
        {
			rtnStatus.message = QString("flyTrack roi config empty");
			return rtnStatus;
		}

        if (configMap.contains("roiType")) {
            if (configMap["roiType"].canConvert<QString>()) {
                QString roiTypeStr = configMap["roiType"].toString();
                if (roiTypeStr == "CIRCLE") roiType = CIRCLE;
                else if (roiTypeStr == "NONE") roiType = NONE;
                else {
                    rtnStatus.success = false;
                    rtnStatus.appendMessage("unable to parse roiType");
                }
            }
            else {
				rtnStatus.success = false;
				rtnStatus.appendMessage("unable to convert roiType to string");
			}
		}

        setRoiFracAbsFromMap(configMap, rtnStatus, "roiCenterXFrac", "roiCenterX", roiCenterXFrac_, roiCenterX, imageWidth_);
        setRoiFracAbsFromMap(configMap, rtnStatus, "roiCenterYFrac", "roiCenterY", roiCenterYFrac_, roiCenterY, imageHeight_);
		setRoiFracAbsFromMap(configMap, rtnStatus, "roiRadiusFrac", "roiRadius", roiRadiusFrac_, roiRadius, std::min(imageWidth_, imageHeight_));

		return rtnStatus;
	}   

    RtnStatus FlyTrackConfig::setBgSubFromMap(QVariantMap configMap) {

        RtnStatus rtnStatus;
		rtnStatus.success = true;
		rtnStatus.message = QString("");

        if (configMap.isEmpty())
        {
			rtnStatus.message = QString("flyTrack bgSub config empty");
			return rtnStatus;
		}

        if (configMap.contains("backgroundThreshold")) {
			if(configMap["backgroundThreshold"].canConvert<int>()) 
                backgroundThreshold = configMap["backgroundThreshold"].toInt();
            else {
				rtnStatus.success = false;
				rtnStatus.appendMessage("unable to convert backgroundThreshold to int");
			}
		}
        if (configMap.contains("flyVsBgMode")) {
            if (configMap["flyVsBgMode"].canConvert<QString>()) {
				QString flyVsBgModeStr = configMap["flyVsBgMode"].toString();
				if (flyVsBgModeStr == "FLY_DARKER_THAN_BG") flyVsBgMode = FLY_DARKER_THAN_BG;
				else if (flyVsBgModeStr == "FLY_BRIGHTER_THAN_BG") flyVsBgMode = FLY_BRIGHTER_THAN_BG;
                else if (flyVsBgModeStr == "FLY_ANY_DIFFERENCE_BG") flyVsBgMode = FLY_ANY_DIFFERENCE_BG;
                else {
					rtnStatus.success = false;
					rtnStatus.appendMessage("unable to parse flyVsBgMode");
				}
			}
            else {
				rtnStatus.success = false;
				rtnStatus.appendMessage("unable to convert flyVsBgMode to string");
			}
		}
		return rtnStatus;
	
    }

    RtnStatus FlyTrackConfig::setHeadTailFromMap(QVariantMap configMap) {
		RtnStatus rtnStatus;
        rtnStatus.success = true;
        rtnStatus.message = QString("");
        if (configMap.isEmpty())
        {
			rtnStatus.message = QString("flyTrack headTail config empty");
			return rtnStatus;
		}
        if (configMap.contains("historyBufferLength")) {
            if (configMap["historyBufferLength"].canConvert<int>())
                historyBufferLength = configMap["historyBufferLength"].toInt();
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert historyBufferLength to int");
            }
        }
        if (configMap.contains("minVelocityMagnitude")) {
            if (configMap["minVelocityMagnitude"].canConvert<double>())
                minVelocityMagnitude = configMap["minVelocityMagnitude"].toDouble();
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert minVelocityMagnitude to double");
            }
        }
        if (configMap.contains("headTailWeightVelocity")) {
            if (configMap["headTailWeightVelocity"].canConvert<double>())
                headTailWeightVelocity = configMap["headTailWeightVelocity"].toDouble();
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert headTailWeightVelocity to double");
            }
        }
        return rtnStatus;
    }

    RtnStatus FlyTrackConfig::setMiscFromMap(QVariantMap configMap) {
		RtnStatus rtnStatus;
		rtnStatus.success = true;
		rtnStatus.message = QString("");
        if (configMap.isEmpty())
        {
			rtnStatus.message = QString("flyTrack misc config empty");
			return rtnStatus;
		}
        if (configMap.contains("DEBUG")) {
            if (configMap["DEBUG"].canConvert<bool>())
                DEBUG = configMap["DEBUG"].toBool();
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert DEBUG to bool");
            }
        }
        if (configMap.contains("imageWidth")) {
            if (configMap["imageWidth"].canConvert<int>())
                imageWidth_ = configMap["imageWidth"].toInt();
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert imageWidth to int");
            }
        }
        if (configMap.contains("imageHeight")) {
			if (configMap["imageHeight"].canConvert<int>())
				imageHeight_ = configMap["imageHeight"].toInt();
            else {
				rtnStatus.success = false;
				rtnStatus.appendMessage("unable to convert imageHeight to int");
			}
		}
        return rtnStatus;
    }

    QVariantMap FlyTrackConfig::toMap()
    {
        // Create Device map
        QVariantMap configMap;
        QVariantMap bgEstMap;
        bgEstMap.insert("nFramesBgEst", nFramesBgEst);
        bgEstMap.insert("lastFrameSample", lastFrameSample);

        QVariantMap roiMap;
        roiMap.insert("roiType", roiType);
        roiMap.insert("roiCenterXFrac", roiCenterXFrac_);
        roiMap.insert("roiCenterYFrac", roiCenterYFrac_);
        roiMap.insert("roiRadiusFrac", roiRadiusFrac_);
        roiMap.insert("roiCenterX", roiCenterX);
        roiMap.insert("roiCenterY", roiCenterY);
        roiMap.insert("roiRadius", roiRadius);

        QVariantMap bgSubMap;
        bgSubMap.insert("backgroundThreshold", backgroundThreshold);
        bgSubMap.insert("flyVsBgMode", flyVsBgMode);

        QVariantMap headTailMap;
        headTailMap.insert("historyBufferLength", historyBufferLength);
        headTailMap.insert("minVelocityMagnitude", minVelocityMagnitude);
        headTailMap.insert("headTailWeightVelocity", headTailWeightVelocity);

        QVariantMap miscMap;
        miscMap.insert("DEBUG", DEBUG);
        miscMap.insert("imageWidth", imageWidth_);
        miscMap.insert("imageHeight", imageHeight_);

		configMap.insert("bgEst", bgEstMap);
        configMap.insert("roi", roiMap);
        configMap.insert("bgSub", bgSubMap);
        configMap.insert("headTail", headTailMap);
        configMap.insert("misc", miscMap);

        return configMap;
    }

}