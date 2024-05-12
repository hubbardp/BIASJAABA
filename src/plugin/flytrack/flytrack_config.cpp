#include "flytrack_config.hpp"
#include "json.hpp"
#include <iostream>
#include <QMessageBox>
#include <QtDebug>

namespace bias
{

    // helper functions

    bool roiTypeToString(ROIType roiType,QString& roiTypeString) {
        switch (roiType) {
        case CIRCLE:
            roiTypeString = QString("CIRCLE");
            return true;
        case NONE:
            roiTypeString = QString("NONE");
            return true;
        default:
            return false;
        }
    }   
    bool roiTypeFromString(QString roiTypeString, ROIType& roiType) {
        if (roiTypeString == "CIRCLE") {
            roiType = CIRCLE;
            return true;
        }
        if (roiTypeString == "NONE") {
            roiType = NONE;
            return true;
        }
        roiTypeString = "UNKNOWN";
        return false;
    }

    bool flyVsBgModeToString(FlyVsBgModeType flyVsBgMode, QString& flyVsBgModeString) {
		switch (flyVsBgMode) {
		case FLY_DARKER_THAN_BG:
			flyVsBgModeString = QString("FLY_DARKER_THAN_BG");
			return true;
		case FLY_BRIGHTER_THAN_BG:
			flyVsBgModeString = QString("FLY_BRIGHTER_THAN_BG");
			return true;
		case FLY_ANY_DIFFERENCE_BG:
			flyVsBgModeString = QString("FLY_ANY_DIFFERENCE_BG");
			return true;
		default:
			return false;
		}
	}

    bool flyVsBgModeFromString(QString flyVsBgModeString, FlyVsBgModeType& flyVsBgMode) {
        if (flyVsBgModeString == "FLY_DARKER_THAN_BG") {
            flyVsBgMode = FLY_DARKER_THAN_BG;
			return true;
		}
        if (flyVsBgModeString == "FLY_BRIGHTER_THAN_BG") {
			flyVsBgMode = FLY_BRIGHTER_THAN_BG;
            return true;
            }
        if (flyVsBgModeString == "FLY_ANY_DIFFERENCE_BG") {
            flyVsBgMode = FLY_ANY_DIFFERENCE_BG;
			return true;
		}
        flyVsBgModeString = "UNKNOWN";
		return false;
	}

    void FlyTrackConfig::setRoiParams(ROIType roiTypeNew, double roiCenterXNew, double roiCenterYNew, double roiRadiusNew) {
    	roiType = roiTypeNew;
		roiCenterX = roiCenterXNew;
		roiCenterY = roiCenterYNew;
		roiRadius = roiRadiusNew;
        roiLocationSet_ = true;
    }

    void FlyTrackConfig::setRoiFracAbsFromMap(QVariantMap configMap, RtnStatus& rtnStatus,
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
                roiLocationSet_ = true;
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

    const QString FlyTrackConfig::DEFAULT_BG_VIDEO_FILE_PATH = QString("dummy_bg_video.avi"); // video to estimate background from
    const QString FlyTrackConfig::DEFAULT_BG_IMAGE_FILE_PATH = QString(""); // saved background median estimate
    const QString FlyTrackConfig::DEFAULT_TMP_OUT_DIR = QString("tmp"); // temporary output directory
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
        bgVideoFilePath = DEFAULT_BG_VIDEO_FILE_PATH;
        bgImageFilePath = DEFAULT_BG_IMAGE_FILE_PATH;
        tmpOutDir = DEFAULT_TMP_OUT_DIR;
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
        roiLocationSet_ = false;
		roiCenterX = 0;
		roiCenterY = 0;
		roiRadius = 0;
		imageWidth_ = -1;
		imageHeight_ = -1;
	}

    FlyTrackConfig FlyTrackConfig::copy() {
    	FlyTrackConfig config;
		config.bgVideoFilePath = bgVideoFilePath;
		config.bgImageFilePath = bgImageFilePath;
		config.tmpOutDir = tmpOutDir;
		config.backgroundThreshold = backgroundThreshold;
		config.nFramesBgEst = nFramesBgEst;
		config.lastFrameSample = lastFrameSample;
		config.flyVsBgMode = flyVsBgMode;
		config.roiType = roiType;
		config.roiCenterXFrac_ = roiCenterXFrac_;
		config.roiCenterYFrac_ = roiCenterYFrac_;
		config.roiRadiusFrac_ = roiRadiusFrac_;
		config.historyBufferLength = historyBufferLength;
		config.minVelocityMagnitude = minVelocityMagnitude;
		config.headTailWeightVelocity = headTailWeightVelocity;
		config.DEBUG = DEBUG;
		config.roiLocationSet_ = roiLocationSet_;
		config.roiCenterX = roiCenterX;
		config.roiCenterY = roiCenterY;
		config.roiRadius = roiRadius;
		config.imageWidth_ = imageWidth_;
		config.imageHeight_ = imageHeight_;
		return config;
	
    }

	void FlyTrackConfig::setImageSize(int width, int height)
	{
        printf("set image size: width: %d, height: %d\n", width, height);
		imageWidth_ = width;
		imageHeight_ = height;
        if (!roiLocationSet_) {
            roiCenterX = imageWidth_ * roiCenterXFrac_;
            roiCenterY = imageHeight_ * roiCenterYFrac_;
            roiRadius = std::min(imageWidth_, imageHeight_) * roiRadiusFrac_;
            printf("Set absolute ROI location based on frac: x: %f, y: %f, r: %f\n", roiCenterX, roiCenterY, roiRadius);
        }
	}
    void FlyTrackConfig::setBgVideoFilePath(QString bgVideoFilePathIn) {
        bgVideoFilePath = bgVideoFilePathIn;
        if (!bgImageFilePath.isEmpty()) {
            return;
		}
        // remove extension from bgVideoFilePath
        QString bgVideoFilePathNoExt = bgVideoFilePath;
        int lastDotIndex = bgVideoFilePath.lastIndexOf(".");
        if (lastDotIndex > 0) {
            bgVideoFilePathNoExt = bgVideoFilePath.left(lastDotIndex);
        }
        bgImageFilePath = bgVideoFilePathNoExt + QString("_bg.png");
    }

    QString FlyTrackConfig::toString() {
        QString configStr;
        configStr += QString("bgVideoFilePath: %1\n").arg(bgVideoFilePath);
        configStr += QString("bgImageFilePath: %1\n").arg(bgImageFilePath);
        configStr += QString("tmpOutDir: %1\n").arg(tmpOutDir);
        configStr += QString("backgroundThreshold: %1\n").arg(backgroundThreshold);
        configStr += QString("nFramesBgEst: %1\n").arg(nFramesBgEst);
        configStr += QString("lastFrameSample: %1\n").arg(lastFrameSample);
        configStr += QString("flyVsBgMode: %1\n").arg(flyVsBgMode);
        QString roiTypeString;
        roiTypeToString(roiType, roiTypeString);
        configStr += QString("roiType: %1\n").arg(roiTypeString);
        configStr += QString("roiCenterX: %1\n").arg(roiCenterX);
        configStr += QString("roiCenterY: %1\n").arg(roiCenterY);
        configStr += QString("roiRadius: %1\n").arg(roiRadius);
        configStr += QString("roiCenterXFrac: %1\n").arg(roiCenterXFrac_);
        configStr += QString("roiCenterYFrac: %1\n").arg(roiCenterYFrac_);
        configStr += QString("roiRadiusFrac: %1\n").arg(roiRadiusFrac_);
        configStr += QString("roiLocationSet: %1\n").arg(roiLocationSet_);
        configStr += QString("historyBufferLength: %1\n").arg(historyBufferLength);
        configStr += QString("minVelocityMagnitude: %1\n").arg(minVelocityMagnitude);
        configStr += QString("headTailWeightVelocity: %1\n").arg(headTailWeightVelocity);
        configStr += QString("DEBUG: %1\n").arg(DEBUG);
        return configStr;

    }

    void FlyTrackConfig::print() {
		std::cout << toString().toStdString();
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
        if (configMap.contains("bgImageFilePath")) {
            if (configMap["bgImageFilePath"].canConvert<QString>())
                bgImageFilePath = configMap["bgImageFilePath"].toString();
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert bgImageFilePath to string");
            }
        }
        if (configMap.contains("bgVideoFilePath")) {
            if (configMap["bgVideoFilePath"].canConvert<QString>()) {
                setBgVideoFilePath(configMap["bgVideoFilePath"].toString());
            }
            else {
				rtnStatus.success = false;
				rtnStatus.appendMessage("unable to convert bgVideoFilePath to string");
			}
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
                bool success = roiTypeFromString(roiTypeStr, roiType);
                if(!success) {
                    rtnStatus.success = false;
                    rtnStatus.appendMessage(QString("unknown roiType %1").arg(roiTypeStr));
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
        if (configMap.contains("tmpOutDir")) {
			if (configMap["tmpOutDir"].canConvert<QString>())
				tmpOutDir = configMap["tmpOutDir"].toString();
            else {
				rtnStatus.success = false;
				rtnStatus.appendMessage("unable to convert tmpOutDir to string");
			}
		}
        return rtnStatus;
    }

    QVariantMap FlyTrackConfig::toMap()
    {
        // Create Device map
        QVariantMap configMap;
        QVariantMap bgEstMap;
        bgEstMap.insert("bgVideoFilePath", bgVideoFilePath);
        bgEstMap.insert("bgImageFilePath", bgImageFilePath);
        bgEstMap.insert("nFramesBgEst", nFramesBgEst);
        bgEstMap.insert("lastFrameSample", lastFrameSample);

        QVariantMap roiMap;
        QString roiTypeString;
        roiTypeToString(roiType, roiTypeString);
        roiMap.insert("roiType", roiTypeString);
        roiMap.insert("roiCenterXFrac", roiCenterXFrac_);
        roiMap.insert("roiCenterYFrac", roiCenterYFrac_);
        roiMap.insert("roiRadiusFrac", roiRadiusFrac_);
        roiMap.insert("roiCenterX", roiCenterX);
        roiMap.insert("roiCenterY", roiCenterY);
        roiMap.insert("roiRadius", roiRadius);

        QVariantMap bgSubMap;
        bgSubMap.insert("backgroundThreshold", backgroundThreshold);
        switch (flyVsBgMode) {
			case FLY_DARKER_THAN_BG:
				bgSubMap.insert("flyVsBgMode", QString("FLY_DARKER_THAN_BG"));
				break;
			case FLY_BRIGHTER_THAN_BG:
				bgSubMap.insert("flyVsBgMode", QString("FLY_BRIGHTER_THAN_BG"));
				break;
			case FLY_ANY_DIFFERENCE_BG:
				bgSubMap.insert("flyVsBgMode", QString("FLY_ANY_DIFFERENCE_BG"));
				break;
		}

        QVariantMap headTailMap;
        headTailMap.insert("historyBufferLength", historyBufferLength);
        headTailMap.insert("minVelocityMagnitude", minVelocityMagnitude);
        headTailMap.insert("headTailWeightVelocity", headTailWeightVelocity);

        QVariantMap miscMap;
        miscMap.insert("DEBUG", DEBUG);
        miscMap.insert("imageWidth", imageWidth_);
        miscMap.insert("imageHeight", imageHeight_);
        miscMap.insert("tmpOutDir", tmpOutDir);

		configMap.insert("bgEst", bgEstMap);
        configMap.insert("roi", roiMap);
        configMap.insert("bgSub", bgSubMap);
        configMap.insert("headTail", headTailMap);
        configMap.insert("misc", miscMap);

        return configMap;
    }

    RtnStatus FlyTrackConfig::fromJson(QByteArray jsonConfigArray) {
        RtnStatus rtnStatus;
        rtnStatus.success = true;
        rtnStatus.message = QString("");

        bool ok;
        QVariantMap configMap = QtJson::parse(QString(jsonConfigArray), ok).toMap();
        if (!ok)
        {
            rtnStatus.success = false;
            rtnStatus.message = QString("FlyTrack unable to parse json configuration string");
            return rtnStatus;

        }
        rtnStatus = fromMap(configMap);
        return rtnStatus;
    }

    QByteArray FlyTrackConfig::toJson()
    {
        QVariantMap configMap = toMap();

        bool ok;
        QByteArray jsonConfigArray = QtJson::serialize(configMap, ok);
        if (!ok)
        {
            jsonConfigArray = QByteArray();
        }
        return jsonConfigArray;
    }

}