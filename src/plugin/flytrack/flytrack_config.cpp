#include "flytrack_config.hpp"
#include "json.hpp"
#include <iostream>
#include <QMessageBox>
#include <QtDebug>
#include <QFileInfo>

namespace bias
{

    const QString FlyTrackConfig::DEFAULT_BG_VIDEO_FILE_PATH = QString(""); // video to estimate background from
    const QString FlyTrackConfig::DEFAULT_BG_IMAGE_FILE_PATH = QString(""); // saved background median estimate
    const QString FlyTrackConfig::DEFAULT_TMP_OUT_DIR = QString("tmp"); // temporary output directory
    const int FlyTrackConfig::DEFAULT_BACKGROUND_THRESHOLD = 75; // foreground/background threshold, between 0 and 255
    const int FlyTrackConfig::DEFAULT_N_FRAMES_BG_EST = 100; // number of frames used for background estimation, set to 0 to use all frames
    const int FlyTrackConfig::DEFAULT_N_FRAMES_SKIP_BG_EST = 500; // number of frames used for background estimation, set to 0 to use all frames
    const int FlyTrackConfig::DEFAULT_LAST_FRAME_SAMPLE = 0; // last frame sampled for background estimation, set to 0 to use last frame of video
    const FlyVsBgModeType FlyTrackConfig::DEFAULT_FLY_VS_BG_MODE = FLY_DARKER_THAN_BG; // whether the fly is darker than the background
    const ROIType FlyTrackConfig::DEFAULT_ROI_TYPE = CIRCLE; // type of ROI
    const int FlyTrackConfig::DEFAULT_HISTORY_BUFFER_LENGTH = 5; // number of frames to buffer velocity, orientation
    const double FlyTrackConfig::DEFAULT_MIN_VELOCITY_MAGNITUDE = 1.0; // minimum velocity magnitude in pixels/frame to consider fly moving
    const double FlyTrackConfig::DEFAULT_HEAD_TAIL_WEIGHT_VELOCITY = 3.0; // weight of velocity dot product in head-tail orientation resolution
    const double FlyTrackConfig::DEFAULT_MIN_VEL_MATCH_DOTPROD = 0.25; // minimum dot product for velocity matching
    const bool FlyTrackConfig::DEFAULT_DEBUG = false; // flag for debugging
    const bool FlyTrackConfig::DEFAULT_COMPUTE_BG_MODE = false; // flag of whether to compute the background (true) when camera is running or track a fly (false)

    FlyTrackConfig::FlyTrackConfig()
    {
        computeBgMode = DEFAULT_COMPUTE_BG_MODE;
        bgVideoFilePath = DEFAULT_BG_VIDEO_FILE_PATH;
        bgImageFilePath = DEFAULT_BG_IMAGE_FILE_PATH;
        tmpOutDir = DEFAULT_TMP_OUT_DIR;
		backgroundThreshold = DEFAULT_BACKGROUND_THRESHOLD;
		nFramesBgEst = DEFAULT_N_FRAMES_BG_EST;
        nFramesSkipBgEst = DEFAULT_N_FRAMES_SKIP_BG_EST;
		lastFrameSample = DEFAULT_LAST_FRAME_SAMPLE;
		flyVsBgMode = DEFAULT_FLY_VS_BG_MODE;
		roiType = DEFAULT_ROI_TYPE;
		historyBufferLength = DEFAULT_HISTORY_BUFFER_LENGTH;
		minVelocityMagnitude = DEFAULT_MIN_VELOCITY_MAGNITUDE;
		headTailWeightVelocity = DEFAULT_HEAD_TAIL_WEIGHT_VELOCITY;
		DEBUG = DEFAULT_DEBUG;
		roiCenterX = 0;
		roiCenterY = 0;
		roiRadius = 0;
        trackFileName = QString(""); // empty string means it is not set
        tmpTrackFilePath = QString(""); // empty string means it is not set
	}

    FlyTrackConfig FlyTrackConfig::copy() {
    	FlyTrackConfig config;
        config.computeBgMode = computeBgMode;
		config.bgVideoFilePath = bgVideoFilePath;
		config.bgImageFilePath = bgImageFilePath;
		config.tmpOutDir = tmpOutDir;
		config.backgroundThreshold = backgroundThreshold;
		config.nFramesBgEst = nFramesBgEst;
        config.nFramesSkipBgEst = nFramesSkipBgEst;
		config.lastFrameSample = lastFrameSample;
		config.flyVsBgMode = flyVsBgMode;
		config.roiType = roiType;
		config.historyBufferLength = historyBufferLength;
		config.minVelocityMagnitude = minVelocityMagnitude;
		config.headTailWeightVelocity = headTailWeightVelocity;
		config.DEBUG = DEBUG;
		config.roiCenterX = roiCenterX;
		config.roiCenterY = roiCenterY;
		config.roiRadius = roiRadius;
        config.trackFileName = trackFileName;
        config.tmpTrackFilePath = tmpTrackFilePath;
		return config;
	
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
        configStr += QString("computeBgMode: %1\n").arg(computeBgMode);
        configStr += QString("bgVideoFilePath: %1\n").arg(bgVideoFilePath);
        configStr += QString("bgImageFilePath: %1\n").arg(bgImageFilePath);
        configStr += QString("tmpOutDir: %1\n").arg(tmpOutDir);
        configStr += QString("trackFileName: %1\n").arg(trackFileName);
        configStr += QString("tmpTrackFilePath: %1\n").arg(tmpTrackFilePath);
        configStr += QString("backgroundThreshold: %1\n").arg(backgroundThreshold);
        configStr += QString("nFramesBgEst: %1\n").arg(nFramesBgEst);
        configStr += QString("lastFrameSample: %1\n").arg(lastFrameSample);
        configStr += QString("nFramesSkipBgEst: %1\n").arg(nFramesSkipBgEst);
        configStr += QString("flyVsBgMode: %1\n").arg(flyVsBgMode);
        QString roiTypeString;
        roiTypeToString(roiType, roiTypeString);
        configStr += QString("roiType: %1\n").arg(roiTypeString);
        configStr += QString("roiCenterX: %1\n").arg(roiCenterX);
        configStr += QString("roiCenterY: %1\n").arg(roiCenterY);
        configStr += QString("roiRadius: %1\n").arg(roiRadius);
        configStr += QString("historyBufferLength: %1\n").arg(historyBufferLength);
        configStr += QString("minVelocityMagnitude: %1\n").arg(minVelocityMagnitude);
        configStr += QString("headTailWeightVelocity: %1\n").arg(headTailWeightVelocity);
        configStr += QString("DEBUG: %1\n").arg(DEBUG);
        return configStr;

    }

    void FlyTrackConfig::print() {
		std::cout << toString().toStdString();
	}

    bool FlyTrackConfig::trackFilePathSet() {
  		return !tmpTrackFilePath.isEmpty();
    }
    bool FlyTrackConfig::trackFileNameSet(){
        return !trackFileName.isEmpty();
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
        if (configMap.contains("computeBgMode")) {
			if (configMap["computeBgMode"].canConvert<bool>())
				computeBgMode = configMap["computeBgMode"].toBool();
            else {
				rtnStatus.success = false;
				rtnStatus.appendMessage("unable to convert computeBgMode to bool");
			}
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
        if (configMap.contains("nFramesSkipBgEst")) {
            if (configMap["nFramesSkipBgEst"].canConvert<int>())
                nFramesSkipBgEst = configMap["nFramesSkipBgEst"].toInt();
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert nFramesSkipBgEst to int");
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

        if (configMap.contains("roiCenterX")) {
            if (configMap["roiCenterX"].canConvert<double>()) {
                roiCenterX = configMap["roiCenterX"].toDouble();
            }
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage(QString("unable to convert roiCenterX to double"));
            }
        }

        if (configMap.contains("roiCenterY")) {
            if (configMap["roiCenterY"].canConvert<double>()) {
                roiCenterY = configMap["roiCenterY"].toDouble();
            }
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage(QString("unable to convert roiCenterY to double"));
            }
        }

        if (configMap.contains("roiRadius")) {
            if (configMap["roiRadius"].canConvert<double>()) {
                roiRadius = configMap["roiRadius"].toDouble();
            }
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage(QString("unable to convert roiRadius to double"));
            }
        }
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
        if (configMap.contains("tmpOutDir")) {
			if (configMap["tmpOutDir"].canConvert<QString>())
				tmpOutDir = configMap["tmpOutDir"].toString();
            else {
				rtnStatus.success = false;
				rtnStatus.appendMessage("unable to convert tmpOutDir to string");
			}
		}
        if (configMap.contains("trackFileName")) {
			if (configMap["trackFileName"].canConvert<QString>())
				trackFileName = configMap["trackFileName"].toString();
            else {
				rtnStatus.success = false;
				rtnStatus.appendMessage("unable to convert trackFileName to string");
			}
		} 
        return rtnStatus;
    }

    QVariantMap FlyTrackConfig::toMap()
    {
        fprintf(stderr, "FlyTrackConfig::toMap\n");
        // Create Device map
        QVariantMap configMap;
        QVariantMap bgEstMap;
        bgEstMap.insert("computeBgMode", computeBgMode);
        bgEstMap.insert("bgVideoFilePath", bgVideoFilePath);
        bgEstMap.insert("bgImageFilePath", bgImageFilePath);
        bgEstMap.insert("nFramesBgEst", nFramesBgEst);
        bgEstMap.insert("lastFrameSample", lastFrameSample);
        bgEstMap.insert("nFramesSkipBgEst", nFramesSkipBgEst);

        QVariantMap roiMap;
        QString roiTypeString;
        roiTypeToString(roiType, roiTypeString);
        roiMap.insert("roiType", roiTypeString);
        roiMap.insert("roiCenterX", roiCenterX);
        roiMap.insert("roiCenterY", roiCenterY);
        roiMap.insert("roiRadius", roiRadius);

        QVariantMap bgSubMap;
        bgSubMap.insert("backgroundThreshold", backgroundThreshold);
        QString flyVsBgModeString;
        flyVsBgModeToString(flyVsBgMode, flyVsBgModeString);
        bgSubMap.insert("flyVsBgMode", flyVsBgModeString);

        QVariantMap headTailMap;
        headTailMap.insert("historyBufferLength", historyBufferLength);
        headTailMap.insert("minVelocityMagnitude", minVelocityMagnitude);
        headTailMap.insert("headTailWeightVelocity", headTailWeightVelocity);

        QVariantMap miscMap;
        miscMap.insert("DEBUG", DEBUG);
        miscMap.insert("tmpOutDir", tmpOutDir);
        miscMap.insert("trackFileName", trackFileName);

		configMap.insert("bgEst", bgEstMap);
        configMap.insert("roi", roiMap);
        configMap.insert("bgSub", bgSubMap);
        configMap.insert("headTail", headTailMap);
        configMap.insert("misc", miscMap);

        fprintf(stderr,"Done with FlyTrackConfig::toMap\n");

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

    // helper functions

    bool roiTypeToString(ROIType roiType, QString& roiTypeString) {
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
    }


}