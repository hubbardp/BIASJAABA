#include "flytrack_plugin.hpp"
#include <QtDebug>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "camera_window.hpp"
#include <iostream>
#include "background_data_ufmf.hpp"
#include "video_utils.hpp"

namespace bias
{

    const QString FlyTrackPlugin::PLUGIN_NAME = QString("FlyTrack"); 
    const QString FlyTrackPlugin::PLUGIN_DISPLAY_NAME = QString("Fly Track");
    const QString FlyTrackPlugin::LOG_FILE_EXTENSION = QString("txt");
    const QString FlyTrackPlugin::LOG_FILE_POSTFIX = QString("plugin_log");

    const unsigned int FlyTrackPlugin::DEFAULT_NUM_BINS = 256;
    const unsigned int FlyTrackPlugin::DEFAULT_BIN_SIZE = 1;
    const unsigned int FlyTrackPlugin::FLY_DARKER_THAN_BG = 0;
    const unsigned int FlyTrackPlugin::FLY_BRIGHTER_THAN_BG = 1;
    const unsigned int FlyTrackPlugin::FLY_ANY_DIFFERENCE_BG = 2;

    // Public
    // ------------------------------------------------------------------------

    FlyTrackPlugin::FlyTrackPlugin(QWidget *parent) : BiasPlugin(parent) 
    { 
        // hard code parameters
        // these should go in a config file/GUI

        // parameters for background subtraction
        backgroundThreshold_ = 75;
        maxNCCs_ = 10;
        flyVsBgMode_ = FLY_DARKER_THAN_BG;

        // parameters for background estimation
        nFramesBgEst_ = 100;
        bgVideoFilePath_ = QString("C:\\Code\\BIAS\\testdata\\20240409T155835_P1_movie1.avi");
        bgImageFilePath_ = QString("C:\\Code\\BIAS\\testdata\\20240409T155835_P1_movie1_bg.png");
        tmpOutDir_ = QString("C:\\Code\\BIAS\\testdata\\tmp");
        lastFrameSample_ = -1;

        // parameters for region of interest
        roiCenterX_ = 468.6963;
        roiCenterY_ = 480.2917;
        roiRadius_ = 428.3618;

        active_ = false;
        isFirst_ = true;

        // initialize the color table
        //fprintf(stderr, "Initializing color table\n");
        //colorTable_.reserve(maxNCCs_);
        //colorTable_[0] = cv::Vec3b(0, 0, 0); // background
        //for (int label = 1; label < maxNCCs_; ++label)
        //{
        //    colorTable_[label] = cv::Vec3b((label * 180 / maxNCCs_) % 180, 255, 255);
        //}
        setRequireTimer(false);
    }

    void FlyTrackPlugin::setBackgroundModel() {

        cv::Mat bgMedianImage;
        // check if bgImageFilePath_ exists
        if (QFile::exists(bgImageFilePath_)) {
            loadBackgroundModel(bgMedianImage, bgImageFilePath_);
            printf("Reading background image from %s\n",bgImageFilePath_.toStdString().c_str());
        	bgMedianImage = cv::imread(bgImageFilePath_.toStdString(), cv::IMREAD_GRAYSCALE);
            printf("Done\n");
            fflush(stdout);
		}
        else {
            computeBackgroundMedian(bgMedianImage);
        }

        // store background model
        storeBackgroundModel(bgMedianImage);
        acquireLock();
        bgMedianImage_ = bgMedianImage.clone();
		bgImageComputed_ = true;
        cv::add(bgMedianImage, backgroundThreshold_, bgUpperBoundImage_);
        cv::subtract(bgMedianImage, backgroundThreshold_, bgLowerBoundImage_);
        releaseLock();

        printf("Finished computing background model\n");
        fflush(stdout);

    }

    void FlyTrackPlugin::loadBackgroundModel(cv::Mat& bgMedianImage, QString bgImageFilePath) {
        printf("Reading background image from %s\n", bgImageFilePath_.toStdString().c_str());
        bgMedianImage = cv::imread(bgImageFilePath_.toStdString(), cv::IMREAD_GRAYSCALE);
        printf("Done\n");
        fflush(stdout);
    }

    void FlyTrackPlugin::storeBackgroundModel(cv::Mat& bgMedianImage) {

        acquireLock();
        bgMedianImage_ = bgMedianImage.clone();
        bgImageComputed_ = true;
        if (flyVsBgMode_ == FLY_DARKER_THAN_BG) {
            bgUpperBoundImage_ = bgMedianImage.clone();
        }
        else {
            cv::add(bgMedianImage, backgroundThreshold_, bgUpperBoundImage_);
        }
        if (flyVsBgMode_ == FLY_BRIGHTER_THAN_BG) {
            bgLowerBoundImage_ = bgMedianImage.clone();
        }
        else {
            cv::subtract(bgMedianImage, backgroundThreshold_, bgLowerBoundImage_);
        }
        releaseLock();

    }

    void FlyTrackPlugin::computeBackgroundMedian(cv::Mat& bgMedianImage) {
        videoBackend vidObj = videoBackend(bgVideoFilePath_);
        int nFrames = vidObj.getNumFrames();

        StampedImage newStampedImg;
        newStampedImg.image = vidObj.grabImage();

        BackgroundData_ufmf backgroundData;
        backgroundData = BackgroundData_ufmf(newStampedImg, DEFAULT_NUM_BINS, DEFAULT_BIN_SIZE);
        backgroundData.addImage(newStampedImg);

        // which frames to sample
        int nFramesBgEst = nFramesBgEst_;
        int lastFrameSample = lastFrameSample_;
        if (nFrames < nFramesBgEst_ || nFramesBgEst_ <= 0) nFramesBgEst = nFrames;
        if (nFrames < lastFrameSample_ || lastFrameSample_ <= 0) lastFrameSample = nFrames;
        int nFramesSkip = lastFrameSample / nFramesBgEst;

        // add evenly spaced frames to the background model
        printf("Reading frames for background estimation\n");
        fflush(stdout);
        for (int f = nFramesSkip; f < lastFrameSample; f += nFramesSkip) {
            printf("Reading frame %d\n", f);
            fflush(stdout);
            vidObj.setFrame(f);
            newStampedImg.image = vidObj.grabImage();
            backgroundData.addImage(newStampedImg);
        }
        printf("Finished reading.\n");
        // compute the median image
        printf("Computing median image\n");
        fflush(stdout);
        bgMedianImage = backgroundData.getMedianImage();
        printf("Done\n");
        fflush(stdout);
        // save the median image
        printf("Saving median image to %s\n", bgImageFilePath_.toStdString().c_str());
        cv::imwrite(bgImageFilePath_.toStdString(), bgMedianImage);
        printf("Done\n");
        fflush(stdout);
        backgroundData.clear();
    }

    void FlyTrackPlugin::reset()
    { }

    void FlyTrackPlugin::setFileAutoNamingString(QString autoNamingString)
    {
        fileAutoNamingString_ = autoNamingString;
    }

    void FlyTrackPlugin::setFileVersionNumber(unsigned verNum)
    {
        fileVersionNumber_ = verNum;
    }

    void FlyTrackPlugin::stop()
    { }

    void FlyTrackPlugin::setActive(bool value)
    {
        active_ = value;
        // compute background model
        if (value && !bgImageComputed_) {
            setBackgroundModel();
        }
    }


    bool FlyTrackPlugin::isActive()
    {
        return active_;
    }


    bool FlyTrackPlugin::requireTimer()
    {
        return requireTimer_;
    }

    void FlyTrackPlugin::processFrames(QList<StampedImage> frameList) 
    { 
        acquireLock();
        StampedImage latestFrame = frameList.back();
        frameList.clear();
        currentImage_ = latestFrame.image;
        timeStamp_ = latestFrame.timeStamp;
        frameCount_ = latestFrame.frameCount;
        // empty frame
        if ((currentImage_.rows == 0) || (currentImage_.cols == 0))
        {
			releaseLock();
			return;
		}
        // mismatched sizes
        if ((bgMedianImage_.rows != currentImage_.rows) || (bgMedianImage_.cols != currentImage_.cols)
            || bgMedianImage_.type() != currentImage_.type())
        {
            fprintf(stderr, "Background model and current image are not the same size\n");
            fprintf(stderr, "Background model: %d x %d, type %d\n", 
                bgMedianImage_.rows, bgMedianImage_.cols, bgMedianImage_.type());
            fprintf(stderr, "Current image: %d x %d, type %d\n",
                currentImage_.rows, currentImage_.cols, currentImage_.type());
			releaseLock();
			return;
		}

        // Get background/foreground membership, 255=background, 0=foreground
        switch(flyVsBgMode_){
			case FLY_DARKER_THAN_BG:
				isFg_ = currentImage_ < bgLowerBoundImage_;
				break;
			case FLY_BRIGHTER_THAN_BG:
				isFg_ = currentImage_ > bgUpperBoundImage_;
				break;
			case FLY_ANY_DIFFERENCE_BG:
				cv::inRange(currentImage_, bgLowerBoundImage_, bgUpperBoundImage_, isFg_);
                cv::bitwise_not(isFg_, isFg_);
				break;
		}
        if (flyVsBgMode_ == FLY_DARKER_THAN_BG)
        {
            isFg_ = currentImage_ < bgLowerBoundImage_;
		}
        else
        {
			cv::inRange(currentImage_, bgUpperBoundImage_, bgLowerBoundImage_, isFg_);
		}
        cv::inRange(currentImage_, bgLowerBoundImage_, bgUpperBoundImage_, isFg_);
        cv::absdiff(bgMedianImage_, currentImage_, dBkgd_);

        if (isFirst_){
			isFirst_ = false;
            //output dBkgd to file
            QString tmpOutFile = tmpOutDir_ + QString("/dBkgd.png");
            cv::imwrite(tmpOutFile.toStdString(), dBkgd_);
            //output isFg to file
            tmpOutFile = tmpOutDir_ + QString("/isFg.png");
            cv::imwrite(tmpOutFile.toStdString(), isFg_);
            //output lower bound to file
            tmpOutFile = tmpOutDir_ + QString("/bgLowerBound.png");
            cv::imwrite(tmpOutFile.toStdString(), bgLowerBoundImage_);
            //output upper bound to file
            tmpOutFile = tmpOutDir_ + QString("/bgUpperBound.png");
            cv::imwrite(tmpOutFile.toStdString(), bgUpperBoundImage_);
			releaseLock();
			return;
		}

        //cv::threshold(currentImage_, isFg_, backgroundThreshold_, 255, cv::THRESH_BINARY_INV);
        // find connected components in isFg_
        //cv::Mat stats;
        //cv::Mat centroids;
        //nCCs_ = cv::connectedComponentsWithStats(isFg_, ccLabels_, stats, centroids);
        // count number of foreground pixels
        //int fgCount = cv::countNonZero(isFg_);
        //printf("Timestamp: %f, nCCs: %d\n", timeStamp_,nCCs_);

        releaseLock();
    } 


    cv::Mat FlyTrackPlugin::getCurrentImage()
    {
        cv::Mat currentImageCopy;
        enum DisplayMode {FGTHRESH,DBKGD};
        DisplayMode displayMode = FGTHRESH;
        bool showfgbg = true;
        acquireLock();
        //cv::Mat currentImageCopy = currentImage_.clone();
        // make an image that is white where ccLabels_ == 1 and black elsewhere
        //cv::Mat currentImageCopy = ccLabels_ == 1;
        // fg/bg thresholded image
        //currentImageCopy = isFg_.clone();
        //cv::normalize(ccLabels_, currentImageCopy, 0, 255, cv::NORM_MINMAX, CV_8U);

        // Map component labels to hue val, 0 - 179 is the hue range in OpenCV
        // allocate a matrix the same size as label_hue that is all 255

        switch(displayMode){
            case FGTHRESH: currentImageCopy = isFg_.clone(); break;
            case DBKGD: currentImageCopy = dBkgd_.clone(); break;
        }
        releaseLock();
        return currentImageCopy;
    }


    QString FlyTrackPlugin::getName()
    {
        return PLUGIN_NAME;
    }


    QString FlyTrackPlugin::getDisplayName()
    {
        return PLUGIN_DISPLAY_NAME;
    }


    QPointer<CameraWindow> FlyTrackPlugin::getCameraWindow()
    {
        QPointer<CameraWindow> cameraWindowPtr = (CameraWindow*)(parent());
        return cameraWindowPtr;
    }


    RtnStatus FlyTrackPlugin::runCmdFromMap(QVariantMap cmdMap, bool showErrorDlg)
    {
        qDebug() << __FUNCTION__;
        RtnStatus rtnStatus;
        return rtnStatus;
    }

    QVariantMap FlyTrackPlugin::getConfigAsMap()  
    {
        QVariantMap configMap;
        return configMap;
    }

    RtnStatus FlyTrackPlugin::setConfigFromMap(QVariantMap configMap)
    {
        RtnStatus rtnStatus;
        rtnStatus.success = true;
        rtnStatus.message = QString("");
        return rtnStatus;
    }

    RtnStatus FlyTrackPlugin::setConfigFromJson(QByteArray jsonArray)
    {
        RtnStatus rtnStatus;
        rtnStatus.success = true;
        rtnStatus.message = QString("");
        return rtnStatus;
    }

    bool FlyTrackPlugin::pluginsEnabled()
    {
        return getCameraWindow() -> isPluginEnabled();
    }


    void FlyTrackPlugin::setPluginsEnabled(bool value)
    {
        getCameraWindow() -> setPluginEnabled(value);
    }


    QString FlyTrackPlugin::getLogFileExtension()
    {
        return LOG_FILE_EXTENSION;
    }

    QString FlyTrackPlugin::getLogFilePostfix()
    {
        return LOG_FILE_POSTFIX;
    }

    QString FlyTrackPlugin::getLogFileName(bool includeAutoNaming)
    {
        QPointer<CameraWindow> cameraWindowPtr = getCameraWindow();
        QString logFileName = cameraWindowPtr -> getVideoFileName() + QString("_") + getLogFilePostfix();
        if (includeAutoNaming)
        {
            if (!fileAutoNamingString_.isEmpty())
            {
                logFileName += QString("_") + fileAutoNamingString_;
            }
            if (fileVersionNumber_ != 0)
            {
                QString verStr = QString("_v%1").arg(fileVersionNumber_,3,10,QChar('0'));
                logFileName += verStr;
            }
        }
        logFileName += QString(".") + getLogFileExtension();
        return logFileName;
    }


    QString FlyTrackPlugin::getLogFileFullPath(bool includeAutoNaming)
    {
        QString logFileName = getLogFileName(includeAutoNaming);
        QPointer<CameraWindow> cameraWindowPtr = getCameraWindow();
        logFileDir_ = cameraWindowPtr -> getVideoFileDir();
        QString logFileFullPath = logFileDir_.absoluteFilePath(logFileName);
        return logFileFullPath;
    }

    // Protected methods
    // ------------------------------------------------------------------------

    void FlyTrackPlugin::setRequireTimer(bool value)
    {
        requireTimer_ = value;
    }


    void FlyTrackPlugin::openLogFile()
    {
        loggingEnabled_ = getCameraWindow() -> isLoggingEnabled();
        if (loggingEnabled_)
        {
            QString logFileFullPath = getLogFileFullPath(true);
            qDebug() << logFileFullPath;
            logFile_.setFileName(logFileFullPath);
            bool isOpen = logFile_.open(QIODevice::WriteOnly | QIODevice::Text);
            if (isOpen)
            {
                logStream_.setDevice(&logFile_);
            }
        }
    }


    void FlyTrackPlugin::closeLogFile()
    {
        if (loggingEnabled_ && logFile_.isOpen())
        {
            logStream_.flush();
            logFile_.close();
        }
    }

}
