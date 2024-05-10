#include "flytrack_plugin.hpp"
#include <QtDebug>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "camera_window.hpp"
#include <iostream>
#include "background_data_ufmf.hpp"
#include "video_utils.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

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

    // helper functions

    // void loadBackgroundModel(QString bgImageFilePath, cv::Mat& bgMedianImage)
    // load background model from file with cv::imread
    // inputs:
    // bgImageFilePath: path to background image file to load
    // bgMedianImage: destination for median background image
    void loadBackgroundModel(QString bgImageFilePath, cv::Mat& bgMedianImage) {
        if (!QFile::exists(bgImageFilePath)) {
            fprintf(stderr, "Background image file %s does not exist\n", bgImageFilePath.toStdString().c_str());
            exit(-1);
        }
        printf("Reading background image from %s\n", bgImageFilePath.toStdString().c_str());
        bgMedianImage = cv::imread(bgImageFilePath.toStdString(), cv::IMREAD_GRAYSCALE);
        printf("Done\n");
        fflush(stdout);
    }

    // void computeBackgroundMedian(cv::Mat& bgMedianImage)
    // compute the median background image from video in bgVideoFilePath_
    // inputs:
    // bgMedianImage: destination for median background image
    void computeBackgroundMedian(QString bgVideoFilePath,
        int nFramesBgEst, int lastFrameSample,
        cv::Mat& bgMedianImage) {
        videoBackend vidObj = videoBackend(bgVideoFilePath);
        int nFrames = vidObj.getNumFrames();

        StampedImage newStampedImg;
        newStampedImg.image = vidObj.grabImage();

        BackgroundData_ufmf backgroundData;
        backgroundData = BackgroundData_ufmf(newStampedImg, 
            FlyTrackPlugin::DEFAULT_NUM_BINS, 
            FlyTrackPlugin::DEFAULT_BIN_SIZE);
        backgroundData.addImage(newStampedImg);

        // which frames to sample
        if (nFrames < nFramesBgEst || nFramesBgEst <= 0) nFramesBgEst = nFrames;
        if (nFrames < lastFrameSample || lastFrameSample <= 0) lastFrameSample = nFrames;
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
        backgroundData.clear();
    }
    
    // int largestConnectedComponent(cv::Mat& isFg)
    // find largest connected components in isFg
    // inputs:
    // isFg: binary image, 255=background, 0=foreground
    // returns: area of largest connected component
    int largestConnectedComponent(cv::Mat& isFg) {
        cv::Mat ccLabels;
        int nCCs = cv::connectedComponents(isFg, ccLabels);
        // find largest connected component
        int maxArea = 0;
        int cc = 0;
        int currArea;
        for (int i = 1; i < nCCs; i++) {
            currArea = cv::countNonZero(ccLabels == i);
            if (currArea > maxArea) {
                maxArea = currArea;
                cc = i;
            }
        }
        isFg = ccLabels == cc;
        return maxArea;
    }

    // void fitEllipse(cv::Mat& isFg, EllipseParams& flyEllipse)
    // fit an ellipse to the foreground pixels in isFg. 
    // computes the principal components of the foreground pixel locations
    // creates an ellipse with center the mean of the pixel locations,
    // orientation the angle of the first principal component,
    // semi-major and semi-minor axes twice the square roots of the eigenvalues.
    // inputs:
    // isFg: binary image, 255=background, 0=foreground
    // flyEllipse: destination for ellipse parameters
    void fitEllipse(cv::Mat& isFg, EllipseParams& flyEllipse) {

        // eigen decomposition of covariance matrix
        // this probably isn't the fastest way to do this, but
        // it seems to work
        cv::Mat fgPixels;
        cv::findNonZero(isFg, fgPixels);
        cv::Mat fgPixelsD = cv::Mat::zeros(fgPixels.rows, 2, CV_64F);
        for (int i = 0; i < fgPixels.rows; i++) {
            fgPixelsD.at<double>(i, 0) = fgPixels.at<cv::Point>(i).x;
            fgPixelsD.at<double>(i, 1) = fgPixels.at<cv::Point>(i).y;
        }
        cv::PCA pca_analysis(fgPixelsD, cv::Mat(), cv::PCA::DATA_AS_ROW);
        flyEllipse.x = pca_analysis.mean.at<double>(0, 0);
        flyEllipse.y = pca_analysis.mean.at<double>(0, 1);
        // orientation of ellipse (modulo pi)
        flyEllipse.theta = std::atan2(pca_analysis.eigenvectors.at<double>(0, 1),
            pca_analysis.eigenvectors.at<double>(0, 0));
        // semi major, minor axis lengths
        double lambda1 = pca_analysis.eigenvalues.at<double>(0);
        double lambda2 = pca_analysis.eigenvalues.at<double>(1);
        flyEllipse.a = std::sqrt(lambda1) * 2.0;
        flyEllipse.b = std::sqrt(lambda2) * 2.0;
    }

    // Public
    // ------------------------------------------------------------------------

    // FlyTrackPlugin(QWidget *parent)
    // Constructor
    // Inputs:
    // parent: parent widget
    // sets all parameters
    // initializes state
    FlyTrackPlugin::FlyTrackPlugin(QWidget *parent) : BiasPlugin(parent) 
    { 
        // hard code parameters
        // these should go in a config file/GUI

        // parameters for background subtraction
        backgroundThreshold_ = 75;
        flyVsBgMode_ = FLY_DARKER_THAN_BG;

        // parameters for background estimation
        nFramesBgEst_ = 100;
        bgVideoFilePath_ = QString("C:\\Code\\BIAS\\testdata\\20240409T155835_P1_movie1.avi");
        bgImageFilePath_ = QString("C:\\Code\\BIAS\\testdata\\20240409T155835_P1_movie1_bg.png");
        tmpOutDir_ = QString("C:\\Code\\BIAS\\testdata\\tmp");
        lastFrameSample_ = -1;
        DEBUG_ = true;

        // parameters for region of interest
        roiType_ = CIRCLE;
        roiCenterX_ = 468.6963;
        roiCenterY_ = 480.2917;
        roiRadius_ = 428.3618;

        bgImageComputed_ = false;
        active_ = false;
        isFirst_ = false;

        setRequireTimer(false);
    }

    // cv::Mat circleROI(double centerX, double centerY, double centerRadius)
    // create a circular region of interest mask, inside is 255, outside 0
    // inputs:
    // centerX, centerY: center of circle
    // centerRadius: radius of circle
    // returns: mask image
    cv::Mat FlyTrackPlugin::circleROI(double centerX, double centerY, double centerRadius){
        cv::Mat mask = cv::Mat::zeros(bgMedianImage_.size(), CV_8U);
  		cv::circle(mask, cv::Point(centerX, centerY), centerRadius, cv::Scalar(255), -1);
   		return mask;
    }

    // void setROI()
    // set the region of interest mask based on roiType_
    // currently only circle implemented
    void FlyTrackPlugin::setROI() {
        // roi mask
        switch (roiType_) {
        case CIRCLE:
            inROI_ = circleROI(roiCenterX_, roiCenterY_, roiRadius_);
            break;
        }
	}

    // void setBackgroundModel()
    // compute and set the background model fields
    // if bgImageFilePath_ exists, load background model from file
    // otherwise, compute median background image from bgVideoFilePath_
    // store background model in bgMedianImage_, bgLowerBoundImage_, bgUpperBoundImage_
    // set inROI_ mask
    void FlyTrackPlugin::setBackgroundModel() {

        cv::Mat bgMedianImage;
        // check if bgImageFilePath_ exists
        if (QFile::exists(bgImageFilePath_)) {
            loadBackgroundModel(bgImageFilePath_, bgMedianImage);
		}
        else {
            computeBackgroundMedian(bgVideoFilePath_,nFramesBgEst_,lastFrameSample_,bgMedianImage);
            // save the median image
            printf("Saving median image to %s\n", bgImageFilePath_.toStdString().c_str());
            bool success = cv::imwrite(bgImageFilePath_.toStdString(), bgMedianImage);
            if (success) printf("Done\n");
            else printf("Failed to write background median image to %s\n", bgImageFilePath_.toStdString().c_str());
        }

        // store background model
        acquireLock();
        storeBackgroundModel(bgMedianImage);

        // roi mask
        setROI();

        bgImageComputed_ = true;
        releaseLock();

        //output lower bound to file
        if (DEBUG_) {
            QString tmpOutFile;
            tmpOutFile = tmpOutDir_ + QString("/bgLowerBound.png");
            cv::imwrite(tmpOutFile.toStdString(), bgLowerBoundImage_);
            //output upper bound to file
            tmpOutFile = tmpOutDir_ + QString("/bgUpperBound.png");
            cv::imwrite(tmpOutFile.toStdString(), bgUpperBoundImage_);
            //output ROI to file
            tmpOutFile = tmpOutDir_ + QString("/inROI.png");
            cv::imwrite(tmpOutFile.toStdString(), inROI_);
        }

        printf("Finished computing background model\n");

    }

    // void storeBackgroundModel(cv::Mat& bgMedianImage)
    // store background model in bgMedianImage_
    // use background subtraction threshold to pre-compute lower bound 
    // and upper bound images. 
    // inputs:
    // bgMedianImage: median background image to store
    void FlyTrackPlugin::storeBackgroundModel(cv::Mat& bgMedianImage) {

        bgMedianImage_ = bgMedianImage.clone();
        cv::add(bgMedianImage, backgroundThreshold_, bgUpperBoundImage_);
        cv::subtract(bgMedianImage, backgroundThreshold_, bgLowerBoundImage_);
    }

    void FlyTrackPlugin::reset()
    { 
        isFirst_ = true;
    }

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

    // void backgroundSubtraction()
    // perform background subtraction on currentImage_ and stores results in isFg_
    // use bgLowerBoundImage_, bgUpperBoundImage_ to threshold
    // difference from bgMedianImage_ to determine background/foreground membership.
    // if roiType_ is not NONE, use inROI_ mask to restrict foreground to ROI.
    void FlyTrackPlugin::backgroundSubtraction() {
        // Get background/foreground membership, 255=background, 0=foreground
        switch (flyVsBgMode_) {
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
        if (roiType_ != NONE) {
            cv::bitwise_and(isFg_, inROI_, isFg_);
        }
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
			releaseLock();
			return;
		}

        // Get background/foreground membership, 255=background, 0=foreground
        backgroundSubtraction();

        if (isFirst_){
			isFirst_ = false;
            QString tmpOutFile;
            tmpOutFile = tmpOutDir_ + QString("/isFg.png");
            cv::imwrite(tmpOutFile.toStdString(), isFg_);
		}

        // find connected components in isFg_
        int ccArea = largestConnectedComponent(isFg_);

        // compute mean and covariance of pixels in foreground
        fitEllipse(isFg_, flyEllipse_);
        releaseLock();
    } 


    cv::Mat FlyTrackPlugin::getCurrentImage()
    {
        cv::Mat currentImageCopy;
        acquireLock();
        currentImageCopy = isFg_.clone(); 
        cv::cvtColor(currentImageCopy, currentImageCopy, cv::COLOR_GRAY2BGR);
        // plot fit ellipse
        cv::ellipse(currentImageCopy, cv::Point(flyEllipse_.x, flyEllipse_.y), 
            		cv::Size(flyEllipse_.a, flyEllipse_.b), 
                    flyEllipse_.theta * 180.0 / M_PI, 
                    0, 360, cv::Scalar(0, 0, 255), 2);
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
