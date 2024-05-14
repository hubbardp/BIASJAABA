#include "flytrack_plugin.hpp"
#include <QtDebug>
#include <QMessageBox>
#include <QFileDialog>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "camera_window.hpp"
#include <iostream>
#include "video_utils.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include "mat_to_qimage.hpp"

namespace bias
{

    const QString FlyTrackPlugin::PLUGIN_NAME = QString("FlyTrack"); 
    const QString FlyTrackPlugin::PLUGIN_DISPLAY_NAME = QString("Fly Track");
    const QString FlyTrackPlugin::LOG_FILE_EXTENSION = QString("json");
    const QString FlyTrackPlugin::LOG_FILE_POSTFIX = QString("flytrack");
    const int FlyTrackPlugin::LOGGING_PRECISION = 6;

    const unsigned int FlyTrackPlugin::BG_HIST_NUM_BINS = 256;
    const unsigned int FlyTrackPlugin::BG_HIST_BIN_SIZE = 1;
    const double FlyTrackPlugin::MIN_VEL_MATCH_DOTPROD = 0.25;

    // helper functions

    // void loadBackgroundModel(QString bgImageFilePath, cv::Mat& bgMedianImage)
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
        cv::Mat& bgMedianImage,
        QProgressBar* progressBar) {
        if (bgVideoFilePath.isEmpty()) {
            fprintf(stderr, "No background video file specified\n");
            return;
        }
        else if (!QFile::exists(bgVideoFilePath)) {
			fprintf(stderr, "Background video file %s does not exist\n", bgVideoFilePath.toStdString().c_str());
			return;
		}
        videoBackend vidObj = videoBackend(bgVideoFilePath);
        int nFrames = vidObj.getNumFrames();

        StampedImage newStampedImg;
        newStampedImg.image = vidObj.grabImage();

        BackgroundData_ufmf backgroundData;
        backgroundData = BackgroundData_ufmf(newStampedImg, 
            FlyTrackPlugin::BG_HIST_NUM_BINS, 
            FlyTrackPlugin::BG_HIST_BIN_SIZE);
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
            if ((progressBar != NULL) && (progressBar->isVisible())) {
                progressBar->setValue((100.0 * (f+nFramesSkip) / lastFrameSample));
            }
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

    double mod2pi(double angle) {
        return std::fmod(angle + M_PI,2.0 * M_PI) - M_PI;
	}

    bool checkFileExists(QString file) {
        if (file.isEmpty()) {
			return false;
		}
        return QFile::exists(file);
    }

    QString ellipseToJson(EllipseParams ell) {
		QString json = QString("{");
        json += QString("\"frame\": %1,").arg(ell.frame);
		json += QString("\"x\": %1,").arg(ell.x);
		json += QString("\"y\": %1,").arg(ell.y);
		json += QString("\"a\": %1,").arg(ell.a);
		json += QString("\"b\": %1,").arg(ell.b);
		json += QString("\"theta\": %1").arg(ell.theta);
		json += QString("}");
		return json;
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

        setupUi(this);
        initializeUi();
        connectWidgets();

        // hard code parameters
        // these should go in a config file/GUI

        //// parameters for background subtraction
        //backgroundThreshold_ = 75;
        //flyVsBgMode_ = FLY_DARKER_THAN_BG;

        // parameters for background estimation
        //nFramesBgEst_ = 100;
        //config_.bgVideoFilePath = QString("C:\\Code\\BIAS\\testdata\\20240409T155835_P1_movie1.avi");
        //config_.bgImageFilePath = QString("C:\\Code\\BIAS\\testdata\\20240409T155835_P1_movie1_bg.png");
        //config_.tmpOutDir = QString("C:\\Code\\BIAS\\testdata\\tmp");
        //config_.DEBUG = true;

        // parameters for region of interest
        //config_.roiType = CIRCLE;
        //config_.roiCenterX = 468.6963;
        //config_.roiCenterY = 480.2917;
        //config_.roiRadius = 428.3618;

        imwriteParams_.push_back(cv::IMWRITE_PNG_COMPRESSION);
        imwriteParams_.push_back(0);

        // parameters for resolving head/tail ambiguity
        //historyBufferLength_ = 5;
        //minVelocityMagnitude_ = 1.0; // .05; // could do this in pixels / second since we have timestamps
        //headTailWeightVelocity_ = 3.0; // weight of head-tail dot product vs previous orientation dot product

        bgImageComputed_ = false;
        active_ = false;
        lastFramePreviewed_ = -1;
        flyEllipseDequePtr_ = std::make_shared<LockableDeque<EllipseParams>>();
        initialize();


        setRequireTimer(false);
    }

    void FlyTrackPlugin::reset()
    { 
        initialize();
        openLogFile();
    }

    void FlyTrackPlugin::setFileAutoNamingString(QString autoNamingString)
    {
        fileAutoNamingString_ = autoNamingString;
    }

    void FlyTrackPlugin::setFileVersionNumber(unsigned verNum)
    {
        fileVersionNumber_ = verNum;
    }

    void FlyTrackPlugin::finishComputeBgMode() {
        printf("Computing median image\n");
        fflush(stdout);
        bgMedianImage_ = backgroundData_.getMedianImage();
        lastFrameMedianComputed_ = backgroundData_.getNFrames();
        bgImageComputed_ = true;
        printf("Saving median image to %s\n", config_.bgImageFilePath.toStdString().c_str());
        bool success = cv::imwrite(config_.bgImageFilePath.toStdString(), bgMedianImage_, imwriteParams_);
        if (!success) {
			fprintf(stderr, "Error writing background image to %s\n", config_.bgImageFilePath.toStdString().c_str());
		}
        fflush(stdout);
        FlyTrackConfig config = config_.copy();
        config.nFramesBgEst = nFramesAddedBgEst_;
        config.lastFrameSample = lastFrameAdded_;
        if (loggingEnabled_) {
            config.bgVideoFilePath = getCameraWindow()->getVideoFile();
        }
        else{
            config.bgVideoFilePath = QString("");
		}
        setFromConfig(config);
    }

    void FlyTrackPlugin::stop(){ 
        BiasPlugin::stop();
        closeLogFile();
        if (config_.computeBgMode) {
            finishComputeBgMode();
        }
    }

    void FlyTrackPlugin::setActive(bool value)
    {
        active_ = value;
        // compute background model
        acquireLock();
        if (value && !bgImageComputed_) {
            setBackgroundModel();
        }
        releaseLock();
    }

    void FlyTrackPlugin::processFrames(QList<StampedImage> frameList) {
        acquireLock();
        if (config_.computeBgMode) {
            processFramesBgEstMode(frameList);
        }
        else {
            processFramesTrackMode(frameList);
        }
        releaseLock();
    }


    void FlyTrackPlugin::processFramesTrackMode(QList<StampedImage> frameList)
    { 
        StampedImage latestFrame = frameList.back();
        frameList.clear();
        currentImage_ = latestFrame.image;
        timeStamp_ = latestFrame.timeStamp;
        frameCount_ = latestFrame.frameCount;

        if (!bgImageComputed_) {
            fprintf(stderr, "Background model not computed\n");
			return;
		}
        //printf("\nProcessing frame %lu, timestamp = %f\n", frameCount_, timeStamp_);
        // empty frame
        if ((currentImage_.rows == 0) || (currentImage_.cols == 0))
        {
            fprintf(stderr, "Empty frame\n");
			return;
		}
        // mismatched sizes
        if ((bgMedianImage_.rows != currentImage_.rows) || (bgMedianImage_.cols != currentImage_.cols)
            || bgMedianImage_.type() != currentImage_.type())
        {
            fprintf(stderr, "Background model and current image are not the same size\n");
			return;
		}

        // Get background/foreground membership, 255=background, 0=foreground
        backgroundSubtraction();

        // find connected components in isFg_
        int ccArea = largestConnectedComponent(isFg_);

        // compute mean and covariance of pixels in foreground
        flyEllipse_.frame = frameCount_;
        fitEllipse(isFg_, flyEllipse_);

        // store velocity
        updateVelocityHistory();

        // resolve head/tail ambiguity
        resolveHeadTail();

        // store ellipse
        updateEllipseHistory();

        // store orientation
        updateOrientationHistory();

        if (loggingEnabled_) {
            logCurrentFrame();
        }

        isFirst_ = false;

    } 

    void FlyTrackPlugin::processFramesBgEstMode(QList<StampedImage> frameList) {

        StampedImage latestFrame = frameList.back();
        frameList.clear();
        currentImage_ = latestFrame.image;
        timeStamp_ = latestFrame.timeStamp;
        frameCount_ = latestFrame.frameCount;

        if (isFirst_) {
            backgroundData_ = BackgroundData_ufmf(latestFrame,
                FlyTrackPlugin::BG_HIST_NUM_BINS,
                FlyTrackPlugin::BG_HIST_BIN_SIZE);
            backgroundData_.addImage(latestFrame);
            lastFrameAdded_ = latestFrame.frameCount;
            nFramesAddedBgEst_ = 1;
            isFirst_ = false;
            return;
        }
        if (latestFrame.frameCount <= lastFrameAdded_ + config_.nFramesSkipBgEst) {
			return;
		}
        backgroundData_.addImage(latestFrame);
        lastFrameAdded_ += config_.nFramesSkipBgEst;
        nFramesAddedBgEst_ += 1;

    }

    void FlyTrackPlugin::getCurrentImageTrackMode(cv::Mat& currentImageCopy)
    {
        if (!bgImageComputed_) {
            currentImageCopy = currentImage_.clone();
            return;
		}
        currentImageCopy = isFg_.clone(); 
        cv::cvtColor(currentImageCopy, currentImageCopy, cv::COLOR_GRAY2BGR);
        // plot fit ellipse
        cv::ellipse(currentImageCopy, cv::Point(flyEllipse_.x, flyEllipse_.y), 
            		cv::Size(flyEllipse_.a, flyEllipse_.b), 
                    flyEllipse_.theta * 180.0 / M_PI, 
                    0, 360, cv::Scalar(0, 0, 255), 2);
        cv::Point2d head = cv::Point2d(flyEllipse_.x + flyEllipse_.a * std::cos(flyEllipse_.theta),
            			flyEllipse_.y + flyEllipse_.a * std::sin(flyEllipse_.theta));
        cv::drawMarker(currentImageCopy, head, cv::Scalar(255, 0, 0), cv::MARKER_CROSS, 10, 2);
    }

    void FlyTrackPlugin::getCurrentImageComputeBgMode(cv::Mat& currentImageCopy)
    {
        if (isFirst_) {
            currentImageCopy = currentImage_.clone();
            return;
        }
        if (backgroundData_.getNFrames() == lastFrameMedianComputed_) {
            currentImageCopy = lastImagePreviewed_;
            return;
        }
        currentImageCopy = backgroundData_.getMedianImage().clone();
        lastFrameMedianComputed_ = backgroundData_.getNFrames();
        cv::cvtColor(currentImageCopy, currentImageCopy, cv::COLOR_GRAY2BGR);

        // add text
        std::stringstream statusStream;
        statusStream << "N Frames added: " << nFramesAddedBgEst_ << ", Last frame added: " << lastFrameAdded_;
        double fontScale = 1.0;
        int thickness = 2;
        int baseline = 0;

        //cv::Size textSize = cv::getTextSize(foundStream.str(), CV_FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
        cv::Size textSize = cv::getTextSize(statusStream.str(), cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
        cv::Point textPoint(currentImageCopy.cols / 2 - textSize.width / 2, textSize.height + baseline);
        //cv::putText(currentImageBGR, foundStream.str(), textPoint, CV_FONT_HERSHEY_SIMPLEX, fontScale, boxColor,thickness);
        cv::putText(currentImageCopy, statusStream.str(), textPoint, 
            cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 255),thickness);

    }

    cv::Mat FlyTrackPlugin::getCurrentImage() {
        acquireLock();
        if (frameCount_ == lastFramePreviewed_) {
            releaseLock();
            return lastImagePreviewed_;
        }
        cv::Mat currentImageCopy;
        if (config_.computeBgMode) {
            getCurrentImageComputeBgMode(currentImageCopy);
        }
        else {
            getCurrentImageTrackMode(currentImageCopy);
        }
        lastFramePreviewed_ = frameCount_;
        lastImagePreviewed_ = currentImageCopy;
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


    RtnStatus FlyTrackPlugin::runCmdFromMap(QVariantMap cmdMap, bool showErrorDlg, QString& value)
    {
        RtnStatus rtnStatus;
        rtnStatus.success = true;
        rtnStatus.message = QString("");

        QString errMsgTitle("Plugin runCmdFromMap Error");

        if (!cmdMap.contains("cmd"))
        {
            QString errMsgText("FlyTrackPlugin::runPluginCmd: cmd not found in map");
            if (showErrorDlg)
            {
                QMessageBox::critical(this, errMsgTitle, errMsgText);
            }
            rtnStatus.success = false;
            rtnStatus.message = errMsgText;
            return rtnStatus;
        }
        if (!cmdMap["cmd"].canConvert<QString>())
        {
            QString errMsgText("FlyTrackPlugin::runPluginCmd: unable to convert plugin name to string");
            if (showErrorDlg)
            {
                QMessageBox::critical(this, errMsgTitle, errMsgText);
            }
            rtnStatus.success = false;
            rtnStatus.message = errMsgText;
            return rtnStatus;
        }
        QString cmd = cmdMap["cmd"].toString();

        if (cmd == QString("pop-front-track"))
        {
            EllipseParams ell;
            rtnStatus = popFrontTrack(ell);
            if (rtnStatus.success) {
				value = ellipseToJson(ell);
            }
        }
        else if (cmd == QString("pop-back-track"))
        {
			EllipseParams ell;
			rtnStatus = popBackTrack(ell);
			if (rtnStatus.success) {
                value = ellipseToJson(ell);
            }
        }
        else
        {
            QString errMsgText = QString("FlyTrackPlugin::runPluginCmd: unknown cmd %1").arg(cmd);
            if (showErrorDlg)
            {
                QMessageBox::critical(this, errMsgTitle, errMsgText);
            }
            rtnStatus.success = false;
            rtnStatus.message = errMsgText;
        }

        return rtnStatus;
    }

    RtnStatus FlyTrackPlugin::popFrontTrack(EllipseParams& ell) {
        RtnStatus rtnStatus;
		rtnStatus.success = false;
		rtnStatus.message = QString("");
        if (flyEllipseDequePtr_ == NULL) {
            rtnStatus.message = QString("Ellipse queue not allocated");
            return rtnStatus;
        }
        flyEllipseDequePtr_->acquireLock();
        if (flyEllipseDequePtr_->empty()) {
			rtnStatus.message = QString("Ellipse queue empty");
		}
        else {
            ell = flyEllipseDequePtr_->front();
            flyEllipseDequePtr_->pop_front();
            rtnStatus.success = true;
        }
        flyEllipseDequePtr_->releaseLock();
		return rtnStatus;
	
    }

    RtnStatus FlyTrackPlugin::popBackTrack(EllipseParams& ell) {
        RtnStatus rtnStatus;
        rtnStatus.success = false;
        rtnStatus.message = QString("");
        if (flyEllipseDequePtr_ == NULL) {
            rtnStatus.message = QString("Ellipse queue not allocated");
            return rtnStatus;
        }
        flyEllipseDequePtr_->acquireLock();
        if (flyEllipseDequePtr_->empty()) {
            rtnStatus.message = QString("Ellipse queue empty");
        }
        else {
            ell = flyEllipseDequePtr_->back();
            flyEllipseDequePtr_->pop_back();
            rtnStatus.success = true;
        }
        flyEllipseDequePtr_->releaseLock();
        return rtnStatus;

    }

    QVariantMap FlyTrackPlugin::getConfigAsMap()  
    {
        QVariantMap configMap = config_.toMap();
        return configMap;
    }

    void FlyTrackPlugin::setRoiUIValues() {
        roiTypeComboBox->setCurrentIndex(config_.roiType);
        roiCenterXSpinBox->setValue(config_.roiCenterX);
        roiCenterYSpinBox->setValue(config_.roiCenterY);
        roiRadiusSpinBox->setValue(config_.roiRadius);

        if(config_.roiType == NONE) {
            roiCenterXSpinBox->setEnabled(false);
			roiCenterYSpinBox->setEnabled(false);
			roiRadiusSpinBox->setEnabled(false);
		} else {
            roiCenterXSpinBox->setEnabled(true);
			roiCenterYSpinBox->setEnabled(true);
			roiRadiusSpinBox->setEnabled(true);
		}

    }

    void FlyTrackPlugin::connectWidgets()
    {
        connect(
            donePushButton,
            SIGNAL(clicked()),
            this,
            SLOT(donePushButtonClicked())
        );

        connect(
            applyPushButton,
            SIGNAL(clicked()),
            this,
            SLOT(applyPushButtonClicked())
        );

        connect(
            cancelPushButton,
            SIGNAL(clicked()),
            this,
            SLOT(cancelPushButtonClicked())
        );

        connect(
            loadBgPushButton,
            SIGNAL(clicked()),
            this,
            SLOT(loadBgPushButtonClicked())
        );

        connect(
            roiCenterXSpinBox,
            SIGNAL(valueChanged(int)),
            this,
            SLOT(roiUiChanged(int))
        );
        connect(
            roiCenterYSpinBox,
            SIGNAL(valueChanged(int)),
            this,
            SLOT(roiUiChanged(int))
        );
        connect(
            roiRadiusSpinBox,
            SIGNAL(valueChanged(int)),
            this,
            SLOT(roiUiChanged(int))
        );
        connect(
            roiTypeComboBox,
            SIGNAL(activated(int)),
            this,
            SLOT(roiUiChanged(int))
        );
        connect(
            bgImageFilePathToolButton,
            SIGNAL(clicked()),
            this,
            SLOT(bgImageFilePathToolButtonClicked())
        );
        connect(
            bgVideoFilePathToolButton,
            SIGNAL(clicked()),
            this,
            SLOT(bgVideoFilePathToolButtonClicked())
        );
        connect(
            logFilePathToolButton,
            SIGNAL(clicked()),
            this,
            SLOT(logFilePathToolButtonClicked())
        );
        connect(
			tmpOutDirToolButton,
			SIGNAL(clicked()),
			this,
			SLOT(tmpOutDirToolButtonClicked())
		);
        connect(
            computeBgModeComboBox,
            SIGNAL(activated(int)),
            this,
            SLOT(computeBgModeComboBoxChanged())
        );
    }

    void FlyTrackPlugin::showEvent(QShowEvent* event) {
        QWidget::showEvent(event);
        setFromConfig(config_);
    }

    void FlyTrackPlugin::donePushButtonClicked() {
        try {
            applyPushButtonClicked();
            // close the dialog
            close();
        }
        catch (std::exception& e) {
            fflush(stdout);
			fprintf(stderr, "Error closing dialog: %s\n", e.what());
		}
    }
    void FlyTrackPlugin::cancelPushButtonClicked() {
        try {
            close();
        }
        catch (std::exception& e) {
            fflush(stdout);
            fprintf(stderr, "Error closing dialog: %s\n", e.what());
        }
    }

    void FlyTrackPlugin::applyPushButtonClicked() {
        FlyTrackConfig config;
        RtnStatus rtnStatus;
        rtnStatus.success = true;
        rtnStatus.message = QString("");
        getUiValues(config);
        if (rtnStatus.success) {
			rtnStatus = setFromConfig(config);
            if (rtnStatus.success) {
            }
            else{
				QMessageBox::critical(this, QString("Error setting config values"), rtnStatus.message);
            }
		}
        else {
            QMessageBox::critical(this, QString("Error getting config values"), rtnStatus.message);
		}
        fflush(stdout);
    }

    void FlyTrackPlugin::bgImageFilePathToolButtonClicked() {
        QString bgImageFilePath = bgImageFilePathLineEdit->text();
        QString bgImageDir = QFileInfo(bgImageFilePath).absoluteDir().absolutePath();
		bgImageFilePath = QFileDialog::getSaveFileName(this, "Select Background Image File", 
            bgImageDir, "Image Files (*.png *.jpg *.bmp)",NULL,QFileDialog::DontConfirmOverwrite);
        if (bgImageFilePath.isEmpty()) {
            fprintf(stderr,"No background image selected\n");
            return;
        }
	    bgImageFilePathLineEdit->setText(bgImageFilePath);
    }

    void FlyTrackPlugin::bgVideoFilePathToolButtonClicked() {
        QString bgVideoFilePath = bgVideoFilePathLineEdit->text();
        QString bgVideoDir = QFileInfo(bgVideoFilePath).absoluteDir().absolutePath();
        bgVideoFilePath = QFileDialog::getOpenFileName(this, "Select Video to compute background from",
            bgVideoDir, "Video Files (*.avi *.ufmf *.fmf *mp4)");
        if (bgVideoFilePath.isEmpty()) {
            fprintf(stderr,"No background video selected\n");
            //return;
        }
        bgVideoFilePathLineEdit->setText(bgVideoFilePath);
    }

    void FlyTrackPlugin::logFilePathToolButtonClicked() {
        QString logFilePath = logFilePathLineEdit->text();
		QString logFileDir = QFileInfo(logFilePath).absoluteDir().absolutePath();
		logFilePath = QFileDialog::getSaveFileName(this, "Output track file", logFileDir, "JSON Files (*." + LOG_FILE_EXTENSION + ")");
        if (logFilePath.isEmpty()) {
			fprintf(stderr,"No output file selected\n");
			return;
		}
		logFilePathLineEdit->setText(logFilePath);
    }

    void FlyTrackPlugin::tmpOutDirToolButtonClicked() {
        QString tmpOutDir = tmpOutDirLineEdit->text();
        tmpOutDir = QFileDialog::getExistingDirectory(this, "Debug output folder", tmpOutDir);
        if (tmpOutDir.isEmpty()) {
            fprintf(stderr,"No output directory selected\n");
            return;
        }
        tmpOutDirLineEdit->setText(tmpOutDir);
    }

    void FlyTrackPlugin::computeBgModeComboBoxChanged() {
        setUiEnabled();
	}

    void FlyTrackPlugin::roiUiChanged(int v) {
        FlyTrackConfig roiConfig = config_.copy();
        getUiRoiValues(roiConfig);
        setPreviewImage(bgMedianImage_, roiConfig);
        setUiEnabled();
    }

    void FlyTrackPlugin::loadBgPushButtonClicked() {
        FlyTrackConfig bgEstConfig = config_.copy();
        getUiBgEstValues(bgEstConfig);
        if (! (checkFileExists(bgEstConfig.bgImageFilePath) || checkFileExists(bgEstConfig.bgVideoFilePath))) {
			QMessageBox::critical(this, QString("Error loading background model"), 
                				QString("Neither background image or video file paths exist."));
			return;
		}
        progressBar->setValue(0);
        progressBar->setVisible(true);
        setBgEstParams(bgEstConfig);
        if (!bgImageComputed_) {
			setBackgroundModel();
		}
        progressBar->setValue(100);
    }

    RtnStatus FlyTrackPlugin::setFromConfig(FlyTrackConfig config)
	{
		RtnStatus rtnStatus;
        rtnStatus.success = true;
        rtnStatus.message = QString("");

        //printf("Setting config:\n");

        setBgEstParams(config);
        config_ = config;
        setROI(config);
        setROI(config);

        if (config_.computeBgMode) {
            computeBgModeComboBox->setCurrentIndex(0);
		}
        else {
            computeBgModeComboBox->setCurrentIndex(1);
        }
        bgImageFilePathLineEdit->setText(config_.bgImageFilePath);
        bgVideoFilePathLineEdit->setText(config_.bgVideoFilePath);
        nFramesBgEstLineEdit->setText(QString::number(config_.nFramesBgEst));
        nFramesSkipLineEdit->setText(QString::number(config_.nFramesSkipBgEst));
        lastFrameSampleLineEdit->setText(QString::number(config_.lastFrameSample));
        progressBar->setVisible(false);
        flyVsBgModeComboBox->setCurrentIndex(config_.flyVsBgMode);
        backgroundThresholdLineEdit->setText(QString::number(config_.backgroundThreshold));
        setRoiUIValues();
        historyBufferLengthSpinBox->setValue(config_.historyBufferLength);
        minVelocityMagnitudeLineEdit->setText(QString::number(config_.minVelocityMagnitude));
        headTailWeightVelocityLineEdit->setText(QString::number(config_.headTailWeightVelocity));
        logFilePathLineEdit->setText(config_.tmpTrackFilePath);
        logFileNameLineEdit->setText(config_.trackFileName);
        tmpOutDirLineEdit->setText(config_.tmpOutDir);
        DEBUGCheckBox->setChecked(config_.DEBUG);

        //config_.print();
        setUiEnabled();

		return rtnStatus;
	}

    bool FlyTrackPlugin::saveBgMedianImage(cv::Mat bgMedianImage, QString bgImageFilePath) {
        printf("Saving median image to %s\n", bgImageFilePath.toStdString().c_str());
        bool success = cv::imwrite(bgImageFilePath.toStdString(), bgMedianImage, imwriteParams_);
        if (success) printf("Done\n");
        else fprintf(stderr,"Failed to write background median image to %s\n", bgImageFilePath.toStdString().c_str());
		return success;
    }

    void FlyTrackPlugin::setBgEstParams(FlyTrackConfig& configNew) {
        QString bgImageFilePathPrev = config_.bgImageFilePath;
        QString bgVideoFilePathPrev = config_.bgVideoFilePath;
        int nFramesBgEstPrev = config_.nFramesBgEst;
        int lastFrameSamplePrev = config_.lastFrameSample;
        config_.bgImageFilePath = configNew.bgImageFilePath;
        config_.bgVideoFilePath = configNew.bgVideoFilePath;
        config_.nFramesBgEst = configNew.nFramesBgEst;
        config_.lastFrameSample = configNew.lastFrameSample;
        printf("Setting bg file paths:\n");
        printf("Image: %s -> %s\n", bgImageFilePathPrev.toStdString().c_str(), configNew.bgImageFilePath.toStdString().c_str());  
        printf("Video: %s -> %s\n", bgVideoFilePathPrev.toStdString().c_str(), configNew.bgVideoFilePath.toStdString().c_str());
        if (QFile::exists(configNew.bgImageFilePath)) {
            if ((configNew.bgImageFilePath != bgImageFilePathPrev) || !bgImageComputed_) {
                printf("Loading background image from %s\n", configNew.bgImageFilePath.toStdString().c_str());
                cv::Mat bgMedianImage;
                loadBackgroundModel(configNew.bgImageFilePath, bgMedianImage);
                storeBackgroundModel(bgMedianImage,configNew);
                bgImageComputed_ = true;
            }
            else {
                printf("Image file path matches and bgImageComputed_ is true, nothing to do\n");
            }
            // if bgImageFilePath matches and bgImageComputed_ is true, do nothing
        }
        else {
            if (bgImageComputed_ && 
                (configNew.bgVideoFilePath == bgVideoFilePathPrev) && 
                (configNew.nFramesBgEst == nFramesBgEstPrev) && 
                (configNew.lastFrameSample == lastFrameSamplePrev)) {
                // if the video file hasn't changed and the median has been computed
                // just save the image
                printf("Saving background image to %s\n", configNew.bgImageFilePath.toStdString().c_str());
                saveBgMedianImage(bgMedianImage_, configNew.bgImageFilePath);
            }
            else {
                printf("Setting bgImageComputed_ to false\n");
                bgImageComputed_ = false;
            }
        }
    }

    void FlyTrackPlugin::getUiRoiValues(FlyTrackConfig& config) {
        ROIType roiType = (ROIType)roiTypeComboBox->currentIndex();
        double roiCenterX = roiCenterXSpinBox->value();
        double roiCenterY = roiCenterYSpinBox->value();
        double roiRadius = roiRadiusSpinBox->value();
        config.setRoiParams(roiType, roiCenterX, roiCenterY, roiRadius);
    }

    void FlyTrackPlugin::getUiBgEstValues(FlyTrackConfig& config) {
        config.computeBgMode = computeBgModeComboBox->currentIndex() == 0;
        config.bgImageFilePath = bgImageFilePathLineEdit->text();
        config.bgVideoFilePath = bgVideoFilePathLineEdit->text();
        config.nFramesBgEst = nFramesBgEstLineEdit->text().toInt();
        config.nFramesSkipBgEst = nFramesSkipLineEdit->text().toInt();
        config.lastFrameSample = lastFrameSampleLineEdit->text().toInt();

    }

    void FlyTrackPlugin::getUiValues(FlyTrackConfig& config) {
        try {
            getUiBgEstValues(config);
            config.flyVsBgMode = (FlyVsBgModeType)flyVsBgModeComboBox->currentIndex();
            config.backgroundThreshold = backgroundThresholdLineEdit->text().toInt();
            getUiRoiValues(config);
            config.historyBufferLength = historyBufferLengthSpinBox->value();
            config.minVelocityMagnitude = minVelocityMagnitudeLineEdit->text().toDouble();
            config.headTailWeightVelocity = headTailWeightVelocityLineEdit->text().toDouble();
            config.tmpOutDir = tmpOutDirLineEdit->text();
            config.DEBUG = DEBUGCheckBox->isChecked();
            config.tmpTrackFilePath = logFilePathLineEdit->text();
            config.trackFileName = logFileNameLineEdit->text();
        }
        catch (std::exception& e) {
            fflush(stdout);
			fprintf(stderr,"Error getting UI values: %s\n", e.what());
		}
	}

    void FlyTrackPlugin::setUiEnabled() {
        FlyTrackConfig config;
        getUiValues(config);
        if (config.computeBgMode) {
            bgImageFilePathLineEdit->setEnabled(true);
            bgImageFilePathLabel->setEnabled(true);
			bgVideoFilePathLineEdit->setEnabled(false);
            bgVideoFilePathLabel->setEnabled(false);
			nFramesBgEstLineEdit->setEnabled(false);
            nFramesBgEstLabel->setEnabled(false);
			nFramesSkipLineEdit->setEnabled(true);
            nFramesSkipLabel->setEnabled(true);
			lastFrameSampleLineEdit->setEnabled(false);
            lastFrameSampleLabel->setEnabled(false);
			loadBgPushButton->setEnabled(false);
            progressBar->setEnabled(false);
		}
        else {
			bgImageFilePathLineEdit->setEnabled(true);
            bgImageFilePathLabel->setEnabled(true);
			bgVideoFilePathLineEdit->setEnabled(true);
            bgVideoFilePathLabel->setEnabled(true);
			nFramesBgEstLineEdit->setEnabled(true);
            nFramesBgEstLabel->setEnabled(true);
			nFramesSkipLineEdit->setEnabled(false);
            nFramesSkipLabel->setEnabled(false);
			lastFrameSampleLineEdit->setEnabled(true);
            lastFrameSampleLabel->setEnabled(true);
			loadBgPushButton->setEnabled(true);
        }
        flyVsBgModeComboBox->setEnabled(true);
        flyVsBgModeLabel->setEnabled(true);
        backgroundThresholdLineEdit->setEnabled(true);
        backgroundThresholdLabel->setEnabled(true);
        roiTypeComboBox->setEnabled(true);
        roiTypeLabel->setEnabled(true);
        switch (config.roiType) {
			case NONE:
				roiCenterXSpinBox->setEnabled(false);
                roiCenterXLabel->setEnabled(false);
				roiCenterYSpinBox->setEnabled(false);
                roiCenterYLabel->setEnabled(false);
				roiRadiusSpinBox->setEnabled(false);
                roiRadiusLabel->setEnabled(false);
				break;
			case CIRCLE:
				roiCenterXSpinBox->setEnabled(true);
                roiCenterXLabel->setEnabled(true);
				roiCenterYSpinBox->setEnabled(true);
                roiCenterYLabel->setEnabled(true);
				roiRadiusSpinBox->setEnabled(true);
                roiRadiusLabel->setEnabled(true);
				break;
		}
        historyBufferLengthSpinBox->setEnabled(true);
        historyBufferLengthLabel->setEnabled(true);
		minVelocityMagnitudeLineEdit->setEnabled(true);
        minVelocityMagnitudeLabel->setEnabled(true);
		headTailWeightVelocityLineEdit->setEnabled(true);
        headTailWeightVelocityLabel->setEnabled(true);
		logFilePathLineEdit->setEnabled(true);
        logFilePathLabel->setEnabled(true);
		tmpOutDirLineEdit->setEnabled(true);
        tmpOutDirLabel->setEnabled(true);
		DEBUGCheckBox->setEnabled(true);
    }

	RtnStatus FlyTrackPlugin::setConfigFromMap(QVariantMap configMap)
	{
		FlyTrackConfig config;
		RtnStatus rtnStatus = config.fromMap(configMap);
        if (rtnStatus.success)
        {
			rtnStatus = setFromConfig(config);
		}
		return rtnStatus;
	}

	RtnStatus FlyTrackPlugin::setConfigFromJson(QByteArray jsonArray)
	{
		FlyTrackConfig config;
		RtnStatus rtnStatus = config.fromJson(jsonArray);
        if (rtnStatus.success)
        {
			rtnStatus = setFromConfig(config);
		}
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
        QString logFileName;
        if (config_.trackFileNameSet()) {
            logFileName = config_.trackFileName;
        }
        else{
            QPointer<CameraWindow> cameraWindowPtr = getCameraWindow();
            logFileName = cameraWindowPtr->getVideoFileName() + QString("_") + getLogFilePostfix();
        }
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
        if (config_.trackFilePathSet()) {
            return config_.tmpTrackFilePath;
        }
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
        if ((config_.computeBgMode == false) && loggingEnabled_)
        {
            QString logFileFullPath = getLogFileFullPath(true);
            qDebug() << logFileFullPath;
            fprintf(stderr,"Outputting trajectory to file: %s",logFileFullPath.toStdString().c_str());
            logFile_.setFileName(logFileFullPath);
            bool isOpen = logFile_.open(QIODevice::WriteOnly | QIODevice::Text);
            if (isOpen)
            {
                logStream_.setDevice(&logFile_);
                logStream_.setRealNumberNotation(QTextStream::ScientificNotation);
                logStream_.setRealNumberPrecision(FlyTrackPlugin::LOGGING_PRECISION);
                logStream_ << "{\n  \"track\": [\n";
            }
            else
            {
				fprintf(stderr,"Failed to open log file: %s\n",logFileFullPath.toStdString().c_str());
                loggingEnabled_ = false;
			}
        }
    }

    void FlyTrackPlugin::closeLogFile()
    {
        if (loggingEnabled_ && (config_.computeBgMode == false) && logFile_.isOpen())
        {
            logStream_ << "\n  ]\n}";
            logStream_.flush();
            logFile_.close();
        }
    }

    // Protected
    // ------------------------------------------------------------------------

    // void initialize()
    // (re-)initialize state
    void FlyTrackPlugin::initialize() {
        isFirst_ = true;
        meanFlyVelocity_ = cv::Point2d(0.0, 0.0);
        meanFlyOrientation_ = 0.0;
        flyEllipseHistory_.clear();
        flyEllipseDequePtr_->acquireLock();
        flyEllipseDequePtr_->clear();
        flyEllipseDequePtr_->releaseLock();
        velocityHistory_.clear();
        orientationHistory_.clear();
        headTailResolved_ = false;

        setFromConfig(config_);
    }

    void FlyTrackPlugin::initializeUi() {

        // set items in ROI combobox to match order of enum
        roiTypeComboBox->clear();
        QString s;
        for(int i=0; i<N_ROI_TYPES; i++){
			roiTypeToString((ROIType)i, s);
			roiTypeComboBox->addItem(s, i);
		}
        // set items in flyVsBgMode combobox to match order of enum
        flyVsBgModeComboBox->clear();
        for (int i = 0; i < N_FLY_VS_BG_MODES; i++) {
            flyVsBgModeToString((FlyVsBgModeType)i, s);
            flyVsBgModeComboBox->addItem(s, i);
        }

        previewImageLabel->setBackgroundRole(QPalette::Base);
        previewImageLabel->setScaledContents(true);

    }


    // cv::Mat circleROI(double centerX, double centerY, double centerRadius)
    // create a circular region of interest mask, inside is 255, outside 0
    // inputs:
    // centerX, centerY: center of circle
    // centerRadius: radius of circle
    // returns: mask image
    cv::Mat FlyTrackPlugin::circleROI(double centerX, double centerY, double centerRadius) {
        cv::Mat mask = cv::Mat::zeros(bgMedianImage_.size(), CV_8U);
        cv::circle(mask, cv::Point(centerX, centerY), centerRadius, cv::Scalar(255), -1);
        return mask;
    }

    // void setROI()
    // set the region of interest mask based on roiType_
    // currently only circle implemented
    void FlyTrackPlugin::setROI(FlyTrackConfig config) {
        printf("setting ROI\n");
        // roi mask
        switch (config.roiType) {
        case CIRCLE:
            printf("setting circle ROI: center %f, %f, radius %f\n", config.roiCenterX, config.roiCenterY, config.roiRadius);
            inROI_ = circleROI(config.roiCenterX, config.roiCenterY, config.roiRadius);
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

        printf("Computing background model\n");

        cv::Mat bgMedianImage;
        // check if bgImageFilePath_ exists
        if (QFile::exists(config_.bgImageFilePath)) {
            loadBackgroundModel(config_.bgImageFilePath, bgMedianImage);
        }
        else {
            if (!checkFileExists(config_.bgVideoFilePath)) {
				printf("Background video file %s does not exist\n", config_.bgVideoFilePath.toStdString().c_str());
				return;
			}
            computeBackgroundMedian(config_.bgVideoFilePath, config_.nFramesBgEst, 
                config_.lastFrameSample, bgMedianImage, progressBar);
            // save the median image
            printf("Saving median image to %s\n", config_.bgImageFilePath.toStdString().c_str());
            bool success = cv::imwrite(config_.bgImageFilePath.toStdString(), bgMedianImage, imwriteParams_);
            if (success) printf("Done\n");
            else printf("Failed to write background median image to %s\n", config_.bgImageFilePath.toStdString().c_str());
        }

        // store background model
        storeBackgroundModel(bgMedianImage,config_);

        bgImageComputed_ = true;

        //output lower bound to file
        if (config_.DEBUG) {
            printf("Outputting background model debug images\n");
            bool success;
            QString tmpOutFile;
            tmpOutFile = config_.tmpOutDir + QString("\\bgLowerBound.png");
            printf("Writing lower bound to %s\n", tmpOutFile.toStdString().c_str());
            success = cv::imwrite(tmpOutFile.toStdString(), bgLowerBoundImage_, imwriteParams_);
            if(!success) printf("Failed writing lower bound to %s\n", tmpOutFile.toStdString().c_str());
            //output upper bound to file
            printf("Writing upper bound to %s\n", tmpOutFile.toStdString().c_str());
            tmpOutFile = config_.tmpOutDir + QString("\\bgUpperBound.png");
            success = cv::imwrite(tmpOutFile.toStdString(), bgUpperBoundImage_, imwriteParams_);
            if (!success) printf("Failed writing upper bound to %s\n", tmpOutFile.toStdString().c_str());
        }

        printf("Finished computing background model\n");

    }

    // void storeBackgroundModel(cv::Mat& bgMedianImage)
    // store background model in bgMedianImage_
    // use background subtraction threshold to pre-compute lower bound 
    // and upper bound images. 
    // inputs:
    // bgMedianImage: median background image to store
    void FlyTrackPlugin::storeBackgroundModel(cv::Mat& bgMedianImage, FlyTrackConfig& config) {

        printf("Storing background model\n");
        bgMedianImage_ = bgMedianImage.clone();
        cv::add(bgMedianImage, config_.backgroundThreshold, bgUpperBoundImage_);
        cv::subtract(bgMedianImage, config_.backgroundThreshold, bgLowerBoundImage_);
        roiCenterXSpinBox->setRange(0, bgMedianImage.cols);
        roiCenterYSpinBox->setRange(0, bgMedianImage.rows);
        roiRadiusSpinBox->setRange(0, std::max(bgMedianImage.cols,bgMedianImage.rows));

        // roi mask
        setROI(config);

        setPreviewImage(bgMedianImage_,config);
        printf("Done\n");
    }

    void FlyTrackPlugin::setPreviewImage(cv::Mat matImage,FlyTrackConfig config)
    {
        if (matImage.empty()) {
			fprintf(stderr,"preview image is empty\n");
			return;
		}

        cv::Mat colorMatImage = matImage.clone();
        cv::cvtColor(colorMatImage, colorMatImage, cv::COLOR_GRAY2BGR);
        switch (config.roiType) {
            case CIRCLE:
                cv::circle(colorMatImage, cv::Point(config.roiCenterX, config.roiCenterY), config.roiRadius, cv::Scalar(0, 0, 255), 2);
				break;
        }

		QImage img = matToQImage(colorMatImage);
        if (img.isNull()) {
            fprintf(stderr,"preview image is null\n");
            return;
        }
        QPixmap pixmapOriginal = QPixmap::fromImage(img);
        QPixmap pixmapScaled = pixmapOriginal.scaled(previewImageLabel->size(), 
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation);
		previewImageLabel->setPixmap(pixmapScaled);
	}

    // void backgroundSubtraction()
    // perform background subtraction on currentImage_ and stores results in isFg_
    // use bgLowerBoundImage_, bgUpperBoundImage_ to threshold
    // difference from bgMedianImage_ to determine background/foreground membership.
    // if roiType_ is not NONE, use inROI_ mask to restrict foreground to ROI.
    // lock must be acquired outside of this function
    void FlyTrackPlugin::backgroundSubtraction() {
        // Get background/foreground membership, 255=background, 0=foreground
        switch (config_.flyVsBgMode) {
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
        if (config_.roiType != NONE) {
            cv::bitwise_and(isFg_, inROI_, isFg_);
        }
        if (config_.DEBUG && isFirst_) {
            printf("Outputting background subtraction debug images\n");
			QString tmpOutFile;
            bool success;
            cv::Mat dBkgd;
            cv::absdiff(currentImage_, bgMedianImage_, dBkgd);
            tmpOutFile = config_.tmpOutDir + QString("\\dBkgd.png");
            printf("Writing difference from background to %s\n", tmpOutFile.toStdString().c_str());
            success = cv::imwrite(tmpOutFile.toStdString(), dBkgd, imwriteParams_);
            if (!success) printf("Failed writing difference from background to %s\n", tmpOutFile.toStdString().c_str());
            tmpOutFile = config_.tmpOutDir + QString("\\isFg.png");
            printf("Writing foreground mask to %s\n", tmpOutFile.toStdString().c_str());
            success = cv::imwrite(tmpOutFile.toStdString(), isFg_);
            if (!success) printf("Failed writing foreground mask to %s\n", tmpOutFile.toStdString().c_str());
            if (config_.roiType != NONE) {
				tmpOutFile = config_.tmpOutDir + QString("\\inROI.png");
                printf("Writing ROI mask to %s\n", tmpOutFile.toStdString().c_str());
				success = cv::imwrite(tmpOutFile.toStdString(), inROI_);
                if (!success) printf("Failed writing ROI mask to %s\n", tmpOutFile.toStdString().c_str());
            }
        }
    }

    // void updateVelocityHistory()
    // update velocity history buffer velocityHistory_ and mean velocity meanFlyVelocity_ over that buffer
    // with velocity between current flyEllipse_ and previous center flyEllipseHistory_.back()
    void FlyTrackPlugin::updateVelocityHistory() {

        if (flyEllipseHistory_.size() == 0)
            return;

        double nHistory;
        // update velocity history
        cv::Point2d velocityLast;
        // compute velocity of center between current ellipse and last ellipse
        velocityLast = cv::Point2d(flyEllipse_.x - flyEllipseHistory_.back().x,
            flyEllipse_.y - flyEllipseHistory_.back().y);

        // update mean velocity for adding velocityLast
        nHistory = (double)velocityHistory_.size();
        meanFlyVelocity_ = (meanFlyVelocity_ * nHistory + velocityLast) / (nHistory + 1.0);

        // add to velocity history
        velocityHistory_.push_back(velocityLast);
        nHistory = nHistory + 1.0;

        // if we are removing from buffer, update mean velocity
        if (velocityHistory_.size() > config_.historyBufferLength) {
            meanFlyVelocity_ = (meanFlyVelocity_ * nHistory - velocityHistory_.front()) / (nHistory - 1);
            velocityHistory_.pop_front();
        }
    }

    // void updateEllipseHistory()
    // add current flyEllipse_ to end of flyEllipseHistory_
    void FlyTrackPlugin::updateEllipseHistory() {
        // add ellipse to history
        flyEllipseHistory_.push_back(flyEllipse_);
        flyEllipseDequePtr_->acquireLock();
        flyEllipseDequePtr_->push_back(flyEllipse_);
        flyEllipseDequePtr_->releaseLock();
    }

    // void updateOrientationHistory()
    // update orientation history buffer orientationHistory_ and mean orientation meanFlyOrientation_ over that buffer
    // orientations will be stored so that they are in the same range of 2*pi
    void FlyTrackPlugin::updateOrientationHistory() {
        double nHistory;
        // update orientation history
        double currOrientation = flyEllipse_.theta;
        if (orientationHistory_.size() > 0) {
            // make orientations in same range of 2*pi
            double prevOrientation = orientationHistory_.back();
            // compute orientation change
            double orientationChange = mod2pi(currOrientation - prevOrientation);
            // this could become way out of the range -pi, pi if we run for a really long time
            currOrientation = prevOrientation + orientationChange;
        }
        // add to orientation history
        orientationHistory_.push_back(currOrientation);
        // update mean orientation for adding currOrientation
        nHistory = (double)orientationHistory_.size();
        meanFlyOrientation_ = (meanFlyOrientation_ * (nHistory - 1) + currOrientation) / nHistory;
        // if we are removing from buffer, update mean orientation
        if (orientationHistory_.size() > config_.historyBufferLength) {
            meanFlyOrientation_ = (meanFlyOrientation_ * nHistory - orientationHistory_.front()) / (nHistory - 1);
            orientationHistory_.pop_front();
        }
    }

    // void flipFlyOrientationHistory()
    // flip all orientations in orientationHistory_ and the mean meanFlyOrientation_ by adding pi
    void FlyTrackPlugin::flipFlyOrientationHistory() {
        meanFlyOrientation_ = meanFlyOrientation_ + M_PI;
        for (int i = 0; i < orientationHistory_.size(); i++) {
            orientationHistory_[i] += M_PI;
        }
    }

    // void resolveHeadTail()
    // resolve head/tail ambiguity by comparing orientation flyEllipse_.theta
    // to velocity meanFlyVelocity_ and past orientation meanFlyOrientation_
    // flyEllipse_.theta is updated 
    void FlyTrackPlugin::resolveHeadTail() {

        double velmag = 0.0;
        double dotprod;
        double costVel0 = 0.0, costVel1 = 0.0;
        double costOri0 = 0.0, costOri1 = 0.0;
        double cost0 = 0.0, cost1 = 0.0;
        double theta0 = flyEllipse_.theta;
        cv::Point2d headDir = cv::Point2d(std::cos(flyEllipse_.theta), std::sin(flyEllipse_.theta));
        cv::Point2d headDirPrev = cv::Point2d(0.0, 0.0);

        // velocity magnitude
        if (velocityHistory_.size() > 0) velmag = cv::norm(meanFlyVelocity_);

        // if fly is walking fast enough, try to match the velocity direction
        if (velmag > config_.minVelocityMagnitude) {
            dotprod = headDir.dot(meanFlyVelocity_) / velmag;
            costVel1 = dotprod;
            costVel0 = -dotprod;
            // if we haven't ever resolved headTail, we don't care about orientation history
            if (!headTailResolved_ && std::abs(dotprod) > MIN_VEL_MATCH_DOTPROD) {
                if (costVel1 < costVel0) {
                    // add pi
                    flyEllipse_.theta += M_PI;
                    flipFlyOrientationHistory();
                }
                headTailResolved_ = true;
                return;
            }
        }

        // try to match current and previous orientation
        if (orientationHistory_.size() > 0) {
            headDirPrev.x = std::cos(meanFlyOrientation_);
            headDirPrev.y = std::sin(meanFlyOrientation_);
            dotprod = headDir.dot(headDirPrev);
            costOri1 = dotprod;
            costOri0 = -dotprod;
        }

        cost0 = config_.headTailWeightVelocity * costVel0 + costOri0;
        cost1 = config_.headTailWeightVelocity * costVel1 + costOri1;
        //printf("Total cost0: %f, cost1: %f\n", cost0, cost1);

        if (cost1 < cost0) {
            // add pi
            flyEllipse_.theta += M_PI;
        }

        // store theta in range -pi, pi
        flyEllipse_.theta = mod2pi(flyEllipse_.theta);
    }

    void FlyTrackPlugin::logCurrentFrame(){
        if (!loggingEnabled_) return;
        if (!logFile_.isOpen()) return;
        if (!isFirst_) logStream_ << ",\n";
        logStream_ << ellipseToJson(flyEllipse_);
    }
}
