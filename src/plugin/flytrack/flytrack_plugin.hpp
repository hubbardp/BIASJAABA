#ifndef FLYTRACK_PLUGIN_HPP
#define FLYTRACK_PLUGIN_HPP
#include <QDialog>
#include <QWidget>
#include <QList>
#include "bias_plugin.hpp"
#include "stamped_image.hpp"
#include "rtn_status.hpp"
#include <QDir>
#include <QTextStream>

namespace cv
{
    class Mat;
}

namespace bias
{

    class CameraWindow;

    enum ROIType { CIRCLE, NONE};

    enum FlyVsBgModeType { FLY_DARKER_THAN_BG, FLY_BRIGHTER_THAN_BG, FLY_ANY_DIFFERENCE_BG };
    
    struct EllipseParams
    {
        double x;
        double y;
        double a;
        double b;
        double theta;
    };

    // helper functions
    void loadBackgroundModel(QString bgImageFilePath, cv::Mat& bgMedianImage);
    void computeBackgroundMedian(QString bgVideoFilePath, int nFramesBgEst, 
        int lastFrameSample,cv::Mat& bgMedianImage);
    int largestConnectedComponent(cv::Mat& isFg);
    void fitEllipse(cv::Mat& isFg, EllipseParams& flyEllipse);
    double mod2pi(double angle);

    class FlyTrackPlugin : public BiasPlugin
    {
        Q_OBJECT

        public:

            static const QString PLUGIN_NAME;
            static const QString PLUGIN_DISPLAY_NAME;
            static const QString LOG_FILE_EXTENSION;
            static const QString LOG_FILE_POSTFIX;
            static const unsigned int DEFAULT_NUM_BINS;
            static const unsigned int DEFAULT_BIN_SIZE;
            static const double MIN_VEL_MATCH_DOTPROD; // minimum dot product for velocity matching

            FlyTrackPlugin(QWidget *parent=0);
            bool pluginsEnabled();
            void setPluginsEnabled(bool value);

            QPointer<CameraWindow> getCameraWindow();

            virtual void reset();
            virtual void stop();
            virtual void setActive(bool value);
            virtual void processFrames(QList<StampedImage> frameList);
            virtual void setFileAutoNamingString(QString autoNamingString);
            virtual void setFileVersionNumber(unsigned verNum);
            virtual cv::Mat getCurrentImage();
            virtual QString getName();
            virtual QString getDisplayName();
            virtual QVariantMap getConfigAsMap();  
            virtual RtnStatus setConfigFromMap(QVariantMap configMap);
            virtual RtnStatus setConfigFromJson(QByteArray jsonArray);
            virtual RtnStatus runCmdFromMap(QVariantMap cmdMap, bool showErrorDlg=true);
            virtual QString getLogFileExtension();
            virtual QString getLogFilePostfix();
            virtual QString getLogFileName(bool includeAutoNaming);
            virtual QString getLogFileFullPath(bool includeAutoNaming);

        signals:

            void setCaptureDurationRequest(unsigned long);

        protected:

            void initialize();
            void setBackgroundModel();
            void storeBackgroundModel(cv::Mat& bgMedianImage);
            cv::Mat circleROI(double centerX, double centerY, double centerRadius);
            void backgroundSubtraction();
            void setROI();
            void updateVelocityHistory();
            void updateOrientationHistory();
            void updateEllipseHistory();
            void resolveHeadTail();
            void flipFlyOrientationHistory();

            bool active_;
            bool requireTimer_;
            cv::Mat currentImage_;

            double timeStamp_;
            unsigned long frameCount_;

            QString fileAutoNamingString_;
            unsigned int fileVersionNumber_;

            QDir logFileDir_;
            bool loggingEnabled_;
            QFile logFile_;
            QTextStream logStream_;

            // parameters
            int backgroundThreshold_; // foreground threshold
            int nFramesBgEst_; // number of frames used for background estimation, set to 0 to use all frames
            int lastFrameSample_; // last frame sampled for background estimation, set to 0 to use last frame of video
            FlyVsBgModeType flyVsBgMode_; // whether the fly is darker than the background
            ROIType roiType_; // type of ROI
            double roiCenterX_; // x-coordinate of ROI center
            double roiCenterY_; // y-coordinate of ROI center
            double roiRadius_; // radius of ROI
            bool DEBUG_; // flag for debugging
            int historyBufferLength_; // number of frames to buffer velocity, orientation
            double minVelocityMagnitude_; // minimum velocity magnitude in pixels/frame to consider fly moving
            double headTailWeightVelocity_; // weight of velocity dot product in head-tail orientation resolution

            // background model
            QString bgVideoFilePath_; // video to estimate background from
            QString bgImageFilePath_; // saved background median estimate
            QString tmpOutDir_; // temporary output directory
            cv::Mat bgMedianImage_; // median background image
            cv::Mat bgLowerBoundImage_; // lower bound image for background
            cv::Mat bgUpperBoundImage_; // upper bound image for background
			bool bgImageComputed_; // flag indicating if background image has been computed

			// processing of current frame
            bool isFirst_; // flag indicating if this is the first frame
            cv::Mat isFg_; // foreground mask
            cv::Mat inROI_; // mask for ROI
            EllipseParams flyEllipse_; // fly ellipse parameters

            // tracking history
            std::vector<EllipseParams> flyEllipseHistory_; // tracked ellipses
            std::deque<cv::Point2d> velocityHistory_; // tracked velocity buffer
            std::deque<double> orientationHistory_; // tracked orientation buffer
            cv::Point2d meanFlyVelocity_; // mean velocity of fly
            double meanFlyOrientation_; // mean orientation of fly
            bool headTailResolved_; // flag indicating if head-tail orientation has been resolved ever

            void setRequireTimer(bool value);
            void openLogFile();
            void closeLogFile();

    };

}


#endif 


