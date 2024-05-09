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
            static const unsigned int FLY_DARKER_THAN_BG;
            static const unsigned int FLY_BRIGHTER_THAN_BG;
            static const unsigned int FLY_ANY_DIFFERENCE_BG;

            FlyTrackPlugin(QWidget *parent=0);
            bool pluginsEnabled();
            void setPluginsEnabled(bool value);

            bool requireTimer();
            bool isActive();
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

            void setBackgroundModel();
            void computeBackgroundMedian(cv::Mat& bgMedianImage);
            void storeBackgroundModel(cv::Mat& bgMedianImage);
            void loadBackgroundModel(cv::Mat& bgMedianImage,QString bgImageFilePath);

        signals:

            void setCaptureDurationRequest(unsigned long);

        protected:

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
            int flyVsBgMode_; // whether the fly is darker than the background
            double roiCenterX_ = 468.6963; // x-coordinate of ROI center
            double roiCenterY_ = 480.2917; // y-coordinate of ROI center
            double roiRadius_ = 428.3618; // radius of ROI

            bool isFirst_; // flag indicating if this is the first frame
            QString bgVideoFilePath_; // video to estimate background from
            QString bgImageFilePath_; // saved background median estimate
            QString tmpOutDir_; // temporary output directory
            cv::Mat bgMedianImage_; // median background image
            cv::Mat bgLowerBoundImage_; // lower bound image for background
            cv::Mat bgUpperBoundImage_; // upper bound image for background
            cv::Mat inROI_; // mask for ROI
			bool bgImageComputed_; // flag indicating if background image has been computed

            // color table for connected component plotting
            std::vector<cv::Vec3b> colorTable_;

			// processing of current frame
            cv::Mat isFg_; // foreground mask
            cv::Mat dBkgd_; // absolute difference from background
            cv::Mat ccLabels_; // connected component labels
            int nCCs_; // number of connected components found for current frame
            int maxNCCs_; // maximum number of connected components plotted with different colors

            void setRequireTimer(bool value);
            void openLogFile();
            void closeLogFile();



    };

}


#endif 


