#ifndef BIAS_PLUGIN_HPP
#define BIAS_PLUGIN_HPP
#include <QDialog>
#include <QWidget>
#include <QList>
#include "lockable.hpp"
#include "stamped_image.hpp"
#include "rtn_status.hpp"
#include "camera_facade.hpp"
#include <QDir>
#include <QTextStream>
#include <sys/time.h>
#include <fstream>
#include <iomanip>

namespace cv
{
    class Mat;
}

namespace bias
{

    class CameraWindow;

    class BiasPlugin : public QDialog, public Lockable<Empty>
    {
        Q_OBJECT

        public:

            static const QString PLUGIN_NAME;
            static const QString PLUGIN_DISPLAY_NAME;
            static const QString LOG_FILE_EXTENSION;
            static const QString LOG_FILE_POSTFIX;

            BiasPlugin(QWidget *parent=0);

            bool pluginsEnabled();
            void setPluginsEnabled(bool value);

            bool requireTimer();
            bool isActive();

            QPointer<CameraWindow> getCameraWindow();

            virtual void finalSetup();
            virtual void reset();
            virtual void stop();
            virtual void setActive(bool value);
            //virtual void processFrames(QList<StampedImage> frameList);
            virtual void processFrames();
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

            virtual void setImageQueue(std::shared_ptr<LockableQueue<StampedImage>> pluginImageQueuePtr);
            TimeStamp getPCtime();
            TimeStamp cameraOffsetTime(std::shared_ptr<Lockable<Camera>> cameraPtr);
            void write_time(std::string file, int framenum, std::vector<double> timeVec);
            void write_delay(std::string file, int framenum, std::vector<int64_t> timeVec);

            std::shared_ptr<LockableQueue<StampedImage>> pluginImageQueuePtr_;
            std::shared_ptr<Lockable<Camera>> cameraPtr_;

        signals:

            void setCaptureDurationRequest(unsigned long);

        protected:

            bool active_;
            bool requireTimer_;
            bool ofs_isSet;

            cv::Mat currentImage_;

            double timeStamp_;
            unsigned long frameCount_;

            QString fileAutoNamingString_;
            unsigned int fileVersionNumber_;

            QDir logFileDir_;
            bool loggingEnabled_;
            QFile logFile_;
            QTextStream logStream_;

            void setRequireTimer(bool value);
            void openLogFile();
            void closeLogFile();

            
    };

}


#endif 

