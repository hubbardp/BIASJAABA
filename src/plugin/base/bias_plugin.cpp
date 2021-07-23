#include "bias_plugin.hpp"
#include <iostream>
#include <QtDebug>
#include <opencv2/core/core.hpp>
#include "camera_window.hpp"

namespace bias
{

    const QString BiasPlugin::PLUGIN_NAME = QString("basePlugin"); 
    const QString BiasPlugin::PLUGIN_DISPLAY_NAME = QString("Base Plugin"); 
    const QString BiasPlugin::LOG_FILE_EXTENSION = QString("txt");
    const QString BiasPlugin::LOG_FILE_POSTFIX = QString("plugin_log");

    // Pulbic
    // ------------------------------------------------------------------------

    BiasPlugin::BiasPlugin(QWidget *parent) : QDialog(parent) 
    { 
        active_ = false;
        setRequireTimer(false);
        ofs_isSet = true;
    }

    void BiasPlugin::finalSetup() 
    { }

    void BiasPlugin::reset()
    { }

    void BiasPlugin::setFileAutoNamingString(QString autoNamingString)
    {
        fileAutoNamingString_ = autoNamingString;
    }

    void BiasPlugin::setFileVersionNumber(unsigned verNum)
    {
        fileVersionNumber_ = verNum;
    }

    void BiasPlugin::stop()
    { }

    void BiasPlugin::setActive(bool value)
    {
        active_ = value;
    }


    bool BiasPlugin::isActive()
    {
        return active_;
    }


    bool BiasPlugin::requireTimer()
    {
        return requireTimer_;
    }

    //void BiasPlugin::processFrames(QList<StampedImage> frameList) 
    void BiasPlugin::processFrames()
    { 


        pluginImageQueuePtr_ -> acquireLock();
        pluginImageQueuePtr_ -> waitIfEmpty();
        if (pluginImageQueuePtr_ -> empty())
        {
            pluginImageQueuePtr_ -> releaseLock();
            return;
            //break;
        }

        while ( !(pluginImageQueuePtr_ ->  empty()) )
        {
            StampedImage stampedImage = pluginImageQueuePtr_ -> front();
            pluginImageQueuePtr_ -> pop();
        
            acquireLock();
            currentImage_ = stampedImage.image;
            timeStamp_ = stampedImage.timeStamp;
            frameCount_ = stampedImage.frameCount; 
            releaseLock();

        }

        pluginImageQueuePtr_ -> releaseLock(); 
        /*acquireLock();
        StampedImage latestFrame = frameList.back();
        frameList.clear();
        currentImage_ = latestFrame.image;
        timeStamp_ = latestFrame.timeStamp;
        frameCount_ = latestFrame.frameCount;
        releaseLock();*/
    } 


    cv::Mat BiasPlugin::getCurrentImage()
    {
        acquireLock();
        cv::Mat currentImageCopy = currentImage_.clone();
        releaseLock();
        return currentImageCopy;
    }


    QString BiasPlugin::getName()
    {
        return PLUGIN_NAME;
    }


    QString BiasPlugin::getDisplayName()
    {
        return PLUGIN_DISPLAY_NAME;
    }


    QPointer<CameraWindow> BiasPlugin::getCameraWindow()
    {
        QPointer<CameraWindow> cameraWindowPtr = (CameraWindow*)(parent());
        return cameraWindowPtr;
    }


    RtnStatus BiasPlugin::runCmdFromMap(QVariantMap cmdMap, bool showErrorDlg)
    {
        qDebug() << __FUNCTION__;
        RtnStatus rtnStatus;
        return rtnStatus;
    }

    QVariantMap BiasPlugin::getConfigAsMap()  
    {
        QVariantMap configMap;
        return configMap;
    }

    RtnStatus BiasPlugin::setConfigFromMap(QVariantMap configMap)
    {
        RtnStatus rtnStatus;
        rtnStatus.success = true;
        rtnStatus.message = QString("");
        return rtnStatus;
    }

    RtnStatus BiasPlugin::setConfigFromJson(QByteArray jsonArray)
    {
        RtnStatus rtnStatus;
        rtnStatus.success = true;
        rtnStatus.message = QString("");
        return rtnStatus;
    }

    bool BiasPlugin::pluginsEnabled()
    {
        return getCameraWindow() -> isPluginEnabled();
    }


    void BiasPlugin::setPluginsEnabled(bool value)
    {
        getCameraWindow() -> setPluginEnabled(value);
    }


    QString BiasPlugin::getLogFileExtension()
    {
        return LOG_FILE_EXTENSION;
    }

    QString BiasPlugin::getLogFilePostfix()
    {
        return LOG_FILE_POSTFIX;
    }

    QString BiasPlugin::getLogFileName(bool includeAutoNaming)
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


    QString BiasPlugin::getLogFileFullPath(bool includeAutoNaming)
    {
        QString logFileName = getLogFileName(includeAutoNaming);
        QPointer<CameraWindow> cameraWindowPtr = getCameraWindow();
        logFileDir_ = cameraWindowPtr -> getVideoFileDir();
        QString logFileFullPath = logFileDir_.absoluteFilePath(logFileName);
        return logFileFullPath;
    }

    
    void BiasPlugin::setImageQueue(std::shared_ptr<LockableQueue<StampedImage>> pluginImageQueuePtr) 
    {

        pluginImageQueuePtr_ = pluginImageQueuePtr;

    }

    void BiasPlugin::setupNIDAQ(std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task) {}


    /*TimeStamp BiasPlugin::getPCtime()
    {

#ifdef WIN32

        GetTime* getTime = new GetTime(0,0);
        //get computer local time since midnight

        auto since_midnight = getTime->duration_since_midnight();

        auto hours = std::chrono::duration_cast<std::chrono::hours>(since_midnight);
        auto minutes = std::chrono::duration_cast<std::chrono::minutes>(since_midnight - hours);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(since_midnight - hours - minutes);
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(since_midnight - hours - minutes - seconds);
        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(since_midnight - hours - minutes - seconds - milliseconds);
        auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(since_midnight - hours - minutes - seconds - milliseconds - microseconds);

        getTime->secs = (hours.count()*3600 + minutes.count()*60 + seconds.count());
        getTime->usec = (milliseconds.count()*1000 + microseconds.count() + nanoseconds.count()/1000);
        TimeStamp ts = {getTime->secs, unsigned int(getTime->usec)};


#endif
	    
#ifdef linux	    
        unsigned long long int secs=0;
        unsigned int usec=0;
        time_t curr_time;
        timeval tv;

        //get computer local time since midnight
        curr_time = time(NULL);
        tm *tm_local = localtime(&curr_time);
        gettimeofday(&tv, NULL);
        secs = (tm_local->tm_hour*3600) + tm_local->tm_min*60 + tm_local->tm_sec;
        usec = (unsigned int)tv.tv_usec;
        TimeStamp ts = {secs,usec};
#endif
        return ts;

    }*/


    TimeStamp BiasPlugin::cameraOffsetTime(std::shared_ptr<Lockable<Camera>> cameraPtr)
    {

        TimeStamp cam_ofs={0,0};
        int64_t pc_ts;
        TimeStamp cam_ts;
        double pc_s, cam_s, offset_s;
        std::vector<double> timeofs;

        for(int ind=0;ind < 10;ind++)
        {

            //get computer local time since midnight
	        GetTime* gettime = new GetTime(0,0);
            pc_ts = static_cast<double>(gettime->getPCtime());

            //calculate camera time
            if(cameraPtr!=nullptr){
                cam_ts = cameraPtr->getDeviceTimeStamp();
                cam_s = (double)((cam_ts.seconds*1e6) + (cam_ts.microSeconds))*1e-6;
            }else{
       
                std::cout << " No camera found " << std::endl;
            }

            timeofs.push_back(pc_s-cam_s);
            //printf("%0.06f \n" ,pc_s-cam_s); 
            //printf("%0.06f  %0.06f pc_s-cam_us\n ", pc_s ,cam_s); 
            //printf("%0.06f \n", pc_s);
        }

        //write_time("offset.csv",20,timeofs);

        //calculate mean
        offset_s = accumulate(timeofs.begin(),timeofs.end(),0.0)/timeofs.size();
        cam_ofs.seconds = int(offset_s);
        cam_ofs.microSeconds = (offset_s - cam_ofs.seconds)*1e6;
        ofs_isSet = false;

 
        //calculate std dev
        double std_sum=0;
        for(int k=0;k < timeofs.size() ;k++)
        {
           std_sum += (timeofs[k] - offset_s) * (timeofs[k] - offset_s);
        }

        std_sum = std_sum/timeofs.size();
        std_sum = sqrt(std_sum);

        //printf("%0.06f average offset \n" ,offset_s);
        //printf("%0.06f std deviation \n ",std_sum);
        //printf("%d seconds %d microseconds", cam_ofs.seconds, cam_ofs.microSeconds);

        return cam_ofs;

    }


    /*void BiasPlugin::write_time(std::string file, int framenum, std::vector<double> timeVec)
    {

        std::ofstream x_out;
        x_out.open(file.c_str(), std::ios_base::app);

        for(int frame_id= 0; frame_id < framenum; frame_id++)
        {

            x_out << frame_id << "," << std::setprecision(12) << timeVec[frame_id] << "\n";

        }

    }


    void BiasPlugin::write_delay(std::string file, int framenum, std::vector<int64_t> timeVec)
    {

        std::ofstream x_out;
        x_out.open(file.c_str(), std::ios_base::app);

        for(int frame_id= 0; frame_id < framenum; frame_id++)
        {

            x_out << frame_id << "," << timeVec[frame_id] << "\n";

        }
    }*/


    // Protected methods
    // ------------------------------------------------------------------------

    void BiasPlugin::setRequireTimer(bool value)
    {
        requireTimer_ = value;
    }


    void BiasPlugin::openLogFile()
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


    void BiasPlugin::closeLogFile()
    {
        if (loggingEnabled_ && logFile_.isOpen())
        {
            logStream_.flush();
            logFile_.close();
        }
    }

}
