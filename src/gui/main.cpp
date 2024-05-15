#include <list>
#include <QApplication>
#include <QSharedPointer>
#include <QMessageBox>
#include "camera_window.hpp"
#include "camera_facade.hpp"
#include "affinity.hpp"
#include <iostream>
#include <QCommandLineParser>


// ------------------------------------------------------------------------
// TO DO ... temporary main function. Currently just opens a camera
// window for each camera found attached to the system.
// ------------------------------------------------------------------------
int main (int argc, char *argv[])
{
    QApplication app(argc, argv);
   
    QCoreApplication::setApplicationName("BIAS");

    QCommandLineParser parser;
    parser.setApplicationDescription("BIAS help");
    parser.addHelpOption();
    // -i <in-video-file> or --in <in-video-file> or --in-video <in-video-file> 
    // capture from video instead of camera
    parser.addOption(QCommandLineOption(
        QStringList() << "i" << "in" << "in-video",
        QString("Capture video from file <in-video-file>"),
        QString("in-video-file")));
    // -c <config-file> or --config <config-file>
    parser.addOption(QCommandLineOption(
		QStringList() << "c" << "config",
		QString("Load configuration from <config-file>"),
		QString("config-file")));

    parser.process(app);
    bias::CmdLineParams params;
    params.inVideoFile = parser.value("in-video");
    params.configFile = parser.value("config");


    bias::GuidList guidList;
    bias::CameraFinder cameraFinder;
    std::list<QSharedPointer<bias::CameraWindow>> windowPtrList;

    // Get list guids for all cameras found
    try
    { 
        guidList = cameraFinder.getGuidList();
    }
    catch (bias::RuntimeError &runtimeError)
    {
        QString msgTitle("Camera Enumeration Error");
        QString msgText("Camera enumeration failed:\n\nError ID: ");
        msgText += QString::number(runtimeError.id());
        msgText += QString("\n\n");
        msgText += QString::fromStdString(runtimeError.what());
        QMessageBox::critical(0, msgTitle,msgText);
        return 0;
    }

    // If no cameras found - error
    if (guidList.empty()) 
    {
        QString msgTitle("Camera Enumeration Error");
        QString msgText("No cameras found");
        QMessageBox::critical(0, msgTitle,msgText);
        return 0;
    }

    // Get number of cameras
    unsigned int numCam = guidList.size();
    bias::ThreadAffinityService::setNumberOfCameras(numCam);

    // Open camera window for each camera 
    QRect baseGeom;
    QRect nextGeom;
    unsigned int camCnt;
    bias::GuidList::iterator guidIt;
    for (guidIt=guidList.begin(), camCnt=0; guidIt!=guidList.end(); guidIt++, camCnt++)
    {
        bias::Guid guid = *guidIt;
        QSharedPointer<bias::CameraWindow> windowPtr(new bias::CameraWindow(guid, camCnt, numCam, params));
        windowPtr -> show();
        if (camCnt==0)
        {
            baseGeom = windowPtr -> geometry();
        }
        else
        {
            nextGeom.setX(baseGeom.x() + 40*camCnt);
            nextGeom.setY(baseGeom.y() + 40*camCnt);
            nextGeom.setWidth(baseGeom.width());
            nextGeom.setHeight(baseGeom.height());
            windowPtr -> setGeometry(nextGeom);
        }
        windowPtrList.push_back(windowPtr);
    }
    return app.exec();
}

