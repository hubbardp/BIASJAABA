//#include <list>
#include <QList>
#include <QApplication>
#include <QSharedPointer>
#include <QMessageBox>
#include "camera_window.hpp"
#include "camera_facade.hpp"
#include "affinity.hpp"
#include <iostream>
#include "parser.hpp"
#include "getopt.h"


namespace bias {

    // parser defined here because defining in parser giving link errors. Need to fix in future
    void parser(int argc, char *argv[], CmdLineParams& cmdlineparams) {
        int opt;

        // place ':' in the beginning of the string so that program can 
        //tell between '?' and ':' 
        while ((opt = getopt(argc, argv, ":o:i:s:c:l:v:f:k:w:p:d:n:")) != -1)
        {
            switch (opt)
            {
            case 'o':
                cmdlineparams.output_dir = optarg;
                break;
            case 'i':
                cmdlineparams.isVideo = stoi(optarg);
                break;
            case 's':
                cmdlineparams.saveFeat = stoi(optarg);
                break;
            case 'c':
                cmdlineparams.compute_jaaba = stoi(optarg);
                break;
            case 'l':
                cmdlineparams.classify_scores = stoi(optarg);
                break;
            case 'v':
                cmdlineparams.visualize = stoi(optarg);
                break;
            case 'f':
                cmdlineparams.numframes = stoi(optarg);
                break;
            case 'k':
                cmdlineparams.isSkip = stoi(optarg);
                break;
            case 'w':
                cmdlineparams.wait_thres = stoi(optarg);
                break;
            case 'n':
                cmdlineparams.window_size = stoi(optarg);
                break;
            case 'd':
                cmdlineparams.debug = stoi(optarg);
                break;
            case 'p':
                cmdlineparams.comport = optarg;
                break;
            case ':':
                printf("Required argument %c", opt);
                break;
            case '?':
                //printf("unknown option : %c\n", optopt);
                break;
            }
        }

        // optind is for the extra arguments
        // which are not parsed
        //for (; optind < argc; optind++) {
        //    printf(“extra arguments : %s\n”, argv[optind]);
        //}
    }

}

// ------------------------------------------------------------------------
// TO DO ... temporary main function. Currently just opens a camera
// window for each camera found attached to the system.
// ------------------------------------------------------------------------
int main (int argc, char *argv[])
{
    QApplication app(argc, argv);

    bias::CmdLineParams cmdparams;
    bias::parser(argc, argv, cmdparams);
    bias::print(cmdparams);

    bias::GuidList guidList;
    bias::CameraFinder cameraFinder;

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
    unsigned int numCam = uint(guidList.size());
    bias::ThreadAffinityService::setNumberOfCameras(numCam);

    // Open camera window for each camera 
    QRect baseGeom;
    QRect nextGeom;
    unsigned int camCnt;
    bias::GuidList::iterator guidIt;

    QSharedPointer<QList<QPointer<bias::CameraWindow>>> windowPtrList(new QList<QPointer<bias::CameraWindow>>);

    for (guidIt=guidList.begin(), camCnt=0; guidIt!=guidList.end(); guidIt++, camCnt++)
    {
        bias::Guid guid = *guidIt;
        QPointer<bias::CameraWindow> windowPtr(new bias::CameraWindow(guid, camCnt, numCam, windowPtrList, cmdparams));

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
        windowPtrList -> push_back(windowPtr);
    }

    // Run final setup for camera windows. This is for things which require all camera windows to 
    // have already been created and added to the windowPtrList. For example,  creating signals/slots
    // between plugins for different camera windows, etc.
    for (auto windowPtr : *windowPtrList)
    {
        windowPtr -> finalSetup();
    }

    return app.exec();
}

