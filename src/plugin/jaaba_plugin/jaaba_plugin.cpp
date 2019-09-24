#include "jaaba_plugin.hpp"
#include "image_label.hpp"
#include "camera_window.hpp"
#include <QMessageBox>
#include <iostream>
#include <QDebug>
#include <cuda_runtime.h>
#include <string>
#include <QThread>

//

//Camera 0 should always be front view
//Camera 1 should always be side view
//

namespace bias {

    //Public static variables 
    const QString JaabaPlugin::PLUGIN_NAME = QString("jaabaPlugin");
    const QString JaabaPlugin::PLUGIN_DISPLAY_NAME = QString("Jaaba Plugin");


    // Public Methods
    JaabaPlugin::JaabaPlugin(int numberOfCameras, QWidget *parent) : BiasPlugin(parent)
    {

        nviews_ = numberOfCameras;
        setupUi(this);
        connectWidgets();
        initialize();
    }


    QString JaabaPlugin::getName()
    {
        return PLUGIN_NAME;
    }

    
    QString JaabaPlugin::getDisplayName()
    {
        return PLUGIN_DISPLAY_NAME;
    }

    
    cv::Mat JaabaPlugin::getCurrentImage()
    {
        acquireLock();
        cv::Mat currentImageCopy = currentImage_.clone();
        releaseLock();
        return currentImageCopy;
    }


    bool JaabaPlugin::isSender() 
    {

        if(cameraNumber_ ==0)
            return true;
        else 
            return false;
        
    }


    bool JaabaPlugin::isReceiver() 
    {

        if(cameraNumber_==1) 
            return true;
        else 
            return false;

    }


    void JaabaPlugin::stop() 
    {
  
        if( processScoresPtr_ != nullptr )
        {
            if(sideRadioButtonPtr_->isChecked())
            {
                if(processScoresPtr_ -> HOGHOF_side -> isHOGPathSet 
                    && processScoresPtr_ -> HOGHOF_side -> isHOFPathSet
                    && processScoresPtr_ -> classifier -> isClassifierPathSet)
                {

                    HOFTeardown(processScoresPtr_ -> HOGHOF_side -> hof_ctx);
                    HOGTeardown(processScoresPtr_ -> HOGHOF_side -> hog_ctx);
                }
           
            }
        
            if(frontRadioButtonPtr_->isChecked())
            {

                if(processScoresPtr_->HOGHOF_front->isHOGPathSet 
                    && processScoresPtr_ -> HOGHOF_front -> isHOFPathSet 
                    && processScoresPtr_ -> classifier -> isClassifierPathSet)
                {

                    HOFTeardown(processScoresPtr_ -> HOGHOF_front -> hof_ctx);
                    HOGTeardown(processScoresPtr_ -> HOGHOF_front -> hog_ctx);
                }
            }

            processScoresPtr_-> stop(); 
        }

    }


    void JaabaPlugin::reset()
    {
        
        if (isReceiver())
        {
            threadPoolPtr_ = new QThreadPool(this);
            threadPoolPtr_ -> setMaxThreadCount(1);
            if ((threadPoolPtr_ != nullptr) && (processScoresPtr_ != nullptr))
            {
                threadPoolPtr_ -> start(processScoresPtr_);
            }
        }
    }


    void JaabaPlugin::finalSetup()
    {

        QPointer<CameraWindow> partnerCameraWindowPtr = getPartnerCameraWindowPtr();
        if (partnerCameraWindowPtr)
        {
            QPointer<BiasPlugin> partnerPluginPtr = partnerCameraWindowPtr -> getPluginByName("jaabaPlugin");
            qRegisterMetaType<FrameData>("FrameData");
            connect(partnerPluginPtr, SIGNAL(newFrameData(FrameData)), this, SLOT(onNewFrameData(FrameData)));
        }
        //updateMessageLabels();*/
    }


    void JaabaPlugin::processFrames(QList<StampedImage> frameList)
    {
         
        StampedImage latestFrame = frameList.back();
        frameList.clear();
        cv::Mat workingImage = latestFrame.image.clone();    

 
        if((workingImage.rows != 0) && (workingImage.cols != 0))
        {

            acquireLock();  
            currentImage_ = workingImage; 
            frameCount_ = latestFrame.frameCount;
            releaseLock();

            FrameData frameData;
            frameData.count = frameCount_;
            frameData.image = currentImage_;

            if(isSender())
            {

                emit(newFrameData(frameData)); 
            }


            if(isReceiver())
            {

                acquireLock();
                processScoresPtr_->enqueueFrameDataReceiver(frameData);
                releaseLock(); 
 
            }

        }
        
    }

    
    unsigned int JaabaPlugin::getPartnerCameraNumber()
    {
        // Returns camera number of partner camera. For this example
        // we just use camera 0 and 1. In another setting you might do
        // this by GUID or something else.
        if (cameraNumber_ == 0)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }


    QPointer<CameraWindow> JaabaPlugin::getPartnerCameraWindowPtr()
    {

        std::cout << cameraWindowPtrList_ ->size() << std::endl;
        QPointer<CameraWindow> partnerCameraWindowPtr = nullptr;
        if ((cameraWindowPtrList_ -> size()) > 1)
        {
            for (auto cameraWindowPtr : *cameraWindowPtrList_)
            {
                partnerCameraNumber_ = getPartnerCameraNumber();
                if ((cameraWindowPtr -> getCameraNumber()) == partnerCameraNumber_)
                {
                    partnerCameraWindowPtr = cameraWindowPtr;
                }
            }
        }
        return partnerCameraWindowPtr;

    }
    
        
    void JaabaPlugin::initialize()
    {

        QPointer<CameraWindow> cameraWindowPtr = getCameraWindow();
        cameraNumber_ = cameraWindowPtr -> getCameraNumber();
        partnerCameraNumber_ = getPartnerCameraNumber();
        cameraWindowPtrList_ = cameraWindowPtr -> getCameraWindowPtrList();
        processScoresPtr_ = new ProcessScores();
         
        frameCount =0; 
        numMessageSent_=0;
        numMessageReceived_=0;
        frameCount_ = 0;

        updateWidgetsOnLoad();
        setupHOGHOF();
        setupClassifier();

    }


    void JaabaPlugin::connectWidgets()
    { 

        connect(
            sideRadioButtonPtr_,
            SIGNAL(stateChanged(int)),
            this,
            SLOT(SideViewCheckBoxChanged(int))
           );
 
        connect(
            frontRadioButtonPtr_,
            SIGNAL(stateChanged(int)),
            this,
            SLOT(FrontViewCheckBoxChanged(int))
           );   
        connect(
            reloadPushButtonPtr_,
            SIGNAL(clicked()),
            this,
            SLOT(reloadButtonPressed())
           );

        connect(
            detectButtonPtr_,
            SIGNAL(clicked()),
            this,
            SLOT(detectClicked())
           );

        connect(
            saveButtonPtr_,
            SIGNAL(clicked()),
            this,
            SLOT(saveClicked())
           );
  
    }

    
    void JaabaPlugin::setupHOGHOF()
    {
     
        if(sideRadioButtonPtr_->isChecked())
        {
             
            QString file_sde = "/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_sde.avi";
            processScoresPtr_ -> vid_sde = new videoBackend(file_sde);
            processScoresPtr_ -> capture_sde = processScoresPtr_ -> vid_sde -> videoCapObject();
                 
  
            HOGHOF *hoghofside = new HOGHOF(this);
            acquireLock();
            processScoresPtr_->HOGHOF_side = hoghofside;
	    processScoresPtr_->HOGHOF_side->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_side->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_side->CropParam_file = pathtodir_->placeholderText() + CropSideParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_side->loadHOGParams();
            processScoresPtr_->HOGHOF_side->loadHOFParams();
            processScoresPtr_->HOGHOF_side->loadCropParams();
            releaseLock();

        }

        if(frontRadioButtonPtr_->isChecked()) 
        {

            QString file_frt = "/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_frt.avi";  
            processScoresPtr_ -> vid_frt = new videoBackend(file_frt); 
            processScoresPtr_ -> capture_frt = processScoresPtr_ -> vid_frt -> videoCapObject(); 

            HOGHOF *hoghoffront = new HOGHOF(this);  
            acquireLock();
            processScoresPtr_->HOGHOF_front = hoghoffront;
            processScoresPtr_->HOGHOF_front->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_front->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_front->CropParam_file = pathtodir_->placeholderText() + CropFrontParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_front->loadHOGParams();
            processScoresPtr_->HOGHOF_front->loadHOFParams();
            processScoresPtr_->HOGHOF_front->loadCropParams();
            releaseLock();
 
        }

    }


    void JaabaPlugin::setupClassifier() 
    {

        if(sideRadioButtonPtr_->isChecked() || frontRadioButtonPtr_->isChecked())
        {
            
            beh_class *cls = new beh_class(this);
            processScoresPtr_ -> classifier = cls;
            processScoresPtr_ -> classifier->classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
            processScoresPtr_ -> classifier->allocate_model();
            processScoresPtr_ -> classifier->loadclassifier_model();
            
        }
      
    }

    
    int JaabaPlugin::getNumberofViews() 
    {

        return nviews_;

    }

   
    void JaabaPlugin::updateWidgetsOnLoad() 
    {

        sideRadioButtonPtr_ -> setChecked(false);
        frontRadioButtonPtr_ -> setChecked(false);
        detectButtonPtr_ ->setEnabled(false);
        saveButtonPtr_->setEnabled(false);
        save = false;

        if(cameraNumber_ == 0)
        {
            
            this->setEnabled(false);            

        } else {

            this->setEnabled(true);

        }
        //checkviews();

    }
   

    void JaabaPlugin::SideViewCheckBoxChanged(int state)
    {
			
        if (state == Qt::Checked)
	{
	    sideRadioButtonPtr_ -> setChecked(true);
	}
	else
	{  
            sideRadioButtonPtr_ -> setChecked(false);
	}
        //checkviews();
        setupHOGHOF();
        setupClassifier();
        detectEnabled();

    }


    void JaabaPlugin::FrontViewCheckBoxChanged(int state)
    {
            
        if (state == Qt::Checked)
        {   
            frontRadioButtonPtr_ -> setChecked(true);

        }
        else
        {   
            frontRadioButtonPtr_ -> setChecked(false);
        }
        //checkviews();
        setupHOGHOF();
        setupClassifier();
        detectEnabled();

    }


    void JaabaPlugin::detectClicked() 
    {
        
        if(!detectStarted) 
        {
            detectButtonPtr_->setText(QString("Stop Detecting"));
            detectStarted = true;

            if (processScoresPtr_ != nullptr)
            {
                processScoresPtr_ -> acquireLock();
                processScoresPtr_ -> detectOn();
                processScoresPtr_ -> releaseLock();
            }

        } else {

            detectButtonPtr_->setText(QString("Detect"));
            detectStarted = false;
             
            if (processScoresPtr_ != nullptr)
            {
                processScoresPtr_ -> acquireLock();
                processScoresPtr_->  detectOff();
                processScoresPtr_ -> releaseLock();
            }

        }

    }


    void JaabaPlugin::saveClicked()
    {

        if(!save) { 

            saveButtonPtr_->setText(QString("Stop Saving"));
            processScoresPtr_ -> save = true;
 
        } else {

            saveButtonPtr_->setText(QString("Save"));
            processScoresPtr_ -> save = false;

        }       

    }

    
    void JaabaPlugin::detectEnabled() 
    {

        if(sideRadioButtonPtr_->isChecked())
        {

            if(processScoresPtr_->HOGHOF_side->isHOGPathSet 
               && processScoresPtr_->HOGHOF_side->isHOFPathSet
               && processScoresPtr_->classifier->isClassifierPathSet)
            {
                detectButtonPtr_->setEnabled(true);
                saveButtonPtr_->setEnabled(true);
            }
        
        }

        if(frontRadioButtonPtr_->isChecked())
        {

            if(processScoresPtr_->HOGHOF_front->isHOGPathSet 
               && processScoresPtr_->HOGHOF_front->isHOFPathSet
               && processScoresPtr_->classifier->isClassifierPathSet)
            {
                detectButtonPtr_->setEnabled(true);
                saveButtonPtr_->setEnabled(true);
            }

        }

    }    
    

    void JaabaPlugin::reloadButtonPressed()
    {

        pathtodir_->setPlaceholderText(pathtodir_->displayText());
        // load side HOGHOFParams if side view checked
        if(processScoresPtr_->HOGHOF_side == nullptr) 
        {
            setupHOGHOF();

        } else {

            
            processScoresPtr_->HOGHOF_side->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_side->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_side->CropParam_file = pathtodir_->placeholderText() + CropSideParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_side->loadHOGParams();
            processScoresPtr_->HOGHOF_side->loadHOFParams();
            processScoresPtr_->HOGHOF_side->loadCropParams();            
        }

        // load front HOGHOFParams if front view checked
        if(processScoresPtr_->HOGHOF_front == nullptr)
        {

            setupHOGHOF();        
  
        } else {

            processScoresPtr_->HOGHOF_front->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_front->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_front->CropParam_file = pathtodir_->placeholderText() + CropFrontParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_front->loadHOGParams();
            processScoresPtr_->HOGHOF_front->loadHOFParams();
            processScoresPtr_->HOGHOF_front->loadCropParams();
        }

        //load classifier
        if(processScoresPtr_->classifier == nullptr)
        {

            setupClassifier();

        } else {

            processScoresPtr_->classifier->classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
            processScoresPtr_->classifier->allocate_model();
            processScoresPtr_->classifier->loadclassifier_model();

        }
        detectEnabled();
       
    }


    /*void JaabaPlugin::checkviews() 
    {
      
        if(~sideRadioButtonPtr_->isChecked() || ~frontRadioButtonPtr_->isChecked()) 
        {
 
            if(nviews_ == 2) 
            {
                // if both views are checked
                QString errMsgText = QString("Number of cameras not equal to number of views ");
                QString errMsgTitle = QString("Number of Views error");
                QMessageBox::critical(this, errMsgTitle, errMsgText);

            }

        }

            // Only one view checked
        if(sideRadioButtonPtr_->isChecked() && frontRadioButtonPtr_->isChecked()) 
        {

            if(nviews_ < 2) 
            { 

                QString errMsgText = QString("Number of cameras not equal to number of views ");
                QString errMsgTitle = QString("Number of Views error");
                QMessageBox::critical(this, errMsgTitle, errMsgText);

            }

        }       

    }*/

    
    // Private Slots
    // ------------------------------------------------------------------------

    void JaabaPlugin::onNewFrameData(FrameData data)
    {

        if(isReceiver()) 
        {

            //get frame from sender plugin
            acquireLock();
            processScoresPtr_->enqueueFrameDataSender(data);
            releaseLock();
            
            numMessageReceived_++;
          
        }        
    }
    
}

