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
        {
            return true;

        } else {
  
            return false;
        }
        
    }


    bool JaabaPlugin::isReceiver() 
    {

        if(cameraNumber_==1) 
        {
            return true;
        
        } else { 

            return false;

        }

    }


    void JaabaPlugin::stop() 
    {
  
        if( processScoresPtr_ != nullptr )
        {
            if(sideRadioButtonPtr_->isChecked())
            {
                if(processScoresPtr_ -> HOGHOF_frame -> isHOGPathSet 
                    && processScoresPtr_ -> HOGHOF_frame -> isHOFPathSet
                    && processScoresPtr_ -> classifier -> isClassifierPathSet)
                {

                    HOFTeardown(processScoresPtr_ -> HOGHOF_frame -> hof_ctx);
                    HOGTeardown(processScoresPtr_ -> HOGHOF_frame -> hog_ctx);
                }
           
            }
        
            if(frontRadioButtonPtr_->isChecked())
            {

                if(processScoresPtr_->HOGHOF_frame->isHOGPathSet 
                    && processScoresPtr_ -> HOGHOF_frame -> isHOFPathSet 
                    && processScoresPtr_ -> classifier -> isClassifierPathSet)
                {

                    HOFTeardown(processScoresPtr_ -> HOGHOF_frame -> hof_ctx);
                    HOGTeardown(processScoresPtr_ -> HOGHOF_frame -> hog_ctx);
                }
            }

            processScoresPtr_-> stop(); 
        }

    }


    void JaabaPlugin::reset()
    {
        
        /*if (isReceiver())
        {
            threadPoolPtr_ = new QThreadPool(this);
            threadPoolPtr_ -> setMaxThreadCount(1);

            if ((threadPoolPtr_ != nullptr) && (processScoresPtr_ != nullptr))
            {
                threadPoolPtr_ -> start(processScoresPtr_);
            }
        }

        if (isSender())
        {

            threadPoolPtr_ = new QThreadPool(this);
            threadPoolPtr_ -> setMaxThreadCount(1);
            if ((threadPoolPtr_ != nullptr) && (processScoresPtr_ != nullptr))
            {
                threadPoolPtr_ -> start(processScoresPtr_);
            }
        }*/      

    }


    void JaabaPlugin::finalSetup()
    {

        QPointer<CameraWindow> partnerCameraWindowPtr = getPartnerCameraWindowPtr();
        if (partnerCameraWindowPtr)
        {
            QPointer<BiasPlugin> partnerPluginPtr = partnerCameraWindowPtr -> getPluginByName("jaabaPlugin");
            qRegisterMetaType<FrameData>("FrameData");
            qRegisterMetaType<ShapeData>("ShapeData");
            qRegisterMetaType<std::shared_ptr<LockableQueue<StampedImage>>>("std::shared_ptr<LockableQueue<StampedImage>>");
            connect(partnerPluginPtr, SIGNAL(newFrameData(FrameData)), this, SLOT(onNewFrameData(FrameData)));
            connect(partnerPluginPtr, SIGNAL(newShapeData(ShapeData)), this, SLOT(onNewShapeData(ShapeData)));
            connect(partnerPluginPtr, SIGNAL(partnerImageQueue(std::shared_ptr<LockableQueue<StampedImage>>)) , 
                    this, SLOT(onPartnerPlugin(std::shared_ptr<LockableQueue<StampedImage>>)));

        }
        //updateMessageLabels();*/
    }


    //void JaabaPlugin::processFrames(QList<StampedImage> frameList)
    void JaabaPlugin::processFrames()
    {

        cv::Mat sideImage;
        cv::Mat frontImage;
        cv::Mat greySide;
        cv::Mat greyFront;

        if(pluginImageQueuePtr_ != nullptr && isSender() && processScoresPtr_-> processedFrameCount==-1)
        {
            emit(partnerImageQueue(pluginImageQueuePtr_));             
            processScoresPtr_-> processedFrameCount += 1;
        }
        
        if(isReceiver() && pluginImageQueuePtr_ != nullptr && partnerPluginImageQueuePtr_ != nullptr)
        {

            pluginImageQueuePtr_ -> acquireLock();
            pluginImageQueuePtr_ -> waitIfEmpty();
       
            partnerPluginImageQueuePtr_ -> acquireLock();
            partnerPluginImageQueuePtr_ -> waitIfEmpty();

            if (pluginImageQueuePtr_ -> empty() || partnerPluginImageQueuePtr_ -> empty())
            {
                pluginImageQueuePtr_ -> releaseLock();
                partnerPluginImageQueuePtr_ -> releaseLock();
                return;
            }

            while ( !(pluginImageQueuePtr_ -> empty())  && !(partnerPluginImageQueuePtr_ ->  empty()))
            {

                StampedImage stampedImage0 = pluginImageQueuePtr_ -> front();
                StampedImage stampedImage1 = partnerPluginImageQueuePtr_ -> front();

                sideImage = stampedImage0.image.clone();
                frontImage = stampedImage1.image.clone();

                /*if(processScoresPtr_ -> capture_sde.isOpened())
                {
                    //std::cout << "set side" << std::endl;
                    sideImage = processScoresPtr_ -> vid_sde -> getImage(processScoresPtr_ -> capture_sde);
                    processScoresPtr_ -> vid_sde -> convertImagetoFloat(sideImage);
                    greySide = sideImage;
                }
 
                if(processScoresPtr_ -> capture_front.isOpened())
                {

                    frontImage = processScoresPtr_-> vid_front-> getImage(processScoresPtr_ -> capture_front);
                    processScoresPtr_ -> vid_front -> convertImagetoFloat(frontImage);
                    greyFront = frontImage;
                } */

                if((sideImage.rows != 0) && (sideImage.cols != 0) 
                   && (frontImage.rows != 0) && (frontImage.cols != 0))
                {
 
                    acquireLock();  
                    currentImage_ = sideImage; 
                    frameCount_ = stampedImage0.frameCount;
                    releaseLock();

                    if(!processScoresPtr_->isHOGHOFInitialised)
                    {
                        std::cout << processScoresPtr_->isHOGHOFInitialised << std::endl;
                        if(!(processScoresPtr_-> HOGHOF_frame.isNull()) && !(processScoresPtr_-> HOGHOF_partner.isNull()))
                        {
 
                            processScoresPtr_->initHOGHOF(processScoresPtr_ -> HOGHOF_frame, sideImage.rows, sideImage.cols);
                            processScoresPtr_->initHOGHOF(processScoresPtr_ -> HOGHOF_partner, frontImage.rows, frontImage.cols);   
    
                        }
                    }

                    // Test
                    /*if(sideImage.ptr<float>(0) != nullptr && frameCount == 1000)
                    {
                        imwrite("out_feat/side_" + std::to_string(frameCount) + ".jpg", sideImage);
                        imwrite("out_feat/front_" + std::to_string(frameCount) + ".jpg", frontImage);
                        //sideImage.convertTo(sideImage, CV_32FC1);
                        //frontImage.convertTo(frontImage,CV_32FC1);
                        //sideImage = sideImage / 255;
                        //std::cout << sideImage.rows << " " << sideImage.cols << std::endl; 
                        //write_output("out_feat/side" + std::to_string(frameCount_) + ".csv" , sideImage.ptr<float>(0), sideImage.rows, sideImage.cols);
                        //write_output("out_feat/front" + std::to_string(frameCount_) + ".csv" , frontImage.ptr<float>(0), frontImage.rows , frontImage.cols);
                    }*/
                    

                    if( stampedImage0.frameCount == stampedImage1.frameCount) 
                    {

                        if((processScoresPtr_ -> detectStarted_) && (frameCount_  == (processScoresPtr_ -> processedFrameCount+1)))
                        {
                             
                            // preprocessing the frames
                            // convert the frame into RGB2GRAY
                            if(sideImage.channels() == 3)
                            {
                                cv::cvtColor(sideImage, sideImage, cv::COLOR_BGR2GRAY);
                            }

                            if(frontImage.channels() == 3)
                            {
                                cv::cvtColor(frontImage, frontImage, cv::COLOR_BGR2GRAY);
                            }

                            // convert the frame into float32
                            sideImage.convertTo(greySide, CV_32FC1);
                            frontImage.convertTo(greyFront, CV_32FC1);
                            greySide = greySide / 255;
                            greyFront = greyFront / 255;

                            int nDevices;
                            cudaError_t err = cudaGetDeviceCount(&nDevices);
                            if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

                            if(nDevices>=2)
                            {

                                if(processScoresPtr_ -> isSide)
                                {
                                    cudaSetDevice(0);
                                    GpuTimer timer2;
                                    timer2.Start();
                                    processScoresPtr_ -> HOGHOF_frame->img.buf = greySide.ptr<float>(0);
                                    processScoresPtr_ -> genFeatures(processScoresPtr_ -> HOGHOF_frame, frameCount_);
                                    timer2.Stop();
                                    //write_score("timing_gpu1.csv", lastProcessedCount, timer2.Elapsed()/1000);
                                }

                                if(processScoresPtr_ -> isFront)
                                {
                                    cudaSetDevice(1);
                                    GpuTimer timer1;
                                    timer1.Start();
                                    processScoresPtr_ -> HOGHOF_partner->img.buf = greyFront.ptr<float>(0);
                                    processScoresPtr_ -> genFeatures(processScoresPtr_ -> HOGHOF_partner, frameCount_);
                                    timer1.Stop();
                                }
                                //timer1.Stop();
                                //write_score("timing_double.csv", lastProcessedCount, timer1.Elapsed()/1000);

                            } else {

                                GpuTimer timer1;
                                timer1.Start();
                                processScoresPtr_ -> HOGHOF_frame -> img.buf = greySide.ptr<float>(0);
                                processScoresPtr_ -> genFeatures(processScoresPtr_ -> HOGHOF_frame, frameCount_);
                                processScoresPtr_ -> HOGHOF_partner -> img.buf = greyFront.ptr<float>(0);
                                processScoresPtr_ -> genFeatures(processScoresPtr_ -> HOGHOF_partner, frameCount_);
                                timer1.Stop();

                            }
 
                    
                            if(processScoresPtr_->save && frameCount_ == 2000)
                            {

                                QPointer<HOGHOF> HOGHOF_side = processScoresPtr_->HOGHOF_frame;
                                processScoresPtr_ -> write_histoutput("./out_feat/hog_side_" + std::to_string(frameCount_) + ".csv", HOGHOF_side->hog_out.data(),
                                     HOGHOF_side->hog_shape.x, HOGHOF_side->hog_shape.y, HOGHOF_side->hog_shape.bin);
                                processScoresPtr_ -> write_histoutput("./out_feat/hof_side_" + std::to_string(frameCount_) + ".csv", HOGHOF_side->hof_out.data(),
                                     HOGHOF_side->hof_shape.x, HOGHOF_side->hof_shape.y, HOGHOF_side->hof_shape.bin);
                        

                                QPointer<HOGHOF> HOGHOF_front = processScoresPtr_->HOGHOF_partner;
                                processScoresPtr_ -> write_histoutput("./out_feat/hog_front_" + std::to_string(frameCount_) + ".csv", HOGHOF_front->hog_out.data()
                                         , HOGHOF_front->hog_shape.x, HOGHOF_front->hog_shape.y, HOGHOF_front->hog_shape.bin);
                                processScoresPtr_ -> write_histoutput("./out_feat/hof_front_" + std::to_string(frameCount_) + ".csv", HOGHOF_front->hof_out.data()
                                         , HOGHOF_front->hof_shape.x, HOGHOF_front->hof_shape.y, HOGHOF_front->hof_shape.bin);
                            }

                            processScoresPtr_ -> processedFrameCount = frameCount_;
                                        
                        }

                        pluginImageQueuePtr_ -> pop();
                        partnerPluginImageQueuePtr_ -> pop();
                    }

                }
                
            }
       
            pluginImageQueuePtr_ -> releaseLock();
            partnerPluginImageQueuePtr_ -> releaseLock(); 
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
 
        numMessageSent_=0;
        numMessageReceived_=0;
        frameCount_ = 0;
        frameCount = 0;

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
            processScoresPtr_->HOGHOF_frame = hoghofside;
	    processScoresPtr_->HOGHOF_frame->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_frame->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_frame->CropParam_file = pathtodir_->placeholderText() + CropSideParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_frame->loadHOGParams();
            processScoresPtr_->HOGHOF_frame->loadHOFParams();
            processScoresPtr_->HOGHOF_frame->loadCropParams();
            releaseLock();

        }

        if(frontRadioButtonPtr_->isChecked()) 
        {

            QString file_frt = "/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_frt.avi";  
            processScoresPtr_ -> vid_front = new videoBackend(file_frt); 
            processScoresPtr_ -> capture_front = processScoresPtr_ -> vid_front -> videoCapObject(); 

            HOGHOF *hoghoffront = new HOGHOF(this);  
            acquireLock();
            processScoresPtr_->HOGHOF_partner = hoghoffront;
            processScoresPtr_->HOGHOF_partner->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_partner->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_partner->CropParam_file = pathtodir_->placeholderText() + CropFrontParamFilePtr_->placeholderText();
            processScoresPtr_->HOGHOF_partner->loadHOGParams();
            processScoresPtr_->HOGHOF_partner->loadHOFParams();
            processScoresPtr_->HOGHOF_partner->loadCropParams();
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

        if( cameraNumber_ == 0 )
        {
            this -> setEnabled(false);   

        } else {

            sideRadioButtonPtr_ -> setChecked(false);
            frontRadioButtonPtr_ -> setChecked(false);
            detectButtonPtr_ -> setEnabled(false);
            saveButtonPtr_-> setEnabled(false);
            save = false;        
        }
        /* if(cameraNumber_ == 0)
        {
            
            sideRadioButtonPtr_ ->setEnabled(false);            

        } else {

            frontRadioButtonPtr_->setEnabled(false);

        }*/
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

                if(isSender())
                {
                     processScoresPtr_ -> acquireLock();
                     processScoresPtr_ -> isFront = true;
                     processScoresPtr_ -> releaseLock();    
                }

                if(isReceiver())
                {
                     processScoresPtr_ -> acquireLock();
                     processScoresPtr_ -> isSide = true;
                     processScoresPtr_ -> releaseLock();
                }

               
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

            if(processScoresPtr_->HOGHOF_frame->isHOGPathSet 
               && processScoresPtr_->HOGHOF_frame->isHOFPathSet
               && processScoresPtr_->classifier->isClassifierPathSet)
            {
                detectButtonPtr_->setEnabled(true);
                saveButtonPtr_->setEnabled(true);
            }
        
        }

        if(frontRadioButtonPtr_->isChecked())
        {

            if(processScoresPtr_->HOGHOF_partner->isHOGPathSet 
               && processScoresPtr_->HOGHOF_partner->isHOFPathSet
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
        if(sideRadioButtonPtr_->isChecked())
        {
            if(processScoresPtr_->HOGHOF_frame == nullptr) 
            {
                setupHOGHOF();

            } else {
            
                processScoresPtr_->HOGHOF_frame->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
                processScoresPtr_->HOGHOF_frame->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
                processScoresPtr_->HOGHOF_frame->CropParam_file = pathtodir_->placeholderText() + CropSideParamFilePtr_->placeholderText();
                processScoresPtr_->HOGHOF_frame->loadHOGParams();
                processScoresPtr_->HOGHOF_frame->loadHOFParams();
                processScoresPtr_->HOGHOF_frame->loadCropParams();            
            }
        }

        // load front HOGHOFParams if front view checked
        if(frontRadioButtonPtr_->isChecked())
        {
            if(processScoresPtr_->HOGHOF_partner == nullptr)
            {

                setupHOGHOF();        
  
            } else {

                processScoresPtr_->HOGHOF_frame->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
                processScoresPtr_->HOGHOF_frame->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
                processScoresPtr_->HOGHOF_frame->CropParam_file = pathtodir_->placeholderText() + CropFrontParamFilePtr_->placeholderText();
                processScoresPtr_->HOGHOF_frame->loadHOGParams();
                processScoresPtr_->HOGHOF_frame->loadHOFParams();
                processScoresPtr_->HOGHOF_frame->loadCropParams();
            }
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
            processScoresPtr_->enqueueFrameData(data);
            releaseLock();
            
            numMessageReceived_++;
          
        }


        if(isSender())
        {

            //get frame from sender plugin
            acquireLock();
            processScoresPtr_->enqueueFrameData(data);
            releaseLock();

            numMessageReceived_++;

        }
        
    }

    void JaabaPlugin::onNewShapeData(ShapeData data)
    {

        std::cout << "called " << std::endl;
        acquireLock();
        //if(processScoresPtr_-> isHOGHOFInitialised) {
            //processScoresPtr_->HOGHOF_partner->hog_shape.x = data.shapex;  
            //processScoresPtr_->HOGHOF_partner->hog_shape.y = data.shapey;
            //processScoresPtr_->HOGHOF_partner->hog_shape.bin = data.bins; 
            //processScoresPtr_-> isHOGHOFInitialised = false;    
        //}
        releaseLock();
    }


    void JaabaPlugin::onPartnerPlugin(std::shared_ptr<LockableQueue<StampedImage>> partnerPluginImageQueuePtr)
    {

        std::cout << "no no no " << std::endl; 
        if(partnerPluginImageQueuePtr != nullptr)
        { 
            partnerPluginImageQueuePtr_ = partnerPluginImageQueuePtr;
            std::cout << "ho ho ho " << std::endl;
        }
    }


    // Test development

    void JaabaPlugin::write_output(std::string file,float* out_img, unsigned w, unsigned h) {

       std::ofstream x_out;
       x_out.open(file.c_str());

      
       // write hist output to csv file
       for(int i = 0;i < h;i++){
           std::cout << " " << i << std::endl;
           for(int j = 0; j < w;j++){
               x_out << out_img[i*w + j];
                  if(j != w-1 || i != h-1)
                      x_out << ",";
           }
       }

    }
     
}

