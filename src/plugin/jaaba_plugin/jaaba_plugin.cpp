#include "jaaba_plugin.hpp"
#include <iostream>
#include <QDebug>
#include <cuda_runtime.h>
#include <string>


//

//Camera 1 should always be front view
//Camera 0 should always be side view
//

namespace bias {

    //Public static variables 
    const QString JaabaPlugin::PLUGIN_NAME = QString("jaabaPlugin");
    const QString JaabaPlugin::PLUGIN_DISPLAY_NAME = QString("Jaaba Plugin");

    // Public Methods
        JaabaPlugin::JaabaPlugin(int numberOfCameras, QPointer<QThreadPool> threadPoolPtr, 
                             GetTime* gettime, QWidget *parent) : BiasPlugin(parent)
    {

        nviews_ = numberOfCameras;
        threadPoolPtr_ = threadPoolPtr;
        gettime_ = gettime;
        cudaError_t err = cudaGetDeviceCount(&nDevices_);
        if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
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


    void JaabaPlugin::getFormatSettings()
    {

        Format7Settings settings;
        settings = cameraPtr_->getFormat7Settings();
        image_height = settings.height;
        image_width = settings.width;

        //DEVEL
        /*if(processScoresPtr_side->vid_sde)
        {
            image_height = 384; //height of frame from video input
            image_width = 260;   //width of frame from video input 
        }*/

    }


    bool JaabaPlugin::isSender()
    {

        if (cameraNumber_ == 1)
        {
            return true;
             
        }else {

            return false;

        }
    }


    bool JaabaPlugin::isReceiver()
    {

        if (cameraNumber_ == 0)
        {
            return true;

        }else{

            return false;
        }
    }


    void JaabaPlugin::stop()
    {

        if (processScoresPtr_side != nullptr)
        {

            if (sideRadioButtonPtr_->isChecked())
            {
                if (processScoresPtr_side->HOGHOF_frame->isHOGPathSet
                    && processScoresPtr_side->HOGHOF_frame->isHOFPathSet 
                    && classifier->isClassifierPathSet)
                {

                    HOFTeardown(processScoresPtr_side->HOGHOF_frame->hof_ctx);
                    HOGTeardown(processScoresPtr_side->HOGHOF_frame->hog_ctx);
                    delete classifier;
                }

            }

            processScoresPtr_side->stop();
            visplots->stop();
            delete processScoresPtr_side->HOGHOF_frame;
            delete processScoresPtr_side;
            delete visplots;

        }


        if (processScoresPtr_front != nullptr)
        {

            if (frontRadioButtonPtr_->isChecked())
            {

                if (processScoresPtr_side->HOGHOF_partner->isHOGPathSet
                    && processScoresPtr_side->HOGHOF_partner->isHOFPathSet
                    && classifier->isClassifierPathSet)
                {

                    HOFTeardown(processScoresPtr_front->HOGHOF_partner->hof_ctx);
                    HOGTeardown(processScoresPtr_front->HOGHOF_partner->hog_ctx);
                }
            }

            processScoresPtr_front->stop();
            delete processScoresPtr_front->HOGHOF_partner;
            delete processScoresPtr_front;
        }

    }


    void JaabaPlugin::reset()
    {
        
        if (isReceiver())
        {

            if ((threadPoolPtr_ != nullptr) && (processScoresPtr_side != nullptr))
            {
                threadPoolPtr_ -> start(processScoresPtr_side);
            }

            // this thread adds latencyF to process frames
            /*if((threadPoolPtr_ != nullptr) && (visplots != nullptr))
            {
                threadPoolPtr_->start(visplots);
            }*/
        }

    }


    void JaabaPlugin::gpuInit()
    {

        // intitialize the HOGHOF context for the GPU memory
        getFormatSettings();
        std:cout << image_height << " " << image_width << std::endl;

        if (!processScoresPtr_side->isHOGHOFInitialised)
        {

            if (!(processScoresPtr_side->HOGHOF_frame.isNull()))
            {

                if (nDevices_ >= 2)
                {

                    cudaSetDevice(0);
                    processScoresPtr_side->initHOGHOF(processScoresPtr_side->HOGHOF_frame, image_width, image_height);

                }else {

                    processScoresPtr_side->initHOGHOF(processScoresPtr_side->HOGHOF_frame, image_width, image_height);

                }

            }

        }


        if (!processScoresPtr_front->isHOGHOFInitialised)
        {

            if (!(processScoresPtr_front->HOGHOF_partner.isNull()))
            {

                if (nDevices_ >= 2)
                {
                    cudaSetDevice(1);
                    processScoresPtr_front->initHOGHOF(processScoresPtr_front->HOGHOF_partner, image_width, image_height);

                }else {

                    processScoresPtr_front->initHOGHOF(processScoresPtr_front->HOGHOF_partner, image_width, image_height);
                }


                if (processScoresPtr_side->isHOGHOFInitialised && processScoresPtr_front->isHOGHOFInitialised)
                {

                    classifier->translate_mat2C(&processScoresPtr_side->HOGHOF_frame->hog_shape,
                    &processScoresPtr_front->HOGHOF_partner->hog_shape);
                }
            
            }
        }

        // DEVEL
        /*if (isSender())
        {
            processScoresPtr_side->isHOGHOFInitialised = true;
            processScoresPtr_front->isHOGHOFInitialised = true;
        }*/
    }


    void JaabaPlugin::finalSetup()
    {

        QPointer<CameraWindow> partnerCameraWindowPtr = getPartnerCameraWindowPtr();

        if (partnerCameraWindowPtr)
        {

            QPointer<BiasPlugin> partnerPluginPtr = partnerCameraWindowPtr->getPluginByName("jaabaPlugin");
            qRegisterMetaType<std::shared_ptr<LockableQueue<StampedImage>>>("std::shared_ptr<LockableQueue<StampedImage>>");
            connect(partnerPluginPtr, SIGNAL(partnerImageQueue(std::shared_ptr<LockableQueue<StampedImage>>)),
                this, SLOT(onPartnerPlugin(std::shared_ptr<LockableQueue<StampedImage>>)));

        }

    }


    void JaabaPlugin::resetTrigger()
    {
        //triggerArmedState = true;
        //updateTrigStateInfo();
    }


    void JaabaPlugin::trigResetPushButtonClicked()
    {
        resetTrigger();
    }


    void JaabaPlugin::trigEnabledCheckBoxStateChanged(int state)
    {

        if (state == Qt::Unchecked)
        {
            triggerEnabled = false;
        }else{
            triggerEnabled = true;
        }
        updateTrigStateInfo();
    }


    //void JaabaPlugin::processFrames(QList<StampedImage> frameList)
    void JaabaPlugin::processFrames()
    {

        cv::Mat sideImage;
        cv::Mat frontImage;
        cv::Mat greySide;
        cv::Mat greyFront;

        StampedImage stampedImage0 , stampedImage1;

        //DEVEL
        TimeStamp pc_time;
        int64_t pc_ts1, pc_ts2, cam_ts, delay;
            
        // initialize memory on the gpu 
        if (isReceiver() && ((!processScoresPtr_side->isHOGHOFInitialised) || (!processScoresPtr_front->isHOGHOFInitialised)))
        {
            gpuInit();
            cam_ofs = cameraOffsetTime(cameraPtr_);
        }

        // Send frames from front plugin to side
        if(pluginImageQueuePtr_ != nullptr && isSender())
        {
        
            if(ofs_isSet)
            {
                    
                cam_ofs = cameraOffsetTime(cameraPtr_); 
                  
            }
            
            emit(partnerImageQueue(pluginImageQueuePtr_)); //causes the preview images to be slow
            processScoresPtr_side -> processedFrameCount += 1;

        }


        if (isReceiver() && pluginImageQueuePtr_ != nullptr && partnerPluginImageQueuePtr_ != nullptr)
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

            if ( !(pluginImageQueuePtr_ -> empty()) && !(partnerPluginImageQueuePtr_ -> empty()))
            {

                stampedImage0 = pluginImageQueuePtr_ -> front();
                stampedImage1 = partnerPluginImageQueuePtr_ -> front();

                    
                sideImage = stampedImage0.image.clone();
                frontImage = stampedImage1.image.clone();

                if((sideImage.rows != 0) && (sideImage.cols != 0) 
                    && (frontImage.rows != 0) && (frontImage.cols != 0))
                {

                    acquireLock();
                    currentImage_ = sideImage;
                    frameCount_ = stampedImage0.frameCount;
                    releaseLock();

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
                        
                        if((processScoresPtr_side -> detectStarted_) && (frameCount_  == (processScoresPtr_side -> processedFrameCount+1)))
                        {


                            // Test - Uncomment to perform preprocesing of video frames
                            /*if (processScoresPtr_side->capture_sde.isOpened())
                            {

                                sideImage = processScoresPtr_side->vid_sde->getImage(processScoresPtr_side->capture_sde);
                                processScoresPtr_side->vid_sde->convertImagetoFloat(sideImage);
                                greySide = sideImage;

                            }

                            if (processScoresPtr_front->capture_front.isOpened())
                            {

                                frontImage = processScoresPtr_front->vid_front->getImage(processScoresPtr_front->capture_front);
                                processScoresPtr_front->vid_front->convertImagetoFloat(frontImage);
                                greyFront = frontImage;
                                //std:cout << greyFront.rows << " " << greyFront.cols << std::endl;

                            }*/


                            // preprocessing the frames - Comment this section if using video input 
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

                            if(nDevices_>=2)
                            {
                                    
                                if(processScoresPtr_side -> isSide)
                                {

                                    cudaSetDevice(0);
                                    processScoresPtr_side -> HOGHOF_frame->img.buf = greySide.ptr<float>(0);
                                    processScoresPtr_side -> onProcessSide();
                                    //processScoresPtr_side -> genFeatures(processScoresPtr_side -> HOGHOF_frame, frameCount_);

                                }

                                if(processScoresPtr_front -> isFront)
                                {

                                    cudaSetDevice(1);
                                    processScoresPtr_front -> HOGHOF_partner->img.buf = greyFront.ptr<float>(0);
                                    processScoresPtr_front -> genFeatures(processScoresPtr_front -> HOGHOF_partner, frameCount_);

                                }

                                while(!processScoresPtr_side -> isProcessed_side) {}

                            } else {

                                processScoresPtr_side -> HOGHOF_frame -> img.buf = greySide.ptr<float>(0);
                                processScoresPtr_side -> genFeatures(processScoresPtr_side -> HOGHOF_frame, frameCount_);
                                processScoresPtr_front -> HOGHOF_partner -> img.buf = greyFront.ptr<float>(0);
                                processScoresPtr_front -> genFeatures(processScoresPtr_front -> HOGHOF_partner, frameCount_);

                            }

                            // Test
                            /*if(processScoresPtr_->save && frameCount_ == 2000)
                            {

                                QPointer<HOGHOF> HOGHOF_side = processScoresPtr_->HOGHOF_frame;
                                processScoresPtr_ -> write_histoutput("./out_feat/hog_side_" + std::to_string(frameCount_) 
                                + ".csv", HOGHOF_side->hog_out.data(), HOGHOF_side->hog_shape.x, HOGHOF_side->hog_shape.y, HOGHOF_side->hog_shape.bin);
                                processScoresPtr_ -> write_histoutput("./out_feat/hof_side_" + std::to_string(frameCount_) 
                                + ".csv", HOGHOF_side->hof_out.data(), HOGHOF_side->hof_shape.x, HOGHOF_side->hof_shape.y, HOGHOF_side->hof_shape.bin);


                                QPointer<HOGHOF> HOGHOF_front = processScoresPtr_->HOGHOF_partner;
                                processScoresPtr_ -> write_histoutput("./out_feat/hog_front_" + std::to_string(frameCount_) 
                                + ".csv", HOGHOF_front->hog_out.data(), HOGHOF_front->hog_shape.x, HOGHOF_front->hog_shape.y, 
                                HOGHOF_front->hog_shape.bin);
                                    
                                processScoresPtr_ -> write_histoutput("./out_feat/hof_front_" + std::to_string(frameCount_) 
                                + ".csv", HOGHOF_front->hof_out.data(), HOGHOF_front->hof_shape.x, HOGHOF_front->hof_shape.y, 
                                HOGHOF_front->hof_shape.bin);
                            }*/


                            // compute scores
                            if(classifier -> isClassifierPathSet & processScoresPtr_side->processedFrameCount >= 0)
                            {

                                //std::fill(laserRead.begin(), laserRead.end(), 0);
                                classifier->boost_classify(classifier->score, processScoresPtr_side -> HOGHOF_frame -> hog_out,
                                processScoresPtr_front -> HOGHOF_partner -> hog_out, processScoresPtr_side -> HOGHOF_frame->hof_out,
                                processScoresPtr_front -> HOGHOF_partner -> hof_out, &processScoresPtr_side -> HOGHOF_frame->hog_shape,
                                &processScoresPtr_front -> HOGHOF_partner -> hof_shape, classifier -> nframes,classifier -> model);
                               

                                //triggerLaser();
                                /*visplots -> livePlotTimeVec_.append(stampedImage0.timeStamp);
                                visplots -> livePlotSignalVec_Lift.append(double(classifier->score[0]));
                                visplots -> livePlotSignalVec_Handopen.append(double(classifier->score[1]));
                                visplots -> livePlotSignalVec_Grab.append(double(classifier->score[2]));
                                visplots -> livePlotSignalVec_Supinate.append(double(classifier->score[3]));
                                visplots -> livePlotSignalVec_Chew.append(double(classifier->score[4]));
                                visplots -> livePlotSignalVec_Atmouth.append(double(classifier->score[5]));
                                visplots->livePlotPtr_->show();
                                processScoresPtr_side -> write_score("classifierscr.csv", frameCount_ , classifier->score[0]);*/
                                

                            }
                                
                            //pc_time = gettime_->getPCtime();

                            // subtract the offset to get camera time
                            /*pc_ts2 = (pc_time.seconds*1e6 + pc_time.microSeconds);
                            cam_ts = ((pc_time.seconds*1e6 + pc_time.microSeconds)-(cameraPtr_->cam_ofs.seconds*1e6 + cameraPtr_->cam_ofs.microSeconds));
                            delay = cam_ts - int64_t(stampedImage0.timeStampVal.seconds*1e6 + stampedImage0.timeStampVal.microSeconds);
                            process_time = cam_ts2 - cam_ts1;*/
                            //time_seconds.push_back({cam_ts,pc_ts2});
                                
                            //if(time_seconds.size() == 100000)
                            //    gettime_->write_time("process_time.csv", 100000, time_seconds);
                                    
                            processScoresPtr_side -> processedFrameCount = frameCount_;
                            processScoresPtr_front -> processedFrameCount = frameCount_; 
                            processScoresPtr_side -> isProcessed_side = false;

                        }else{ std::cout << "skipped 1 " << frameCount_ << std::endl; }

                    }else{ std::cout << "skipped 2 " << frameCount_ << std::endl; }
                     
                }else { std::cout << "skipped 3" << frameCount_ << std::endl;}
                    
                pluginImageQueuePtr_->pop();
                partnerPluginImageQueuePtr_ -> pop();
             
            }else { std::cout << "skipped 4 " << frameCount_ << std::endl; }

            pluginImageQueuePtr_->releaseLock();
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

        }else{

            return 0;

        }
    }


    QPointer<CameraWindow> JaabaPlugin::getPartnerCameraWindowPtr()
    {

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
            cameraPtr_ = cameraWindowPtr->getCameraPtr();
            processScoresPtr_side = new ProcessScores(this);   
            processScoresPtr_front = new ProcessScores(this);    
            visplots = new VisPlots(livePlotPtr,this);
            std::cout << cameraNumber_ << std::endl;
 
            numMessageSent_=0;
            numMessageReceived_=0;
            frameCount_ = 0;
            laserOn = false;

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
  
 
            connect(
                tabWidgetPtr,
                SIGNAL(currentChanged(currentIndex())),
                this,
                SLOT(setCurrentIndex(currentIndex()))
            );

            /*connect(
                trigEnabledCheckBoxPtr,
                SIGNAL(stateChanged(int)),
                this,
                SLOT(trigEnabledCheckBoxStateChanged(int))
            );

            connect(
               trigResetPushButtonPtr,
               SIGNAL(clicked()),
               this,
               SLOT(trigResetPushButtonClicked())
            );*/

       }

    
       void JaabaPlugin::setupHOGHOF()
       {
     
           if(sideRadioButtonPtr_->isChecked())
           {
             
                //QString file_sde = "/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_sde.avi";
#ifdef WIN32           
                QString file_sde = "C:/Users/27rut/BIAS/BIASJAABA_movies/movie_sde.avi";
#endif

#ifdef linux
                QString file_sde = "/home/patilr/BIAS/BIASJAABA_movies/movie_sde.avi";
#endif
                //processScoresPtr_side -> vid_sde = new videoBackend(file_sde);
                //processScoresPtr_side -> capture_sde = processScoresPtr_side ->vid_sde -> videoCapObject();
                 
  
                HOGHOF *hoghofside = new HOGHOF(this);
                acquireLock();
                processScoresPtr_side -> HOGHOF_frame = hoghofside;
                processScoresPtr_side -> HOGHOF_frame -> HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
                processScoresPtr_side -> HOGHOF_frame -> HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
                processScoresPtr_side -> HOGHOF_frame -> CropParam_file = pathtodir_->placeholderText() + CropSideParamFilePtr_->placeholderText();
                processScoresPtr_side -> HOGHOF_frame -> loadHOGParams();
                processScoresPtr_side -> HOGHOF_frame -> loadHOFParams(); 
                processScoresPtr_side -> HOGHOF_frame -> loadCropParams();
                releaseLock();

            }

            if(frontRadioButtonPtr_->isChecked()) 
            {
//DEVEL
#ifdef WIN32		
                //QString file_frt = "/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_frt.avi";
                QString file_frt = "C:/Users/27rut/BIAS/BIASJAABA_movies/movie_frt.avi"; 
#endif 

#ifdef linux
                QString file_frt = "/home/patilr/BIAS/BIASJAABA_movies/movie_frt.avi";
#endif
                //processScoresPtr_front -> vid_front = new videoBackend(file_frt); 
                //processScoresPtr_front -> capture_front = processScoresPtr_front -> vid_front -> videoCapObject(); 

                HOGHOF *hoghoffront = new HOGHOF(this);  
                acquireLock();
                processScoresPtr_front -> HOGHOF_partner = hoghoffront;
                processScoresPtr_front -> HOGHOF_partner -> HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
                processScoresPtr_front -> HOGHOF_partner -> HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
                processScoresPtr_front -> HOGHOF_partner -> CropParam_file = pathtodir_->placeholderText() + CropFrontParamFilePtr_->placeholderText();
                processScoresPtr_front -> HOGHOF_partner -> loadHOGParams();
                processScoresPtr_front -> HOGHOF_partner -> loadHOFParams();
                processScoresPtr_front -> HOGHOF_partner -> loadCropParams();
                releaseLock();
 
            }

        }


        void JaabaPlugin::setupClassifier() 
        {

            if(sideRadioButtonPtr_->isChecked() || frontRadioButtonPtr_->isChecked())
            {
            
                beh_class *cls = new beh_class(this);
                classifier = cls;
                classifier -> classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
                //qDebug()  << classifier->classifier_file;
                classifier -> allocate_model();
                classifier -> loadclassifier_model();
            
            }
      
        }

    
        int JaabaPlugin::getNumberofViews() 
        {

            return nviews_;

        }

        int JaabaPlugin::getNumberOfDevices()
        {
  
            return nDevices_;

        }

   
        int JaabaPlugin::getLaserTrigger()
        {

            return laserOn;

        }

   
        void JaabaPlugin::updateWidgetsOnLoad() 
        {

            if( cameraNumber_ == 1 )
            {
 
                this -> setEnabled(false);   

            } else {

                sideRadioButtonPtr_ -> setChecked(false);
                frontRadioButtonPtr_ -> setChecked(false);
                detectButtonPtr_ -> setEnabled(false);
                saveButtonPtr_-> setEnabled(false);
                save = false;        

                tabWidgetPtr -> setEnabled(true);
                tabWidgetPtr -> repaint();        
            }

        }
   

        void JaabaPlugin::SideViewCheckBoxChanged(int state)
        {
            
            if (state == Qt::Checked)
            {
                sideRadioButtonPtr_ -> setChecked(true);
        }else{  
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

                if (processScoresPtr_side != nullptr)
                {

                    processScoresPtr_side -> acquireLock();
                    processScoresPtr_side -> detectOn();
                    processScoresPtr_side -> releaseLock();

                    /*if(isSender())
                    {
                         processScoresPtr_ -> acquireLock();
                         processScoresPtr_ -> isFront = true;
                         processScoresPtr_ -> releaseLock();    
                    }*/

                    if(isReceiver())
                    {
                        processScoresPtr_side -> acquireLock();
                        processScoresPtr_side -> isSide = true;
                        processScoresPtr_side -> releaseLock();

                        processScoresPtr_front -> acquireLock();
                        processScoresPtr_front -> isFront = true;
                        processScoresPtr_front -> releaseLock();
                    }

                }

            } else {

                detectButtonPtr_->setText(QString("Detect"));
                detectStarted = false;
             
                if (processScoresPtr_side != nullptr)
                {
                    processScoresPtr_side -> acquireLock();
                    processScoresPtr_side ->  detectOff();
                    processScoresPtr_side -> releaseLock();
                }

            }

        }


        void JaabaPlugin::saveClicked()
        {

            if(!save) { 

                saveButtonPtr_->setText(QString("Stop Saving"));
                processScoresPtr_side -> save = true;
 
            } else {

                saveButtonPtr_->setText(QString("Save"));
                processScoresPtr_side -> save = false;

            }       

        }

    
        void JaabaPlugin::detectEnabled() 
        {

            if(sideRadioButtonPtr_->isChecked())
            {

                if(processScoresPtr_side -> HOGHOF_frame->isHOGPathSet 
                  && processScoresPtr_side -> HOGHOF_frame->isHOFPathSet
                  && classifier->isClassifierPathSet)
                {
                    detectButtonPtr_->setEnabled(true);
                    saveButtonPtr_->setEnabled(true);
                }
        
            }

            if(frontRadioButtonPtr_->isChecked())
            {

                if(processScoresPtr_front -> HOGHOF_partner->isHOGPathSet 
                 && processScoresPtr_front -> HOGHOF_partner->isHOFPathSet
                 && classifier->isClassifierPathSet)
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
                if(processScoresPtr_side -> HOGHOF_frame == nullptr) 
                {
                    setupHOGHOF();

                } else {
            
                    processScoresPtr_side -> HOGHOF_frame->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
                    processScoresPtr_side -> HOGHOF_frame->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
                    processScoresPtr_side -> HOGHOF_frame->CropParam_file = pathtodir_->placeholderText() + CropSideParamFilePtr_->placeholderText();
                    processScoresPtr_side -> HOGHOF_frame->loadHOGParams();
                    processScoresPtr_side -> HOGHOF_frame->loadHOFParams();
                    processScoresPtr_side -> HOGHOF_frame->loadCropParams();            
                }
            }

            // load front HOGHOFParams if front view checked
            if(frontRadioButtonPtr_->isChecked())
            {
                if(processScoresPtr_front -> HOGHOF_partner == nullptr)
                {

                    setupHOGHOF();        
  
                } else {

                    processScoresPtr_front -> HOGHOF_partner->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
                    processScoresPtr_front -> HOGHOF_partner->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
                    processScoresPtr_front -> HOGHOF_partner->CropParam_file = pathtodir_->placeholderText() + CropFrontParamFilePtr_->placeholderText();
                    processScoresPtr_front -> HOGHOF_partner->loadHOGParams();
                    processScoresPtr_front -> HOGHOF_partner->loadHOFParams();
                    processScoresPtr_front -> HOGHOF_partner->loadCropParams();
                }
            }

            //load classifier
            if(classifier == nullptr)
            {

                setupClassifier();

            } else {

                classifier->classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
                classifier->allocate_model();
                classifier->loadclassifier_model();

            }
            detectEnabled();
       
        }


        void JaabaPlugin::triggerLaser()
        {

            int num_beh = static_cast<int>(classifier->beh_present.size()); 
            for(int nbeh =0;nbeh < num_beh;nbeh++)
            {
                if (classifier->score[nbeh] > 0)  
                    laserRead[nbeh] = 1;
                else
                    laserRead[nbeh] = 0;
            }

        }


        RtnStatus JaabaPlugin::connectTriggerDev()
        {
            RtnStatus rtnStatus;

            tabWidgetPtr -> setEnabled(false);
            tabWidgetPtr -> repaint();


            tabWidgetPtr -> setEnabled(true);
            rtnStatus.success = true;
            rtnStatus.message = QString("");
            return rtnStatus;
        }

    
        void JaabaPlugin::updateTrigStateInfo()
        {

            /*if (triggerArmedState)
            {
                trigStateLabelPtr -> setText("State: Ready");
            }
            else
            {
                trigStateLabelPtr -> setText("State: Stopped");
            }


            if (trigEnabledCheckBoxPtr -> isChecked())
            {
                trigStateLabelPtr -> setEnabled(true);
                trigResetPushButtonPtr -> setEnabled(true);
            }
            else
            {
                trigStateLabelPtr -> setEnabled(false);
                trigResetPushButtonPtr -> setEnabled(false);
            }*/
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

    
        void JaabaPlugin::onPartnerPlugin(std::shared_ptr<LockableQueue<StampedImage>> partnerPluginImageQueuePtr)
        {

            if(partnerPluginImageQueuePtr != nullptr)
            { 
                partnerPluginImageQueuePtr_ = partnerPluginImageQueuePtr;
            }
        }



        // Test development

        double JaabaPlugin::convertTimeStampToDouble(TimeStamp curr, TimeStamp init)
        {
            double timeStampDbl = 0;
            timeStampDbl  = double(curr.seconds);
            timeStampDbl -= double(init.seconds);
            timeStampDbl += (1.0e-6)*double(curr.microSeconds);
            timeStampDbl -= (1.0e-6)*double(init.microSeconds);
            return timeStampDbl;
        }

       void JaabaPlugin::write_output(std::string file,float* out_img, unsigned w, unsigned h) 
       {

           std::ofstream x_out;
           x_out.open(file.c_str());
      
          // write hist output to csv file
          for(unsigned i = 0;i < h;i++)
          {
              std::cout << " " << i << std::endl;
              for(unsigned j = 0; j < w;j++) 
              {
                  x_out << out_img[i*w + j];
                  if(j != w-1 || i != h-1)
                     x_out << ",";
              }
          }
       }
}

