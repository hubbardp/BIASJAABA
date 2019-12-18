#include "jaaba_plugin.hpp"
#include "image_label.hpp"
#include "camera_window.hpp"
#include <QMessageBox>
#include <iostream>
#include <QDebug>
#include <cuda_runtime.h>
#include <string>
#include <QThread>

#include "time.h"

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
  
        if( processScoresPtr_side != nullptr )
        {

            if(sideRadioButtonPtr_->isChecked())
            {
                if(processScoresPtr_side -> HOGHOF_frame -> isHOGPathSet 
                    && processScoresPtr_side -> HOGHOF_frame -> isHOFPathSet
                    && classifier -> isClassifierPathSet)
                {

                    HOFTeardown(processScoresPtr_side -> HOGHOF_frame -> hof_ctx);
                    HOGTeardown(processScoresPtr_side -> HOGHOF_frame -> hog_ctx);
                }
           
            }
          
            processScoresPtr_side-> stop();

        }


        if(processScoresPtr_front != nullptr )
        {
        
            if(frontRadioButtonPtr_->isChecked())
            {

                if(processScoresPtr_side ->HOGHOF_partner->isHOGPathSet 
                    && processScoresPtr_side -> HOGHOF_partner -> isHOFPathSet 
                    && classifier -> isClassifierPathSet)
                {

                    HOFTeardown(processScoresPtr_front -> HOGHOF_partner -> hof_ctx);
                    HOGTeardown(processScoresPtr_front -> HOGHOF_partner -> hog_ctx);
                }
            }

            processScoresPtr_front-> stop(); 
        }

    }


    void JaabaPlugin::reset()
    {
        
        if (isReceiver())
        {
            //threadPoolPtr_side = new QThreadPool(this);
            //threadPoolPtr_side -> setMaxThreadCount(1);

            threadPoolPtr_front = new QThreadPool(this);
            threadPoolPtr_front -> setMaxThreadCount(1);

            /*if ((threadPoolPtr_side != nullptr) && (processScoresPtr_side != nullptr))
            {
                threadPoolPtr_side -> start(processScoresPtr_side);
            }*/

            if ((threadPoolPtr_front !=nullptr) && (processScoresPtr_front != nullptr))
            {
          
                threadPoolPtr_front -> start(processScoresPtr_front);
            }

        }

        /*if (isSender())
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
            //qRegisterMetaType<FrameData>("FrameData");
            //qRegisterMetaType<ShapeData>("ShapeData");
            qRegisterMetaType<std::shared_ptr<LockableQueue<StampedImage>>>("std::shared_ptr<LockableQueue<StampedImage>>");
            //connect(partnerPluginPtr, SIGNAL(newFrameData(FrameData)), this, SLOT(onNewFrameData(FrameData)));
            //connect(partnerPluginPtr, SIGNAL(newShapeData(ShapeData)), this, SLOT(onNewShapeData(ShapeData)));
            connect(partnerPluginPtr, SIGNAL(partnerImageQueue(std::shared_ptr<LockableQueue<StampedImage>>)) , 
                    this, SLOT(onPartnerPlugin(std::shared_ptr<LockableQueue<StampedImage>>)));
            //connect(processScoresPtr_side, SIGNAL(sideProcess(bool)) , this, SLOT(setProcess_side(bool)));
            //connect(this->processScoresPtr_front, SIGNAL(processScoresPtr_front->frontProcess(bool)) , this, SLOT(setProcess_front(bool)));

        }
        qRegisterMetaType<bool>("bool");
        connect(this, SIGNAL(processSide(bool)) , this, SLOT(onProcessSide(bool)));
        connect(this, SIGNAL(processFront(bool)) , this, SLOT(onProcessFront(bool)));

        //updateMessageLabels();*/
    }


    //void JaabaPlugin::processFrames(QList<StampedImage> frameList)
    void JaabaPlugin::processFrames()
    {

        cv::Mat sideImage;
        cv::Mat frontImage;
        cv::Mat greySide;
        cv::Mat greyFront;

        if(pluginImageQueuePtr_ != nullptr && isSender() && processScoresPtr_side -> processedFrameCount == -1)
        {
            emit(partnerImageQueue(pluginImageQueuePtr_));             
            processScoresPtr_side -> processedFrameCount += 1;
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

   
            if ( !(pluginImageQueuePtr_ -> empty())  && !(partnerPluginImageQueuePtr_ -> empty()))
            {

                //timing info 
                //struct timespec start, end;
                //clock_gettime(CLOCK_REALTIME, &start);
                
                StampedImage stampedImage0 = pluginImageQueuePtr_ -> front();
                StampedImage stampedImage1 = partnerPluginImageQueuePtr_ -> front();

                sideImage = stampedImage0.image.clone();
                frontImage = stampedImage1.image.clone();

                // Test
                if(processScoresPtr_side -> capture_sde.isOpened())
                {
                    //std::cout << "set side" << std::endl;
                    sideImage = processScoresPtr_side-> vid_sde -> getImage(processScoresPtr_side -> capture_sde);
                    processScoresPtr_side -> vid_sde -> convertImagetoFloat(sideImage);
                    greySide = sideImage;

                }
 
                if(processScoresPtr_front -> capture_front.isOpened())
                {

                    frontImage = processScoresPtr_front-> vid_front-> getImage(processScoresPtr_front -> capture_front);
                    processScoresPtr_front -> vid_front -> convertImagetoFloat(frontImage);
                    greyFront = frontImage;

                }


                if((sideImage.rows != 0) && (sideImage.cols != 0) 
                   && (frontImage.rows != 0) && (frontImage.cols != 0))
                {
 
                    acquireLock();  
                    currentImage_ = sideImage; 
                    frameCount_ = stampedImage0.frameCount;
                    releaseLock();

                    if(!processScoresPtr_side -> isHOGHOFInitialised)
                    {

                        if(!(processScoresPtr_side -> HOGHOF_frame.isNull()) )
                        {
                        
                            cudaSetDevice(0);
                            processScoresPtr_side -> initHOGHOF(processScoresPtr_side -> HOGHOF_frame, sideImage.rows, sideImage.cols);

                        }

                    }


                    if(!processScoresPtr_front -> isHOGHOFInitialised)
                    {

                        if(!(processScoresPtr_front -> HOGHOF_partner.isNull()) )
                        {
                         
                            cudaSetDevice(1);
                            processScoresPtr_front -> initHOGHOF(processScoresPtr_front -> HOGHOF_partner, frontImage.rows, frontImage.cols);
                         
                        }


                        if(processScoresPtr_side -> isHOGHOFInitialised && processScoresPtr_front -> isHOGHOFInitialised)
                        {

                            std::cout << "once" << std::endl;
                            classifier->translate_mat2C(&processScoresPtr_side -> HOGHOF_frame->hog_shape,
                                                    &processScoresPtr_front -> HOGHOF_partner->hog_shape);

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

                        if((processScoresPtr_side -> detectStarted_) && (frameCount_  == (processScoresPtr_side -> processedFrameCount+1)))
                        {
                             
                            // preprocessing the frames
                            // convert the frame into RGB2GRAY
                            /*if(sideImage.channels() == 3)
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
                            greyFront = greyFront / 255;*/

                            int nDevices;
                            cudaError_t err = cudaGetDeviceCount(&nDevices);
                            if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

                            if(nDevices>=2)
                            {

                                if(processScoresPtr_side -> isSide)
                                {
                                    cudaSetDevice(0);
                                    //GpuTimer timer2;
                                    //timer2.Start();
                                    std::cout << "launced" << std::endl;
                                    processScoresPtr_side -> HOGHOF_frame->img.buf = greySide.ptr<float>(0);
                                    //emit(processSide(true));
                                    processScoresPtr_side -> genFeatures(processScoresPtr_side -> HOGHOF_frame, frameCount_);
                                    //timer2.Stop();
                                    //gpuSide.push_back(timer2.Elapsed()/1000);

                                }

                                if(processScoresPtr_front -> isFront)
                                {

                                    cudaSetDevice(1);
                                    //GpuTimer timer1;
                                    //timer1.Start();
                                    processScoresPtr_front -> HOGHOF_partner->img.buf = greyFront.ptr<float>(0);
                                    emit(processFront(true));
                                    //processScoresPtr_front -> genFeatures(processScoresPtr_front -> HOGHOF_partner, frameCount_);
                                    std::cout << "runnng front" << std::endl;
                                    //timer1.Stop();
                                    //gpuFront.push_back(timer1.Elapsed()/1000);

                                }

                                while(!processScoresPtr_side -> isProcessed_front ) //&& !processScoresPtr_front -> isProcessed_front)
                                //while( !processScoresPtr_side -> isProcessed_front )
                                {

                                    //wait_to_process_.wait(&mutex_);       
                                }


                            } else {

                                //GpuTimer timer3;
                                //timer3.Start();
                                //cudaSetDevice(0);
                                processScoresPtr_side -> HOGHOF_frame -> img.buf = greySide.ptr<float>(0);
                                processScoresPtr_side -> genFeatures(processScoresPtr_side -> HOGHOF_frame, frameCount_);
                                processScoresPtr_front -> HOGHOF_partner -> img.buf = greyFront.ptr<float>(0);
                                processScoresPtr_front -> genFeatures(processScoresPtr_front -> HOGHOF_partner, frameCount_);
                                //timer3.Stop();
                                //gpuSide.push_back(timer3.Elapsed()/1000);

                            }

                            
                            //mutex_.lock();
                            /*if( !processScoresPtr_side -> isProcessed_side)
                            {
                                std::cout << "inside" << std::endl;
                                wait_to_process_.wait();     
                            }*/
                            //mutex_.unlock(); 

                            /*processScoresPtr_ -> mutex_.lock();
                            if(processScoresPtr_side -> isProcessed_side  && processScoresPtr_front -> isProcessed_front )
                            {
                                signal_to_process_.wakeOne();
                            }
                            processScoresPtr_ -> mutex_.unlock();*/
                                       
    
                            /*if(processScoresPtr_->save && frameCount_ == 2000)
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
                            }*/


                            // compute scores
                            if(classifier -> isClassifierPathSet && processScoresPtr_side -> processedFrameCount >= 0)
                               //&& processScoresPtr_side -> isProcessed_side  && processScoresPtr_front -> isProcessed_front)
                            {
                       
                                //std::cout << "classify" << std::endl; 
                                classifier->score = 0.0;
                                classifier->boost_classify(classifier->score, processScoresPtr_side -> HOGHOF_frame -> hog_out, 
                                                           processScoresPtr_front -> HOGHOF_partner -> hog_out, processScoresPtr_side -> HOGHOF_frame->hof_out,
                                                           processScoresPtr_front -> HOGHOF_partner -> hof_out, &processScoresPtr_side -> HOGHOF_frame->hog_shape, 
                                                           &processScoresPtr_front -> HOGHOF_partner -> hof_shape, classifier -> nframes, 
                                                           classifier -> model);
                                processScoresPtr_side -> write_score("classifierscr.csv", frameCount_ , classifier->score);

                            }

                            processScoresPtr_side -> isProcessed_side = false;
                            processScoresPtr_front -> isProcessed_front = false;
                            processScoresPtr_side -> processedFrameCount = frameCount_;
                            processScoresPtr_front -> processedFrameCount = frameCount_;
                            //frameCount = frameCount + 1;
                            
                                        
                        }

                    }

                }

                /*double time_taken;
                clock_gettime(CLOCK_REALTIME, &end);
                time_taken = (end.tv_sec - start.tv_sec) * 1e9;
                time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
                gpuOverall.push_back(time_taken);

                if(stampedImage0.frameCount==6000)
                {
                   
                    processScoresPtr_side->write_time("gpu_overall_double_2.csv", 6000, gpuOverall);
                    //processScoresPtr_->write_time("timing_singlegpu.csv", 6000, gpuSide);
                    //processScoresPtr_->write_time("timing_gpu1.csv", 6000, gpuSide);
                    //processScoresPtr_->write_time("timing_gpu2.csv", 6000, gpuFront);
                }*/


                pluginImageQueuePtr_ -> pop();
                partnerPluginImageQueuePtr_ -> pop();
                                                
            }
       
            /****************************
            //timer4.Stop();
            //processScoresPtr_->write_score("gpu_execution_double.csv", 1, timer4.Elapsed()/1000);
            *****************************/

            /************
            //end = clock();
            //double time_taken = double(end-start)/double(CLOCKS_PER_SEC);
            //processScoresPtr_->write_score("gpu_execution_double.csv", 1, time_taken);
            ***************/

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
        processScoresPtr_side = new ProcessScores();   
        processScoresPtr_front = new ProcessScores();    
 
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
            processScoresPtr_side -> vid_sde = new videoBackend(file_sde);
            processScoresPtr_side -> capture_sde = processScoresPtr_side -> vid_sde -> videoCapObject();
                 
  
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

            QString file_frt = "/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_frt.avi";  
            processScoresPtr_front -> vid_front = new videoBackend(file_frt); 
            processScoresPtr_front -> capture_front = processScoresPtr_front -> vid_front -> videoCapObject(); 

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
            classifier -> allocate_model();
            classifier -> loadclassifier_model();
            
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

    /*void JaabaPlugin::onNewFrameData(FrameData data)
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

        acquireLock();
        //if(processScoresPtr_-> isHOGHOFInitialised) {
            //processScoresPtr_->HOGHOF_partner->hog_shape.x = data.shapex;  
            //processScoresPtr_->HOGHOF_partner->hog_shape.y = data.shapey;
            //processScoresPtr_->HOGHOF_partner->hog_shape.bin = data.bins; 
            //processScoresPtr_-> isHOGHOFInitialised = false;    
        //}
        releaseLock();
    }*/


    void JaabaPlugin::onPartnerPlugin(std::shared_ptr<LockableQueue<StampedImage>> partnerPluginImageQueuePtr)
    {

        if(partnerPluginImageQueuePtr != nullptr)
        { 
            partnerPluginImageQueuePtr_ = partnerPluginImageQueuePtr;
        }
    }


    void JaabaPlugin::onProcessSide(bool side)
    {

        if(processScoresPtr_side != nullptr)
        {
           acquireLock();
           processScoresPtr_side -> processSide  = side;
           releaseLock();
           std::cout << processScoresPtr_side -> processSide  << "hello from within" << std::endl;
        }
    }


    void JaabaPlugin::onProcessFront(bool front)
    {

        if(processScoresPtr_front != nullptr)
        {
           acquireLock();
           processScoresPtr_front -> processFront  = front;
           releaseLock();
           std::cout << processScoresPtr_front -> processFront << "inside" << std::endl;
        }
    }


    void JaabaPlugin::setProcess_side(bool set_side)
    {

        if(set_side)
        {
            //mutex_.lock();
            //wait_to_process_.wakeAll();
            //processScoresPtr_side->isProcessed_side  = false;
            //std::cout << "false" << std::endl;
            //mutex_.unlock();
            
        }
    
    }


    void JaabaPlugin::setProcess_front(bool set_front)
    {

        if(set_front)
        {

            //wait_to_process_.wakeAll();
            //processScoresPtr_front->isProcessed_front  = false;

        }

    }


    // Test development

    void JaabaPlugin::write_output(std::string file,float* out_img, unsigned w, unsigned h) 
    {

       std::ofstream x_out;
       x_out.open(file.c_str());
      
       // write hist output to csv file
       for(int i = 0;i < h;i++)
       {
           std::cout << " " << i << std::endl;
           for(int j = 0; j < w;j++) 
           {
               x_out << out_img[i*w + j];
                  if(j != w-1 || i != h-1)
                      x_out << ",";
           }
       }

    }
     
}

