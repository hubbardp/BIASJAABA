#include "jaaba_plugin.hpp"
#include "image_label.hpp"
#include "camera_window.hpp"
//#include "image_grabber.hpp"
#include <QMessageBox>
#include <iostream>
#include <QDebug>
#include <cuda_runtime.h>
#include <string>
#include <QThread>

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

        if(!(sideRadioButtonPtr_->isChecked()) && !(frontRadioButtonPtr_->isChecked()))
            return true;
        else 
            return false;
        
    }


    bool JaabaPlugin::isReceiver() 
    {

        if(sideRadioButtonPtr_->isChecked() && frontRadioButtonPtr_->isChecked())
            return true;
        else 
            return false;

    }


    void JaabaPlugin::stop() 
    {

        if(sideRadioButtonPtr_->isChecked())
        {
            if(HOGHOF_side->isHOGPathSet 
               && HOGHOF_side->isHOFPathSet
               && classifier-> isClassifierPathSet)
            {

                HOFTeardown(HOGHOF_side->hof_ctx);
                HOGTeardown(HOGHOF_side->hog_ctx);

            }
           
        }
        
        if(frontRadioButtonPtr_->isChecked())
        {

            if(HOGHOF_front->isHOGPathSet 
               && HOGHOF_front->isHOFPathSet 
               && classifier-> isClassifierPathSet)
            {

                HOFTeardown(HOGHOF_front->hof_ctx);
                HOGTeardown(HOGHOF_front->hog_ctx);
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


    void JaabaPlugin::initHOGHOF(QPointer<HOGHOF> hoghof)
    {
 
        //std::cout << " " << currentImage_.cols << " " << currentImage_.cols << std::endl;
        hoghof->loadImageParams(384, 260);
        struct HOGContext hogctx = HOGInitialize(logger, hoghof->HOGParams, 384, 260, hoghof->Cropparams);
        struct HOFContext hofctx = HOFInitialize(logger, hoghof->HOFParams, hoghof->Cropparams);
        hoghof->hog_ctx = (HOGContext*)malloc(sizeof(hogctx));
        hoghof->hof_ctx = (HOFContext*)malloc(sizeof(hofctx));
        memcpy(hoghof->hog_ctx, &hogctx, sizeof(hogctx));
        memcpy(hoghof->hof_ctx, &hofctx, sizeof(hofctx));
        hoghof->startFrameSet = false;
        
        //allocate output bytes HOG/HOF per frame 
        hoghof->hog_outputbytes = HOGOutputByteCount(hoghof->hog_ctx);
        hoghof->hof_outputbytes = HOFOutputByteCount(hoghof->hof_ctx);

        //output shape 
        struct HOGFeatureDims hogshape;
        HOGOutputShape(&hogctx, &hogshape);
        struct HOGFeatureDims hofshape;
        HOFOutputShape(&hofctx, &hofshape);
        hoghof->hog_shape = hogshape;
        hoghof->hof_shape = hofshape;
        hoghof->hog_out.resize(hoghof->hog_shape.x * hoghof->hog_shape.y * hoghof->hog_shape.bin);
        hoghof->hof_out.resize(hoghof->hof_shape.x * hoghof->hof_shape.y * hoghof->hof_shape.bin);

    }


    void JaabaPlugin::genFeatures(QPointer<HOGHOF> hoghof,int frame)
    {

        size_t hog_num_elements = hoghof->hog_shape.x * hoghof->hog_shape.y * hoghof->hog_shape.bin;
        size_t hof_num_elements = hoghof->hof_shape.x * hoghof->hof_shape.y * hoghof->hof_shape.bin;

        //Compute and copy HOG/HOF
        
        HOFCompute(hoghof->hof_ctx, hoghof->img.buf, hof_f32); // call to compute and copy is asynchronous
        HOFOutputCopy(hoghof->hof_ctx, hoghof->hof_out.data(), hoghof->hof_outputbytes); // should be called one after 
                                                           // the other to get correct answer
        HOGCompute(hoghof->hog_ctx, hoghof->img);
        HOGOutputCopy(hoghof->hog_ctx, hoghof->hog_out.data(), hoghof->hog_outputbytes);
        
    }

              
    /*void JaabaPlugin::processFrmeData(frameData));
ames(QList<StampedImage> frameList)
    {

        StampedImage latestFrame = frameList.back();
        frameList.clear();

         //get current image
        cv::Mat currentImageFloat;
        cv::Mat image_side;
        cv::Mat image_front;
        cv::Mat workingImage = latestFrame.image.clone();

        // initialize gpu HOGHOF Context
        if((workingImage.rows != 0) && (workingImage.cols != 0)) 
        {
        
            acquireLock();
            currentImage_ = workingImage;
            releaseLock();
            currentImage_.convertTo(currentImageFloat, CV_32FC1);
            currentImageFloat = currentImageFloat / 255;
            frameCount_ = latestFrame.frameCount;
            releaseLock();

            if(isSender())
            {
             
               //Test Development
                //image_front = vid_front->getImage(capture_front); 
                //vid_front->convertImagetoFloat(image_front);
                //image_side = vid_sde->getImage(capture_sde);
                //vid_sde->convertImagetoFloat(image_side);
                //pthread_mutex_lock(&mut);  
                acquireLock();    
                image_front = frameCount_*cv::Mat::ones(260, 352, CV_8UC1);
                frameData.count = frameCount_;
                frameData.image = image_front;
                releaseLock();
                cond_.wakeAll();
                //pthread_mutex_unlock(&mut);
               
                //frameData.image = image_side;

                numMessageSent_++; 
                emit(newFrameData(frameData));
            }

                               
            if(sideRadioButtonPtr_->isChecked())             
            {
  
                ///Test Development
                //capture frame and convert to grey
                acquireLock();
                currentImage_ = vid_sde->getImage(capture_sde);
                //convert to Float and normalize
                vid_sde->convertImagetoFloat(currentImage_);
                currentImageFloat = currentImage_;
                

                //HOGHOF_side->img.buf = frameData.image.ptr<float>(0);
                //HOGHOF_side->img.buf = currentImageFloat.ptr<float>(0);
                releaseLock(); 

                if(HOGHOF_side->isHOGPathSet 
                   && HOGHOF_side->isHOFPathSet)
                {
 
                    if(HOGHOF_side->startFrameSet)
                    {

                        initHOGHOF(HOGHOF_side); 
                        //write_output_shape("./output_shape.csv", "side", HOGHOF_side->hog_shape.x , 
                        //                  HOGHOF_side->hog_shape.y , HOGHOF_side->hog_shape.bin);
                        //read_output_shape("./output_shape.csv");
                        //classifier->translate_mat2C(&tmp_sideshape,&tmp_frontshape);
                        //loadfromSharedMemory();
                                                                     
                    } 

                }

               
                if(detectStarted)
                {

                    genFeatures(HOGHOF_side, frameCount);
                    if(save && frameCount == 500)
                    {

                        //createh5("./out_feat/hoghof_side", frame, 1, hog_num_elements, hof_num_elements, hoghof->hog_out, hoghof->hof_out);
                        write_histoutput("./out_feat/hog_side_" + std::to_string(frameCount) + ".csv", HOGHOF_side->hog_out.data(),
                                 HOGHOF_side->hog_shape.x, HOGHOF_side->hog_shape.y, HOGHOF_side->hog_shape.bin);
                        write_histoutput("./out_feat/hof_side_" + std::to_string(frameCount) + ".csv", HOGHOF_side->hof_out.data(),
                                 HOGHOF_side->hof_shape.x, HOGHOF_side->hof_shape.y, HOGHOF_side->hof_shape.bin);
                    }


                    if(classifier->isClassifierPathSet && frameCount > 1)
                    {

                        classifier->score = 0.0;
                        classifier->boost_classify_side(classifier->score, HOGHOF_side->hog_out, HOGHOF_side->hof_out,
                                                        &HOGHOF_side->hog_shape, classifier->nframes, frameCount,classifier->model);
                        //write_score("classifierscr_side.csv", frameCount, classifier->score);
                        //read_score("classifierscr_side.csv","classifierscr_front.csv", frameCount);

                        //load shared Memory
                        //if(frameCount == 50)
                        //    loadfromSharedMemory();
                        //std::cout << "classifier score side" << classifier->score << "classifier score front " 
                        //<< frontclsscr << std::endl;
                    }
                    //frameCount++;
                }
                //frameCount++;

            }
            
 
            if(frontRadioButtonPtr_->isChecked())
            {

                std::cout << "got:" << frameData.count << "current frame:" << frameCount_ << std::endl;
                acquireLock();
                //pthread_mutex_lock(&mut);
                while(!(frameData.count==frameCount_))                                 
                {
                    //cond_.wait(&mutex_);
                }
                //pthread_mutex_unlock(&mut);
                releaseLock();
                std::cout << "re" << frameData.count << std::endl;
                //HOGHOF_front->img.buf = frameData.image.ptr<float>(0);
               
                //imwrite("./out_feat/img" + std::to_string(frameData.count) + ".bmp", frameData.image);
                //currentImage_ = vid_front->getImage(capture_front); 
                //vid_front->convertImagetoFloat(currentImage_);
                //currentImageFloat = currentImage_;

                //HOGHOF_front->img.buf = currentImageFloat.ptr<float>(0);
                //releaseLock();
                

                if(HOGHOF_front->isHOGPathSet 
                   && HOGHOF_front->isHOFPathSet)
                {

                    if(HOGHOF_front->startFrameSet)
                    {
                        //initialize HOGHOF
                        initHOGHOF(HOGHOF_front);
                        //write_output_shape("./output_shape.csv", "front", HOGHOF_front->hog_shape.x , 
                        //                   HOGHOF_front->hog_shape.y , HOGHOF_front->hog_shape.bin);
                        //read_output_shape("./output_shape.csv");
                        //classifier->translate_mat2C(&HOGHOF_side->hog_shape,&HOGHOF_front->hog_shape);
                    }

                }

                if(detectStarted)
                {  

                    genFeatures(HOGHOF_front, frameCount);  
                    if(save && frameCount == 50) 
                    {

                        write_histoutput("./out_feat/hog_front_" + std::to_string(frameCount) + ".csv", HOGHOF_front->hog_out.data()
                                         , HOGHOF_front->hog_shape.x, HOGHOF_front->hog_shape.y, HOGHOF_front->hog_shape.bin);
                        write_histoutput("./out_feat/hof_front_" + std::to_string(frameCount) + ".csv", HOGHOF_front->hof_out.data()
                                         , HOGHOF_front->hof_shape.x, HOGHOF_front->hof_shape.y, HOGHOF_front->hof_shape.bin);
                    }
            

                    if(classifier->isClassifierPathSet && frameCount > 1)
                    {

                        classifier->score = 0.0;
                        classifier->boost_classify_front(classifier->score, HOGHOF_front->hog_out, HOGHOF_front->hof_out,
                                                     &tmp_frontshape, classifier->nframes, frameCount, classifier->model);
                        //write_score("classifierscr_front.csv", frameCount, classifier->score);
                        //std::cout << "se" << classifier->score  << "re " << frameData.score << std::endl;

                    }
                }
                frameCount++;       
            }
        }
    }*/


    void JaabaPlugin::processFrames(QList<StampedImage> frameList)
    {
         
        cv::Mat currentImageFloat;
        cv::Mat image_front;
        cv::Mat image_side;
        StampedImage latestFrame = frameList.back();
        frameList.clear();
        cv::Mat workingImage = latestFrame.image.clone();    

        
        if((workingImage.rows != 0) && (workingImage.cols != 0))
        {

            acquireLock();  
            currentImage_ = workingImage; 
            frameCount_ = latestFrame.frameCount;
            releaseLock();
 

            if(isSender())
            {

                acquireLock();
                //image_front = frameCount_*cv::Mat::ones(260, 352, CV_8UC1);
                frameData.count = frameCount_;
                frameData.image = currentImage_;
                releaseLock();
                emit(newFrameData(frameData)); 
            }


            if(isReceiver())
            {

                receiverImageQueue_.acquireLock();
                //image_side = frameCount_*cv::Mat::ones(260, 352, CV_8UC1);
                //latestFrame.image = image_side.clone();
                receiverImageQueue_.push(latestFrame);
                receiverImageQueue_.releaseLock();

                startProcessthread();
                //std::cout << "grabThread:receiver" << QThread::currentThreadId() << std::endl;
                //std::cout << "re" << receiverImageQueue_.size() << std::endl;
                
            }

        }
        
    }

 
    void JaabaPlugin::startProcessthread()
    {

        // thread to monitor the queues
        QThread *thread = new QThread;
	connect(thread, SIGNAL(started()), this , SLOT(processData()));
	connect(thread, SIGNAL(finished()), thread, SLOT(deleteLater()));
	thread->start();

    }


    void JaabaPlugin::processData()
    {

        int frame;

        if(!(senderImageQueue_.empty()) && !(receiverImageQueue_.empty()))
        {
        
            acquireLock();   
            FrameData sendImage = senderImageQueue_.front();
            StampedImage receiveImage = receiverImageQueue_.front();
            cv::Mat currImage_side; //= receiveImage.image.clone();
            cv::Mat currImage_front; //= sendImage.image.clone();

            currImage_side = vid_sde->getImage(capture_sde);
            vid_sde->convertImagetoFloat(currImage_side);
            currImage_front = vid_front->getImage(capture_front); 
            vid_front->convertImagetoFloat(currImage_front);
           

            if(receiveImage.frameCount == sendImage.count)
            {
                frame = receiveImage.frameCount;

            } else {
               
                std::cout << "frameCount not synchronized" << std::endl;
            }
            HOGHOF_side->img.buf = currImage_side.ptr<float>(0);
            HOGHOF_front->img.buf = currImage_front.ptr<float>(0);
            //imwrite("./out_feat/se_img" + std::to_string(frameCount) + ".bmp", currImage_front);
            //imwrite("./out_feat/re_img" + std::to_string(frameCount) + ".bmp", currImage_side);
            receiverImageQueue_.pop(); 
            senderImageQueue_.pop();
            releaseLock();
            std::cout << "video framecount" << frameCount << std::endl;
            std::cout << "sesz:" << senderImageQueue_.size() << "resz :" << receiverImageQueue_.size() << std::endl;

            //initialize HOGHOF on cpu/gpu
            if(HOGHOF_side->isHOGPathSet 
             && HOGHOF_side->isHOFPathSet)
            {
                if(HOGHOF_side->startFrameSet)
                {

                    initHOGHOF(HOGHOF_side);

                }
            }

                
            if(HOGHOF_front->isHOGPathSet
                   && HOGHOF_front->isHOFPathSet)
            {
                if(HOGHOF_front->startFrameSet)
                {
                        
                    initHOGHOF(HOGHOF_front);

                }
            }


            if(detectStarted)
            {
                genFeatures(HOGHOF_side,frameCount);
                genFeatures(HOGHOF_front,frameCount);
            
            if(save && frameCount == 200) 
            {


                write_histoutput("./out_feat/hog_side_" + std::to_string(frameCount) + ".csv", HOGHOF_side->hog_out.data(),
                                 HOGHOF_side->hog_shape.x, HOGHOF_side->hog_shape.y, HOGHOF_side->hog_shape.bin);
                write_histoutput("./out_feat/hof_side_" + std::to_string(frameCount) + ".csv", HOGHOF_side->hof_out.data(),
                                 HOGHOF_side->hof_shape.x, HOGHOF_side->hof_shape.y, HOGHOF_side->hof_shape.bin);

                write_histoutput("./out_feat/hog_front_" + std::to_string(frameCount) + ".csv", HOGHOF_front->hog_out.data()
                                         , HOGHOF_front->hog_shape.x, HOGHOF_front->hog_shape.y, HOGHOF_front->hog_shape.bin);
                write_histoutput("./out_feat/hof_front_" + std::to_string(frameCount) + ".csv", HOGHOF_front->hof_out.data()
                                         , HOGHOF_front->hof_shape.x, HOGHOF_front->hof_shape.y, HOGHOF_front->hof_shape.bin);
            }}
            frameCount++;

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
    
        //connect(this, SIGNAL(finished()), thread, SLOT(quit()));
        
    
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

            //Test DEvelopment
            QString file("/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_sde.avi");
            vid_sde = new videoBackend(file);
            capture_sde = vid_sde->videoCapObject();
            ///

            HOGHOF *hoghofside = new HOGHOF(this);  
	    HOGHOF_side = hoghofside;
	    HOGHOF_side->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            HOGHOF_side->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            HOGHOF_side->CropParam_file = pathtodir_->placeholderText() + CropSideParamFilePtr_->placeholderText();
            HOGHOF_side->loadHOGParams();
            HOGHOF_side->loadHOFParams();
            HOGHOF_side->loadCropParams();
 
        }

        if(frontRadioButtonPtr_->isChecked()) 
        {

            QString file("/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_frt.avi");
            vid_front = new videoBackend(file);
            capture_front = vid_front->videoCapObject();

              
            HOGHOF *hoghoffront = new HOGHOF(this);
      	    HOGHOF_front = hoghoffront;
            HOGHOF_front->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            HOGHOF_front->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            HOGHOF_front->CropParam_file = pathtodir_->placeholderText() + CropFrontParamFilePtr_->placeholderText();
            HOGHOF_front->loadHOGParams();
            HOGHOF_front->loadHOFParams();
            HOGHOF_front->loadCropParams();
 
        }

        //Test Development
        if(isSender())
        {
        
            //QString file("/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_frt.avi");
            //vid_front = new videoBackend(file);
            //capture_front = vid_front->videoCapObject();

        }

    }


    void JaabaPlugin::setupClassifier() 
    {

        if(sideRadioButtonPtr_->isChecked() || frontRadioButtonPtr_->isChecked())
        {
            
            beh_class *cls = new beh_class(this);
            classifier = cls;
            classifier->classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
            classifier->allocate_model();
            classifier->loadclassifier_model();
            
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
        detectStarted = false;
        save = false;
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
            //acquireLock();
            detectButtonPtr_->setText(QString("Stop Detecting"));
            detectStarted = true;
            //releaseLock();
            //QThread *thread = new QThread;
            //this->moveToThread(thread);
            //connect(thread, SIGNAL(started()), this , SLOT(run()));
            //connect(this, SIGNAL(finished()), thread, SLOT(quit()));
            //thread->start();
            //thread -> setPriority(QThread::LowPriority);

        } else {

            //acquireLock();
            detectButtonPtr_->setText(QString("Detect"));
            detectStarted = false;
            //releaseLock();
            save = false;

        }

    }


    void JaabaPlugin::saveClicked()
    {

        if(!save) { 

            saveButtonPtr_->setText(QString("Stop Saving"));
            save = true;
 
        } else {

            saveButtonPtr_->setText(QString("Save"));
            save = false;

        }       

    }

    
    void JaabaPlugin::detectEnabled() 
    {

        if(sideRadioButtonPtr_->isChecked())
        {
            if(HOGHOF_side->isHOGPathSet 
               && HOGHOF_side->isHOFPathSet
               && classifier->isClassifierPathSet)
            {
                detectButtonPtr_->setEnabled(true);
                saveButtonPtr_->setEnabled(true);
            }
        
        }

        if(frontRadioButtonPtr_->isChecked())
        {
            if(HOGHOF_front->isHOGPathSet 
               && HOGHOF_front->isHOFPathSet
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
        if(HOGHOF_side == nullptr) 
        {
            setupHOGHOF();

        } else {

            HOGHOF_side->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            HOGHOF_side->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            HOGHOF_side->CropParam_file = pathtodir_->placeholderText() + CropSideParamFilePtr_->placeholderText();
            HOGHOF_side->loadHOGParams();
            HOGHOF_side->loadHOFParams();
            HOGHOF_side->loadCropParams();            
        }

        // load front HOGHOFParams if front view checked
        if(HOGHOF_front == nullptr)
        {
            setupHOGHOF();

        } else {

            HOGHOF_front->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            HOGHOF_front->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            HOGHOF_front->CropParam_file = pathtodir_->placeholderText() + CropFrontParamFilePtr_->placeholderText();
            HOGHOF_front->loadHOGParams();
            HOGHOF_front->loadHOFParams();
            HOGHOF_front->loadCropParams();
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


    void JaabaPlugin::checkviews() 
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

    }

    
    // Private Slots
    // ------------------------------------------------------------------------

    void JaabaPlugin::onNewFrameData(FrameData data)
    {

        if(isReceiver()) {

            acquireLock();
            frameData.count = data.count;
            frameData.image = data.image;
            releaseLock();
            //std::cout << "2" << std::endl;

            //get frame from sender plugin
            senderImageQueue_.acquireLock();
            senderImageQueue_.push(frameData);
            senderImageQueue_.releaseLock();
            //std::cout << "4" << std::endl;
            //std::cout << frameData.count << std::endl;
            std::cout <<senderImageQueue_.size() << std::endl;
            
            numMessageReceived_++;
          
        }
        //updateMessageLabels();

    }


    // TEST DEVELOPMENT 
    
    void JaabaPlugin::copy_features1d(int frame_num, int num_elements, std::vector<float> &vec_feat, 
                                      float* array_feat) 
    {        
        // row to add is num_elements * frame_num
        int start_idx = frame_num * num_elements;
        for(int i = 0; i < num_elements; i++) {
            vec_feat[i + start_idx] = array_feat[i];
        }
    }


    int JaabaPlugin::createh5(std::string filename, int frame_num,
                              int num_frames, int hog_elements, int hof_elements,
                              std::vector<float> hog, std::vector<float> hof) 
    {
                 
        std::string out_file = filename + std::to_string(frame_num) + ".h5";
        //std::cout << out_file << std::endl;
        H5::H5File file(out_file.c_str(), H5F_ACC_TRUNC);

        // Create 2 datasets
        create_dataset(file, "hog", hog, num_frames, hog_elements);
        create_dataset(file, "hof", hof, num_frames, hof_elements);
    
        file.close();

        return 0;
    }

     
    void JaabaPlugin::create_dataset(H5::H5File& file, std::string key,
        std::vector<float> features, int num_frames, int num_elements) 
    {
  
        hsize_t dims[2];
        dims[0] = num_frames;
        dims[1] = num_elements;
        H5::DataSpace dataspace(2, dims);
        H5::DataSet dataset = file.createDataSet(key, H5::PredType::IEEE_F32LE, dataspace);

        dataset.write(&features.data()[0], H5::PredType::IEEE_F32LE);

        dataset.close();
        dataspace.close();
    }

    
    /*void JaabaPlugin::write_output_shape(std::string filename, std::string view, unsigned x, unsigned y, unsigned bin) 
    {

        std::ofstream x_out;
        x_out.open(filename.c_str(), std::ios_base::app);

        // write hist output to csv file
        x_out << view  << "," << x  <<  ","  << y << "," << bin << "\n"; 
        x_out.close();
    }*/


    /*void JaabaPlugin::read_output_shape(std::string filename)
    {

        std::ifstream x_in(filename);
        std::string lin;
        // read output shape from a csv file
        if(x_in)
        {
            while(getline(x_in, lin))
            {
                std::stringstream iss(lin);
                std::string result;
                while(std::getline(iss, result, ','))
                {

                    if(result == "front")
                    {
                        std::getline(iss, result, ',');
                        tmp_frontshape.x = atoi(result.c_str());
                        std::getline(iss, result, ',');
                        tmp_frontshape.y = atoi(result.c_str()); 
                        std::getline(iss, result, ',');
                        tmp_frontshape.bin = atoi(result.c_str());                                                         
                    }

                    if(result == "side")
                    {
                        std::getline(iss, result, ',');
                        tmp_sideshape.x = atoi(result.c_str());
                        std::getline(iss, result, ',');
                        tmp_sideshape.y = atoi(result.c_str());
                        std::getline(iss, result, ',');
                        tmp_sideshape.bin = atoi(result.c_str());
                    }

                }
            }
            std::cout << tmp_sideshape.x << " " << tmp_sideshape.y << " " << tmp_sideshape.bin << std::endl; 
            std::cout << tmp_frontshape.x << " " << tmp_frontshape.y <<  " " << tmp_frontshape.bin << std::endl;
        } else {

            std::cout << "File not present.Enter a valid filename." << std::endl;
            exit(1);
        }

    }*/

  
    void JaabaPlugin::write_score(std::string file, int framenum, float score)
    {

        std::ofstream x_out;
        x_out.open(file.c_str(), std::ios_base::app);

        // write score to csv file
        x_out << framenum << ","<< score << "\n";
        x_out.close();

    }


    void JaabaPlugin::read_score(std::string file_side, std::string file_front, int framenum)
    {

               
        std::ifstream xin_side(file_side);
        std::ifstream xin_front(file_front);
        std::string lin_side;
        std::string lin_front;
        float scr_side =0.0;
        float scr_front=0.0; 
        float acc_score=0.0;

        // read score side
        if(xin_side)
        {
            while(getline(xin_side, lin_side))
            {
                std::stringstream iss(lin_side);
                std::string result;
                while(std::getline(iss, result, ','))
                {
                    if(atoi(result.c_str()) == framenum)
                    {
                        std::getline(iss, result, ',');
                        scr_side = atof(result.c_str());
                        break;
                    }
                }
            }
        }


        // read score front
        if(xin_front)
        {
            while(getline(xin_front, lin_front))
            {
                std::stringstream iss(lin_front);
                std::string result;
                while(std::getline(iss, result, ','))
                {
                    if(atoi(result.c_str()) == framenum)
                    {
                        std::getline(iss, result, ',');
                        scr_front = atof(result.c_str());
                        break;
                    }
                }
            }
        }

        acc_score = scr_side + scr_front;
        //std::cout << "frame: " << framenum <<  " score: " << acc_score << std::endl;

    }


    void JaabaPlugin::write_histoutput(std::string file,float* out_img, unsigned w, unsigned h,unsigned nbins)
    {

	std::ofstream x_out;
	x_out.open(file.c_str());

	// write hist output to csv file
	for(int k=0;k < nbins;k++){
	    for(int i = 0;i < h;i++){
		for(int j = 0; j < w;j++){
		    x_out << out_img[k*w*h +i*w + j]; // i*h +j
		    if(j != w-1 || i != h-1 || k != nbins -1)
			x_out << ",";
		}
	    }
	}
    }

}

