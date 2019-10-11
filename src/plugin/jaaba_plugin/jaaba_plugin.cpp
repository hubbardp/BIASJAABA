#include "jaaba_plugin.hpp"
#include "image_label.hpp"
#include "camera_window.hpp"
#include <QMessageBox>
#include <iostream>
#include <QDebug>
#include <cuda_runtime.h>
#include <string>
#include <QThread>

#include "tictoc.h" 
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
  
        /*if( processScoresPtr_ != nullptr )
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
        }*/

    }


    void JaabaPlugin::reset()
    {
        
        if (isReceiver())
        {
            threadPoolPtr_ = new QThreadPool(this);
            threadPoolPtr_ -> setMaxThreadCount(5);
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
            qRegisterMetaType<ShapeData>("ShapeData");
            connect(partnerPluginPtr, SIGNAL(newFrameData(FrameData)), this, SLOT(onNewFrameData(FrameData)));
            connect(partnerPluginPtr, SIGNAL(newShapeData(ShapeData)), this, SLOT(onNewShapeData(ShapeData)));
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
            timeStamp = latestFrame.timeStamp;
            releaseLock();

            FrameData frameData;
            frameData.count = frameCount_;
            frameData.image = currentImage_;
            //TicTocTimer clock;
 
            if(isSender())
            {

                senderbuf_.enqueue(frameData);
                sizeQueue0 = senderbuf_.size();
                write_score("bufsize0.csv", frameCount_, sizeQueue0);
                /*acquireLock();
                if( !(HOGHOF_front.isNull()) && !HOGHOF_front->isInitialized)
                {
                    //cudaSetDevice(0);
                    //initHOGHOF(HOGHOF_front, 260, 384);
                    std::cout << "sender" << std::endl; 
                    initHOGHOF(HOGHOF_front, currentImage_.rows, currentImage_.cols);
                    if(HOGHOF_front->hog_outputbytes > 0){
                        std::cout << "sender initialized" << std::endl;
                        HOGHOF_front->isInitialized = true;
                    
                        ShapeData shapedata;
                        shapedata.shape_x = HOGHOF_front -> hog_shape.x;
                        shapedata.shape_y = HOGHOF_front -> hog_shape.y;
                        shapedata.shape_bin = HOGHOF_front -> hog_shape.bin;
                        emit(newShapeData(shapedata));
                    }

                }*/
               
                
                /*if(HOGHOF_front->isInitialized)
                {

                    //Test development capture framme and normalize frame
                    //if(capture_frt.isOpened())
                    //{
                    //    std::cout << "processing front" << std::endl;
                    //    curr_front = vid_frt->getImage(capture_frt); 
                    //    vid_frt->convertImagetoFloat(curr_front);
                    //    grey_frt = curr_front;
                    //}

                    curr_front = currentImage_;

                    if(curr_front.channels() == 3)
                    {                
                        cv::cvtColor(curr_front, curr_front, cv::COLOR_BGR2GRAY);
                    }

                    // convert the frame into float32
                    curr_front.convertTo(grey_frt, CV_32FC1);
                    grey_frt = grey_frt / 255;                         
                    HOGHOF_front->img.buf = grey_frt.ptr<float>(0);
                    genFeatures(HOGHOF_front, frameCount_);

                    if(classifier_front->isClassifierPathSet && frameCount_ > 0)
                    {

                        classifier_front->score = 0.0;
                        classifier_front->boost_classify_front(classifier_front->score, HOGHOF_front->hog_out, HOGHOF_front->hof_out,
                                                               &HOGHOF_front->hog_shape, classifier_front->nframes, frameCount_,
                                                               classifier_front->model);
                        //write_score("classifierscr_front.csv", frameCount_, classifier_front->score);
                    }
                    std::cout << "sender processed frame: " << frameCount_ << std::endl;
                }
                releaseLock();*/

                //emit(newFrameData(frameData));
                //std::cout << "I should be sending" << std::endl;
                //long int now = unix_timestamp(); 
                //clock = tic(); 
                //write_score("sender.csv",frameData.count, clock.last/1000000);
            }


            if(isReceiver())
            {

                receiverbuf_.enqueue(frameData);
                sizeQueue1 = receiverbuf_.size();
                write_score("bufsize1.csv", frameCount_, sizeQueue1);                               

                /*acquireLock();
                if(!(HOGHOF_side.isNull()) && !HOGHOF_side->isInitialized)
                {
                    //cudaSetDevice(1);
                    //initHOGHOF(HOGHOF_side, 260, 384);
                    std::cout << "receiver " << std::endl;
                    initHOGHOF(HOGHOF_side, currentImage_.rows, currentImage_.cols);
                    if(HOGHOF_side->hog_outputbytes > 0 ) 
                    {
                        std::cout << "receiver initialized " << std::endl;
                        HOGHOF_side->isInitialized = true; 
                        ShapeData shapedata;
                        shapedata.shape_x = HOGHOF_side -> hog_shape.x;
                        shapedata.shape_y = HOGHOF_side -> hog_shape.y;
                        shapedata.shape_bin = HOGHOF_side -> hog_shape.bin;
                        emit(newShapeData(shapedata));

                    }
                }*/


                /*if(HOGHOF_side->isInitialized)
                {


                    //Test development capture framme and normalize frame
                    //if(capture_sde.isOpened())
                    //{
                    //    std::cout << "processing side" << std::endl;
                    //    curr_side = vid_sde->getImage(capture_sde);
                    //    vid_sde->convertImagetoFloat(curr_side);
                    //    grey_sde = curr_side;
                    //}

                    curr_side = currentImage_;
               
                    if(curr_side.channels() == 3)
                    {
                        cv::cvtColor(curr_side, curr_side, cv::COLOR_BGR2GRAY);
                    }

                    curr_side.convertTo(grey_sde, CV_32FC1);
                    grey_sde = grey_sde / 255;
                    HOGHOF_side->img.buf = grey_sde.ptr<float>(0);
                    genFeatures(HOGHOF_side, frameCount_);

                    if(classifier_side->isClassifierPathSet && frameCount_ > 0)
                    {

                        classifier_side->score = 0.0;
                        classifier_side->boost_classify_side(classifier_side->score, HOGHOF_side->hog_out, HOGHOF_side->hof_out,
                                                             &HOGHOF_side->hog_shape,classifier_side->nframes, frameCount_,
                                                             classifier_side->model);
                        //write_score("classifierscr_side.csv", frameCount_, classifier_side->score);
                    }
 
                    std::cout << "receiver processed frame: " << frameCount_ << std::endl;
                }
                releaseLock();*/
 
                //acquireLock();
                //processScoresPtr_->enqueueFrameDataReceiver(frameData);
                //releaseLock();
                //std::cout << " I am receving side" << std::endl; 
                //long int now = unix_timestamp();
                //clock = tic();
                //write_score("receiver.csv",frameData.count, clock.last/1000000);
                
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
        //HOGHOF_side = new HOGHOF(this);
        //HOGHOF_front = new HOGHOF(this);
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
    
        //cudaSetDevice(2); 
        if(sideRadioButtonPtr_->isChecked())
        {
             
            QString file_sde = "/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_sde.avi";
            /*processScoresPtr_ -> vid_sde = new videoBackend(file_sde);
            processScoresPtr_ -> capture_sde = processScoresPtr_ -> vid_sde -> videoCapObject();*/
            vid_sde = new videoBackend(file_sde);
            capture_sde = vid_sde -> videoCapObject();
                 
  
            HOGHOF *hoghofside = new HOGHOF(this);
            acquireLock();
            /*processScoresPtr_->*/HOGHOF_side = hoghofside;
	    /*processScoresPtr_->*/HOGHOF_side->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            /*processScoresPtr_->*/HOGHOF_side->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            /*processScoresPtr_->*/HOGHOF_side->CropParam_file = pathtodir_->placeholderText() + CropSideParamFilePtr_->placeholderText();
            /*processScoresPtr_->*/HOGHOF_side->loadHOGParams();
            /*processScoresPtr_->*/HOGHOF_side->loadHOFParams();
            /*processScoresPtr_->*/HOGHOF_side->loadCropParams();
            releaseLock();

        }

        if(frontRadioButtonPtr_->isChecked()) 
        {

            QString file_frt = "/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_frt.avi";  
            /*processScoresPtr_ -> vid_frt = new videoBackend(file_frt); 
            processScoresPtr_ -> capture_frt = processScoresPtr_ -> vid_frt -> videoCapObject();*/ 
            vid_frt = new videoBackend(file_frt);
            capture_frt = vid_frt -> videoCapObject();

            HOGHOF *hoghoffront = new HOGHOF(this);  
            acquireLock();
            /*processScoresPtr_->*/HOGHOF_front = hoghoffront;
            /*processScoresPtr_->*/HOGHOF_front->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            /*processScoresPtr_->*/HOGHOF_front->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            /*processScoresPtr_->*/HOGHOF_front->CropParam_file = pathtodir_->placeholderText() + CropFrontParamFilePtr_->placeholderText();
            /*processScoresPtr_->*/HOGHOF_front->loadHOGParams();
            /*processScoresPtr_->*/HOGHOF_front->loadHOFParams();
            /*processScoresPtr_->*/HOGHOF_front->loadCropParams();
            releaseLock();
 
        }

    }


    void JaabaPlugin::setupClassifier() 
    {

        /*if(sideRadioButtonPtr_->isChecked() || frontRadioButtonPtr_->isChecked() && cameraNumber_ == 1)
        {
            
            beh_class *cls = new beh_class(this);
            processScoresPtr_ -> classifier = cls;
            processScoresPtr_ -> classifier->classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
            processScoresPtr_ -> classifier->allocate_model();
            processScoresPtr_ -> classifier->loadclassifier_model();
            
        }*/

        if(sideRadioButtonPtr_->isChecked() && cameraNumber_ == 1) {
     
            beh_class *cls = new beh_class(this);
            classifier_side = cls;
            classifier_side->classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
            classifier_side->allocate_model();
            classifier_side->loadclassifier_model();
        }

        if(frontRadioButtonPtr_->isChecked() && cameraNumber_ == 0) {

            beh_class *cls = new beh_class(this);
            classifier_front = cls;
            classifier_front->classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
            classifier_front->allocate_model();
            classifier_front->loadclassifier_model();
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
        detectButtonPtr_ -> setEnabled(false);
        saveButtonPtr_-> setEnabled(false);
        save = false;

        if(cameraNumber_ == 0)
        {
            
            //this->setEnabled(false);
            this->sideRadioButtonPtr_ ->setEnabled(false);           

        } else {

            //this->setEnabled(true);
            this->frontRadioButtonPtr_ ->setEnabled(false);

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

            if(/*processScoresPtr_-> HOGHOF_side->isHOGPathSet 
               && processScoresPtr_-> HOGHOF_side->isHOFPathSet
               && processScoresPtr_->classifier->isClassifierPathSet*/
               HOGHOF_side->isHOGPathSet 
               && HOGHOF_side->isHOFPathSet
               && classifier_side->isClassifierPathSet)
            {
                detectButtonPtr_->setEnabled(true);
                saveButtonPtr_->setEnabled(true);
            }
        
        }

        if(frontRadioButtonPtr_->isChecked())
        {

            /*if(processScoresPtr_-> HOGHOF_front->isHOGPathSet 
               && processScoresPtr_-> HOGHOF_front->isHOFPathSet
               && processScoresPtr_->classifier->isClassifierPathSet)*/
               if(HOGHOF_front->isHOGPathSet 
                  && HOGHOF_front->isHOFPathSet
                  && classifier_front->isClassifierPathSet)

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
        if(/*processScoresPtr_->*/HOGHOF_side == nullptr) 
        {
            setupHOGHOF();

        } else {

            
            /*processScoresPtr_->*/HOGHOF_side->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            /*processScoresPtr_->*/HOGHOF_side->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            /*processScoresPtr_->*/HOGHOF_side->CropParam_file = pathtodir_->placeholderText() + CropSideParamFilePtr_->placeholderText();
            /*processScoresPtr_->*/HOGHOF_side->loadHOGParams();
            /*processScoresPtr_->*/HOGHOF_side->loadHOFParams();
            /*processScoresPtr_->*/HOGHOF_side->loadCropParams();            
        }

        // load front HOGHOFParams if front view checked
        if(/*processScoresPtr_->*/HOGHOF_front == nullptr)
        {

            setupHOGHOF();        
  
        } else {

            /*processScoresPtr_->*/HOGHOF_front->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            /*processScoresPtr_->*/HOGHOF_front->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            /*processScoresPtr_->*/HOGHOF_front->CropParam_file = pathtodir_->placeholderText() + CropFrontParamFilePtr_->placeholderText();
            /*processScoresPtr_->*/HOGHOF_front->loadHOGParams();
            /*processScoresPtr_->*/HOGHOF_front->loadHOFParams();
            /*processScoresPtr_->*/HOGHOF_front->loadCropParams();
        }

        //load classifier
        //if(processScoresPtr_->classifier == nullptr)
        if(classifier_side==nullptr || classifier_front==nullptr)
        {

            setupClassifier();

        } else {

            //processScoresPtr_->classifier->classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
            //processScoresPtr_->classifier->allocate_model();
            //processScoresPtr_->classifier->loadclassifier_model();
            if(sideRadioButtonPtr_->isChecked()) {

                classifier_side->classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
                classifier_side->allocate_model();
                classifier_side->loadclassifier_model();
            }

            if(frontRadioButtonPtr_->isChecked()) {

                classifier_front->classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
                classifier_front->allocate_model();
                classifier_front->loadclassifier_model();
            }

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

    void JaabaPlugin::initHOGHOF(QPointer<HOGHOF> hoghof, int img_height, int img_width)
    {

        //hoghof->loadImageParams(384, 260);
        std::cout << img_height << " " << img_width << std::endl;
        struct HOGFeatureDims hogshape;
        struct HOGFeatureDims hofshape;
        hoghof->loadImageParams(img_width, img_height);
        struct HOGContext hogctx = HOGInitialize(logger, hoghof->HOGParams, img_width, img_height, hoghof->Cropparams);
        struct HOFContext hofctx = HOFInitialize(logger, hoghof->HOFParams, hoghof->Cropparams);
        if(hogctx.workspace != nullptr) {
           
            hoghof->hog_ctx = (HOGContext*)malloc(sizeof(hogctx));
            hoghof->hof_ctx = (HOFContext*)malloc(sizeof(hofctx));
            memcpy(hoghof->hog_ctx, &hogctx, sizeof(hogctx));
            memcpy(hoghof->hof_ctx, &hofctx, sizeof(hofctx));
            hoghof->startFrameSet = false;

            //allocate output bytes HOG/HOF per frame 
            hoghof->hog_outputbytes = HOGOutputByteCount(hoghof->hog_ctx);
            hoghof->hof_outputbytes = HOFOutputByteCount(hoghof->hof_ctx);

            //output shape 
            HOGOutputShape(&hogctx, &hogshape);
            HOFOutputShape(&hofctx, &hofshape);
            hoghof->hog_shape = hogshape;
            hoghof->hof_shape = hofshape;
            hoghof->hog_out.resize(hoghof->hog_shape.x * hoghof->hog_shape.y * hoghof->hog_shape.bin);
            hoghof->hof_out.resize(hoghof->hof_shape.x * hoghof->hof_shape.y * hoghof->hof_shape.bin);

        } else {

            std::cout << " hoghof Context not initialized" << std::endl;
        }
    }

 
    void JaabaPlugin::genFeatures(QPointer<HOGHOF> hoghof,int frame)
    {

        size_t hog_num_elements = hoghof->hog_shape.x * hoghof->hog_shape.y * hoghof->hog_shape.bin;
        size_t hof_num_elements = hoghof->hof_shape.x * hoghof->hof_shape.y * hoghof->hof_shape.bin;

        //Compute and copy HOG/HOF
        /*int nDevices;
        cudaError_t err = cudaGetDeviceCount(&nDevices);
        if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

        std::cout << "no of devices " << nDevices << std::endl;
        if(nDevices>=2)
        {*/
            //cudaSetDevice(0);
            HOFCompute(hoghof->hof_ctx, hoghof->img.buf, hof_f32); // call to compute and copy is asynchronous
            HOFOutputCopy(hoghof->hof_ctx, hoghof->hof_out.data(), hoghof->hof_outputbytes); // should be called one after 
                                                                                     // the other to get correct answer
            //cudaSetDevice(1);
            HOGCompute(hoghof->hog_ctx, hoghof->img);
            HOGOutputCopy(hoghof->hog_ctx, hoghof->hog_out.data(), hoghof->hog_outputbytes);

        //} else {

            //HOFCompute(hoghof->hof_ctx, hoghof->img.buf, hof_f32); // call to compute and copy is asynchronous
            //HOFOutputCopy(hoghof->hof_ctx, hoghof->hof_out.data(), hoghof->hof_outputbytes); // should be called one after 
                                                                                     // the other to get correct answer
            //HOGCompute(hoghof->hog_ctx, hoghof->img);
            //HOGOutputCopy(hoghof->hog_ctx, hoghof->hog_out.data(), hoghof->hog_outputbytes);

        //}

    }

    
    // Private Slots
    // ------------------------------------------------------------------------

    void JaabaPlugin::onNewFrameData(FrameData data)
    {

        if(isReceiver()) 
        {

            //get frame from sender plugin
            acquireLock();
            //processScoresPtr_->enqueueFrameDataSender(data);
            releaseLock();
            
            numMessageReceived_++;
          
        }        
    }


    void JaabaPlugin::onNewShapeData(ShapeData data)
    {

        if(isReceiver())
        {

            //get frame from sender plugin
            acquireLock();
            HOGShape hogshape;
            hogshape.x = data.shape_x;
            hogshape.y = data.shape_y;
            hogshape.bin = data.shape_bin;
            std::cout << hogshape.x << " " << hogshape.y << " " << hogshape.bin << std::endl;
            classifier_side->translate_mat2C(&HOGHOF_side->hog_shape, &hogshape);           
            releaseLock();

        }


        if(isSender())
        {

            acquireLock();
            HOGShape hogshape;
            hogshape.x = data.shape_x;
            hogshape.y = data.shape_y;
            hogshape.bin = data.shape_bin;
            std::cout << hogshape.x << " " << hogshape.y << " " << hogshape.bin << std::endl;
            classifier_front->translate_mat2C(&hogshape, &HOGHOF_front->hog_shape);
            releaseLock();

        }

    }


    void JaabaPlugin::write_score(std::string file, int framenum, long int timeStamp)
    {

        std::ofstream x_out;
        x_out.open(file.c_str(), std::ios_base::app);

        // write score to csv file
        x_out << framenum << ","<< timeStamp << "\n";
        x_out.close();

    }


    long int JaabaPlugin::unix_timestamp()
    {
        time_t t = std::time(0);
        long int now = static_cast<long int> (t);
        return now;
    }

    
}
