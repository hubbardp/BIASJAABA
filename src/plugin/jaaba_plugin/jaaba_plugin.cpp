#include "jaaba_plugin.hpp"
#include <iostream>
#include <QDebug>
#include <cuda_runtime.h>
#include <string>

#define DEBUG 0 
#define compute 0
#define isVidInput 1

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
                             std::shared_ptr<Lockable<GetTime>> gettime,
                             QWidget *parent) : BiasPlugin(parent)
    {

        nviews_ = numberOfCameras;
        threadPoolPtr_ = threadPoolPtr;
        gettime_ = gettime;
        nidaq_task_ = nullptr;

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

#if !isVidInput
        Format7Settings settings;
        settings = cameraPtr_->getFormat7Settings();
        image_height = settings.height;
        image_width = settings.width;
        std::cout << "height " << image_height << std::endl;
        std::cout << "width " << image_width << std::endl;

# else    
        //DEVEL
        if(processScoresPtr_side->isVid)
        {
            std::cout << "inside side " << std::endl;
            image_height = 384; //height of frame from video input
            image_width = 260;   //width of frame from video input 
            
        }

        if (processScoresPtr_front->isVid)
        {
            std::cout << "inside front" << std::endl;
            image_height = 384; //height of frame from video input
            image_width = 260;   //width of frame from video input 
        }
#endif

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
                    && processScoresPtr_side->classifier->isClassifierPathSet)
                {

                    HOFTeardown(processScoresPtr_side->HOGHOF_frame->hof_ctx);
                    HOGTeardown(processScoresPtr_side->HOGHOF_frame->hog_ctx);
                    delete processScoresPtr_side->classifier;
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
                    && processScoresPtr_side->classifier->isClassifierPathSet)
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
                //threadPoolPtr_->start(processScoresPtr_side);
            }
           
            // this thread adds latency to process frames
            /*if((threadPoolPtr_ != nullptr) && (visplots != nullptr))
            {
                threadPoolPtr_->start(visplots);
            }*/
        }

    }


    void JaabaPlugin::gpuInit()
    {

        //cv::Mat dummyImage;

        // intitialize the HOGHOF context for the GPU memory
        getFormatSettings();
        std::cout << image_height << " " << image_width << std::endl;
        //dummyImage.zeros(image_width, image_height, 0.0);

        if (!processScoresPtr_side->isHOGHOFInitialised)
        {
            
            if (!(processScoresPtr_side->HOGHOF_frame.isNull()))
            {
                
                if (nDevices_ >= 2)
                {

                    cudaSetDevice(0);
                    processScoresPtr_side->initHOGHOF(processScoresPtr_side->HOGHOF_frame, image_width, image_height);
                    //processScoresPtr_side->HOGHOF_frame->img.buf = dummyImage.ptr<float>(0);
                    //processScoresPtr_side->genFeatures(processScoresPtr_side->HOGHOF_frame, 0);
                    //emit(passFrontHOFShape(processScoresPtr_side->HOGHOF_frame));

                }else {

                    processScoresPtr_side->initHOGHOF(processScoresPtr_side->HOGHOF_frame, image_width, image_height);
                    //emit(passSideHOGShape(processScoresPtr_front->HOGHOF_partner));
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
                    //processScoresPtr_front->HOGHOF_partner->img.buf = dummyImage.ptr<float>(0);
                    //processScoresPtr_front->genFeatures(processScoresPtr_front->HOGHOF_partner, 0);
                
                }else {

                    processScoresPtr_front->initHOGHOF(processScoresPtr_front->HOGHOF_partner, image_width, image_height);
                }

            }
           
        }

        // histogram parameters initialize
        if (mesPass) {

            if (processScoresPtr_side->isHOGHOFInitialised && processScoresPtr_front->isHOGHOFInitialised)
            {

                processScoresPtr_side->classifier->translate_mat2C(&processScoresPtr_side->HOGHOF_frame->hog_shape,
                    &processScoresPtr_front->HOGHOF_partner->hog_shape);
                std::cout << processScoresPtr_front->HOGHOF_partner->hog_shape.x
                    << processScoresPtr_front->HOGHOF_partner->hog_shape.y
                    << std::endl;
            
            }

        } else {

            /*if (isReceiver()) 
            {
                if (processScoresPtr_side->isHOGHOFInitialised && processScoresPtr_side->isHOGHOFInitialised)
                {

                    classifier->translate_mat2C(&processScoresPtr_side->HOGHOF_frame->hog_shape,
                                                &processScoresPtr_side->HOGHOF_frame->hog_shape);
                }
            }*/
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
            qRegisterMetaType<QPointer<HOGHOF>>("QPointer<HOGHOF>");
            qRegisterMetaType<QPointer<HOGHOF>>("QPointer<HOGHOF>");
            qRegisterMetaType<PredData>("PredData");
            qRegisterMetaType<unsigned int>("unsigned int");
            qRegisterMetaType<int64_t>("int64_t");
            qRegisterMetaType<bool>("bool");
            connect(partnerPluginPtr, SIGNAL(partnerImageQueue(std::shared_ptr<LockableQueue<StampedImage>>)),
                this, SLOT(onPartnerPlugin(std::shared_ptr<LockableQueue<StampedImage>>)));
            connect(partnerPluginPtr, SIGNAL(passSideHOGShape(QPointer<HOGHOF>)),
                this, SLOT(onFrameHOGShape(QPointer<HOGHOF>)));
            connect(partnerPluginPtr, SIGNAL(passFrontHOFShape(QPointer<HOGHOF>)),
                this, SLOT(onFrameHOFShape(QPointer<HOGHOF>)));
            connect(partnerPluginPtr, SIGNAL(passScore(PredData)),
                this, SLOT(scoreCompute(PredData)));
            
            //connect(partnerPluginPtr, SIGNAL(passFrameNum(unsigned int)),
            //        this, SLOT(receiveFrameNum(unsigned int)));
            connect(partnerPluginPtr, SIGNAL(passFrameRead(int64_t, int)),
                this, SLOT(receiveFrameRead(int64_t, int)));
            connect(partnerPluginPtr, SIGNAL(passScoreDone(bool)),
                this, SLOT(scoreCalculated(bool)));
            connect(partnerPluginPtr, SIGNAL(doNotProcess(unsigned int)),
                this, SLOT(setSkipFrameProcess(unsigned int)));
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

    void JaabaPlugin::processFrames()
    {
        
        if(mesPass)
        {

            processFramePass();

        } else {

            processFrame_inPlugin();

        }

    }

    void JaabaPlugin::processFrame_inPlugin()
    {

        cv::Mat sideImage;
        cv::Mat frontImage;
        uInt32 read_buffer = 0, read_ondemand = 0;
        cv::Mat greySide;
        cv::Mat greyFront;

        //DEVEL
        int64_t pc_time;
        uint64_t start_process=0, end_process=0;
        int64_t front_read_time, side_read_time, time_now;
        uint64_t start_delay, end_delay;
        string filename;
        double expTime = 0, curTime = 0;
        uint64_t curTime_vid=0, expTime_vid=0;
        uint64_t frameGrabAvgTime, max_jaaba_compute_time;
        int64_t wait_thres, avgwait_time;
      

#if isVidInput
        frameGrabAvgTime = 2500;
        max_jaaba_compute_time = 500;
        wait_thres = 2000;
        avgwait_time = 0;
        
#else
        frameGrabAvgTime = 2500;
        wait_thres = static_cast<float>(4000/1000);
#endif
       
        if (pluginImageQueuePtr_ != nullptr)
        {
            
            pluginImageQueuePtr_->acquireLock();
            pluginImageQueuePtr_->waitIfEmpty();

            if (pluginImageQueuePtr_->empty())
            {
                pluginImageQueuePtr_->releaseLock();
                return;
            }
            
            if (!(pluginImageQueuePtr_->empty()))
            {
                
                StampedImage stampedImage0 = pluginImageQueuePtr_->front();
                pluginImageQueuePtr_->pop();
                pluginImageQueuePtr_->releaseLock();
                currentImage_ = stampedImage0.image.clone();
                frameCount_ = stampedImage0.frameCount;
                fstfrmtStampRef_ = stampedImage0.fstfrmtStampRef;
                
#if !isVidInput
                if (testConfigEnabled_ && nidaq_task_ != nullptr) {

                    if (frameCount_ <= testConfig_->numFrames) {

                        nidaq_task_->acquireLock();
                        DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_grab_in, 10.0, &read_ondemand, NULL));
                        nidaq_task_->releaseLock();

                    }

                }

#endif
                if (isReceiver() && processScoresPtr_side->processedFrameCount == 0)
                {
                    emit(passFrontHOFShape(processScoresPtr_side->HOGHOF_frame));
                }

                if (isSender() && processScoresPtr_front->processedFrameCount == 0)
                {
                    emit(passSideHOGShape(processScoresPtr_front->HOGHOF_partner));
                }

                start_process = gettime_->getPCtime();
                if (fstfrmtStampRef_ != 0)
                {
#if isVidInput     
                    time_now = gettime_->getPCtime();
                    expTime_vid = fstfrmtStampRef_ + (frameGrabAvgTime * (frameCount_ + 1)) + max_jaaba_compute_time;
                    curTime_vid = static_cast<uint64_t>(time_now);
                    avgwait_time = curTime_vid - expTime_vid;

#else                   
                    expTime = (fstfrmtStampRef_*0.02 + (2.5 * frameCount_));
                    //expTime = nidaq_task_->cam_trigger[frameCount_]*0.02;
                    curTime = (read_ondemand)*0.02; 
#endif
                    
                    if (avgwait_time > wait_thres)
                    {
                        if (cameraNumber_ == 0 && (processScoresPtr_side->processedFrameCount == frameCount_))
                        {

                            side_skip_count++;
                            //ts_nidaqThres[processScoresPtr_side->processedFrameCount] = side_skip_count;
                            processScoresPtr_side->processedFrameCount++;                           

                        }else { assert(processScoresPtr_side->processedFrameCount==frameCount_); }

                        if (cameraNumber_ == 1 && (processScoresPtr_front->processedFrameCount == frameCount_))
                        {

                            front_skip_count++;
                            //ts_nidaqThres[processScoresPtr_front->processedFrameCount] = front_skip_count;
                            processScoresPtr_front->processedFrameCount++;

                        }else{  assert(processScoresPtr_front->processedFrameCount==frameCount_);}

                    } else { 
                        
                        if (currentImage_.rows != 0 && currentImage_.cols != 0)
                        {
#ifndef DEBUG
                            acquireLock();
                            frameCount_ = stampedImage0.frameCount;
                            if (isSender())
                                emit(passFrame(frameCount_));
                            releaseLock();
#endif

                            if (isSender() && detectStarted)
                            {
#if isVidInput                     
                                // convert the frame into float32
                                frontImage = stampedImage0.image.clone();
                                frontImage.convertTo(greyFront, CV_32FC1);
                                greyFront = greyFront / 255;
                  

#else                   
                                frontImage = stampedImage0.image.clone();
                                if (frontImage.channels() == 3)
                                {
                                    cv::cvtColor(frontImage, frontImage, cv::COLOR_BGR2GRAY);
                                }

                                // convert the frame into float32
                                frontImage.convertTo(greyFront, CV_32FC1);
                                greyFront = greyFront / 255;
#endif
                            }


                            if (isReceiver() && detectStarted)
                            {

#if isVidInput                               
                                sideImage = stampedImage0.image.clone();
                                // convert the frame into float32
                                sideImage.convertTo(greySide, CV_32FC1);
                                greySide = greySide / 255;

#else
                                sideImage = stampedImage0.image.clone();
                                if (sideImage.channels() == 3)
                                {
                                    cv::cvtColor(sideImage, sideImage, cv::COLOR_BGR2GRAY);
                                }

                                // convert the frame into float32
                                sideImage.convertTo(greySide, CV_32FC1);
                                greySide = greySide / 255;
#endif                    
                            }
                            
                            if (processScoresPtr_side != nullptr || processScoresPtr_front != nullptr)
                            {

                                if (processScoresPtr_front->isFront)
                                {
#if compute
                                    if (nDevices_ >= 2)
                                    {
                                        cudaSetDevice(1);
                                        processScoresPtr_front->HOGHOF_partner->img.buf = greyFront.ptr<float>(0);
                                        processScoresPtr_front->genFeatures(processScoresPtr_front->HOGHOF_partner, frameCount_);

                                    }
                                    else {

                                        processScoresPtr_front->HOGHOF_partner->img.buf = greyFront.ptr<float>(0);
                                        processScoresPtr_front->genFeatures(processScoresPtr_front->HOGHOF_partner, frameCount_);

                                    }

                                    if (processScoresPtr_front->classifier->isClassifierPathSet &&
                                        processScoresPtr_front->processedFrameCount > 0)
                                    {

                                        processScoresPtr_front->classifier->boost_classify_front(processScoresPtr_front->classifier->predScoreFront.score,
                                            processScoresPtr_front->HOGHOF_partner->hog_out, processScoresPtr_front->HOGHOF_partner->hof_out, &partner_hogshape_,
                                            &processScoresPtr_front->HOGHOF_partner->hof_shape, processScoresPtr_front->classifier->nframes,
                                            processScoresPtr_front->classifier->model);

                                        time_now = gettime_->getPCtime();
                                        processScoresPtr_front->classifier->predScoreFront.frameCount = processScoresPtr_front->processedFrameCount;
                                        processScoresPtr_front->classifier->predScoreFront.score_ts = time_now;
                                        processScoresPtr_front->classifier->predScoreFront.view = 2;
                                        processScoresPtr_front->classifier->predscore_front[processScoresPtr_front->processedFrameCount]
                                            = processScoresPtr_front->classifier->predScoreFront;
                                        emit(passScore(processScoresPtr_front->classifier->predScoreFront));

                                        
                                    }
#endif
                                    processScoresPtr_front->processedFrameCount++;

                                }
                                else {

                                    //processScoresPtr_front->HOGHOF_partner->setLastInput();
                                    //processScoresPtr_front->skip_frameFront = 0;
                                    //processScoresPtr_front->processedFrameCount++;

                                }

                                if (processScoresPtr_side->isSide)
                                {
#if compute
                                    if (nDevices_ >= 2)
                                    {
                                        cudaSetDevice(0);
                                        processScoresPtr_side->HOGHOF_frame->img.buf = greySide.ptr<float>(0);
                                        processScoresPtr_side->genFeatures(processScoresPtr_side->HOGHOF_frame, frameCount_);

                                    }
                                    else {

                                        processScoresPtr_side->HOGHOF_frame->img.buf = greySide.ptr<float>(0);
                                        processScoresPtr_side->genFeatures(processScoresPtr_side->HOGHOF_frame, frameCount_);
                                    }


                                    if (processScoresPtr_side->classifier->isClassifierPathSet &&
                                        processScoresPtr_side->processedFrameCount > 0)
                                    {
                                        
                                        processScoresPtr_side->classifier->boost_classify_side(processScoresPtr_side->classifier->predScoreSide.score,
                                            processScoresPtr_side->HOGHOF_frame->hog_out, processScoresPtr_side->HOGHOF_frame->hof_out,
                                            &processScoresPtr_side->HOGHOF_frame->hog_shape, &partner_hofshape_, processScoresPtr_side->classifier->nframes,
                                            processScoresPtr_side->classifier->model);
                                        
                                        time_now = gettime_->getPCtime();
                                        processScoresPtr_side->classifier->predScoreSide.frameCount = processScoresPtr_side->processedFrameCount;
                                        processScoresPtr_side->classifier->predScoreSide.score_ts = time_now;
                                        processScoresPtr_side->classifier->predScoreSide.view = 1;
                                        processScoresPtr_side->isProcessed_side = 1;

                                        processScoresPtr_side->acquireLock();
                                        processScoresPtr_side->sideScoreQueue.push_back(processScoresPtr_side->classifier->predScoreSide);
                                        processScoresPtr_side->classifier->predscore_side[processScoresPtr_side->processedFrameCount] 
                                                                                    = processScoresPtr_side->classifier->predScoreSide;
                                        processScoresPtr_side->releaseLock();
                                       
                                    }
#endif
                                    processScoresPtr_side->processedFrameCount++;

                                }
                                else {

                                    //processScoresPtr_side->HOGHOF_frame->setLastInput();
                                    //processScoresPtr_side->processedFrameCount++;

                                }
                                
                            }

                        }
                        
                    }// if skip or process 
                    //pluginImageQueuePtr_->pop();
                }  
                end_process = gettime_->getPCtime();

            }// check if plugin queue empty
            //pluginImageQueuePtr_->releaseLock();
        }

        /*if (nidaq_task_ != nullptr) {

            if (frameCount_ <= testConfig_->numFrames) {

                nidaq_task_->acquireLock();
                DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_grab_in, 10.0, &read_ondemand, NULL));
                nidaq_task_->releaseLock();

            }

        }*/

        if (testConfigEnabled_ && frameCount_ < testConfig_->numFrames)
        {

            if (!testConfig_->nidaq_prefix.empty()) {

                ts_nidaq[frameCount_][0] = nidaq_task_->cam_trigger[frameCount_];
                ts_nidaq[frameCount_][1] = read_ondemand;
                
            }

            if (!testConfig_->f2f_prefix.empty()) {

                pc_time = gettime_->getPCtime();
                ts_pc[frameCount_] = pc_time;
            }

            if (!testConfig_->queue_prefix.empty()) {

                //queue_size[frameCount_] = 2;// pluginImageQueuePtr_->size();
                if(cameraNumber_ == 0)
                    queue_size[frameCount_] = side_skip_count;
                else if (cameraNumber_ == 1)
                    queue_size[frameCount_] = front_skip_count;
                  
            }

            if (process_frame_time) {
                
                ts_gpuprocess_time[frameCount_] = (end_process - start_process);
                ts_pc[frameCount_] = start_process;
                ts_nidaqThres[frameCount_] = curTime_vid - expTime_vid;
                
            }

            if (frameCount_ == testConfig_->numFrames - 1
                && !testConfig_->f2f_prefix.empty())
            {

                std::string filename = testConfig_->dir_list[0] + "/"
                    + testConfig_->f2f_prefix + "/" + testConfig_->cam_dir
                    + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                    + testConfig_->plugin_prefix
                    + "_" + testConfig_->f2f_prefix + "cam"
                    + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";

                gettime_->write_time_1d<int64_t>(filename, testConfig_->numFrames, ts_pc);

            }

            if (frameCount_ == testConfig_->numFrames - 1
                && !testConfig_->nidaq_prefix.empty())
            {

                std::string filename = testConfig_->dir_list[0] + "/"
                    + testConfig_->nidaq_prefix + "/" + testConfig_->cam_dir
                    + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                    + testConfig_->plugin_prefix
                    + "_" + testConfig_->nidaq_prefix + "cam"
                    + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";

                std::string filename1 = testConfig_->dir_list[0] + "/"
                    + testConfig_->nidaq_prefix + "/" + testConfig_->cam_dir
                    + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                    + testConfig_->plugin_prefix
                    + "_" + testConfig_->nidaq_prefix + "_thres" + "cam"
                    + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";

                gettime_->write_time_1d<int64_t>(filename1, testConfig_->numFrames, ts_nidaqThres);
                gettime_->write_time_2d<uInt32>(filename, testConfig_->numFrames, ts_nidaq);
            }

            if (frameCount_ == testConfig_->numFrames - 1
                && !testConfig_->queue_prefix.empty()) {


                string filename = testConfig_->dir_list[0] + "/"
                    + testConfig_->queue_prefix + "/" + testConfig_->cam_dir
                    + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                    + testConfig_->plugin_prefix
                    + "_" + testConfig_->queue_prefix + "cam"
                    + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";
                std::cout << frameCount_ << "" << filename << std::endl;
                gettime_->write_time_1d<unsigned int>(filename, testConfig_->numFrames, queue_size);

            }

            if (frameCount_ == testConfig_->numFrames - 1
                && process_frame_time) {

                string filename = testConfig_->dir_list[0] + "/"
                    + testConfig_->nidaq_prefix + "/" + testConfig_->cam_dir
                    + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                    + testConfig_->plugin_prefix
                    + "_" + "process_time" + "cam"
                    + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";

                string filename1 = testConfig_->dir_list[0] + "/"
                    + testConfig_->nidaq_prefix + "/" + testConfig_->cam_dir
                    + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                    + testConfig_->plugin_prefix
                    + "_" + "start_time" + "cam"
                    + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";

                
                gettime_->write_time_1d<int64_t>(filename, testConfig_->numFrames, ts_gpuprocess_time);
                gettime_->write_time_1d<int64_t>(filename1, testConfig_->numFrames, ts_pc);

            }
        }else{}

    }

    //void JaabaPlugin::processFrames(QList<StampedImage> frameList)
    void JaabaPlugin::processFramePass()
    {

        cv::Mat sideImage;
        cv::Mat frontImage;
        cv::Mat greySide;
        cv::Mat greyFront;
        uInt32 read_buffer = 0, read_ondemand = 0;

        StampedImage stampedImage0 , stampedImage1;

        //DEVEL
        int64_t pc_time, start_process=0, end_process=0;
            
        // initialize memory on the gpu 
        if (isReceiver() && ((!processScoresPtr_side->isHOGHOFInitialised) || (!processScoresPtr_front->isHOGHOFInitialised)))
        {
            gpuInit();
        }

        // Send frames from front plugin to side
        if(pluginImageQueuePtr_ != nullptr && isSender())
        {    
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

            start_process = gettime_->getPCtime();
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

#if DEBUG
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
                    
#endif

                    if( stampedImage0.frameCount == stampedImage1.frameCount)
                    {
                        
                        if((processScoresPtr_side -> detectStarted_) && (frameCount_  == (processScoresPtr_side -> processedFrameCount+1)))
                        {

#if DEBUG
                            // Test - Uncomment to perform preprocesing of video frames
                            if (processScoresPtr_side->capture_sde.isOpened())
                            {
                                if (processScoresPtr_side->processedFrameCount < nframes_)
                                {

                                    sideImage = processScoresPtr_side->vid_sde->getImage(processScoresPtr_side->capture_sde);
                                    processScoresPtr_side->vid_sde->convertImagetoFloat(sideImage);
                                    greySide = sideImage;

                                } else {

                                    return;
                                }

                            }

                            if (processScoresPtr_front->capture_front.isOpened())
                            {
                                if (processScoresPtr_front->processedFrameCount < nframes_)
                                {

                                    frontImage = processScoresPtr_front->vid_front->getImage(processScoresPtr_front->capture_front);
                                    processScoresPtr_front->vid_front->convertImagetoFloat(frontImage);
                                    greyFront = frontImage;

                                } else {

                                    return;
                                }

                            }
                            else {

                                processScoresPtr_front->vid_front->releaseCapObject(processScoresPtr_front->capture_front);
                                
                            }

#else

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

#endif
     
                            if(nDevices_>=2)
                            {
                                    
                                if(processScoresPtr_side -> isSide)
                                {

                                    cudaSetDevice(0);
                                    processScoresPtr_side -> HOGHOF_frame->img.buf = greySide.ptr<float>(0);
                                    processScoresPtr_side -> onProcessSide();
                                  
                                }

                                if(processScoresPtr_front -> isFront)
                                {

                                    cudaSetDevice(1);
                                    processScoresPtr_front -> HOGHOF_partner->img.buf = greyFront.ptr<float>(0);
                                    processScoresPtr_front -> genFeatures(processScoresPtr_front -> HOGHOF_partner, frameCount_);
                                    //processScoresPtr_front->onProcessFront();
                                    
                                }

                                while (!processScoresPtr_side->isProcessed_side) {}

                            } else {

                                processScoresPtr_side -> HOGHOF_frame -> img.buf = greySide.ptr<float>(0);
                                processScoresPtr_side -> genFeatures(processScoresPtr_side -> HOGHOF_frame, frameCount_);
                                processScoresPtr_front -> HOGHOF_partner -> img.buf = greyFront.ptr<float>(0);
                                processScoresPtr_front -> genFeatures(processScoresPtr_front -> HOGHOF_partner, frameCount_);

                            }
                            

#if DEBUG
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

#endif

                            // compute scores
                            if(processScoresPtr_side->classifier->isClassifierPathSet & processScoresPtr_side->processedFrameCount >= 0)
                            {

                                //std::fill(laserRead.begin(), laserRead.end(), 0);
                                processScoresPtr_side->classifier->boost_classify_side(processScoresPtr_side->classifier->predScoreSide.score, 
                                 processScoresPtr_side->HOGHOF_frame->hog_out, processScoresPtr_side -> HOGHOF_frame->hof_out, 
                                 &processScoresPtr_side->HOGHOF_frame->hog_shape, &processScoresPtr_front ->HOGHOF_partner->hof_shape, 
                                    processScoresPtr_side->classifier-> nframes, processScoresPtr_side->classifier->model);

                                processScoresPtr_side->classifier->boost_classify_front(processScoresPtr_side->classifier->predScoreFront.score, 
                                    processScoresPtr_front->HOGHOF_partner->hog_out, processScoresPtr_front->HOGHOF_partner->hof_out, 
                                    &processScoresPtr_side->HOGHOF_frame->hog_shape, &processScoresPtr_front->HOGHOF_partner->hof_shape, 
                                    processScoresPtr_side->classifier->nframes, processScoresPtr_side->classifier->model);
                                
                                processScoresPtr_side->classifier->addScores(processScoresPtr_side->classifier->predScoreSide.score,
                                                                             processScoresPtr_side->classifier->predScoreFront.score);
                                processScoresPtr_side->write_score("classifierscr.csv", processScoresPtr_side->processedFrameCount,
                                    processScoresPtr_side->classifier->finalscore);
#if DEBUG
                                //triggerLaser();
                                /*visplots -> livePlotTimeVec_.append(stampedImage0.timeStamp);
                                visplots -> livePlotSignalVec_Lift.append(double(classifier->score[0]));
                                visplots -> livePlotSignalVec_Handopen.append(double(classifier->score[1]));
                                visplots -> livePlotSignalVec_Grab.append(double(classifier->score[2]));
                                visplots -> livePlotSignalVec_Supinate.append(double(classifier->score[3]));
                                visplots -> livePlotSignalVec_Chew.append(double(classifier->score[4]));
                                visplots -> livePlotSignalVec_Atmouth.append(double(classifier->score[5]));
                                visplots->livePlotPtr_->show();*/
                                
#endif
                            }
                           
                            //std::cout << frameCount_ << " "  << processScoresPtr_side->processedFrameCount << std::endl;
                            processScoresPtr_side -> processedFrameCount = frameCount_;
                            processScoresPtr_front -> processedFrameCount = frameCount_; 
                            processScoresPtr_side->isProcessed_side = false;
                            //processScoresPtr_front->isProcessed_front = false;

                        }else{ std::cout << "skipped 1 " << frameCount_ << std::endl; }

                    }else{ std::cout << "skipped 2 " << frameCount_ << std::endl; }
                     
                }else { std::cout << "skipped 3" << frameCount_ << std::endl;}
                    
                pluginImageQueuePtr_->pop();
                partnerPluginImageQueuePtr_ -> pop();
             
            }else { std::cout << "skipped 4 " << frameCount_ << std::endl; }

            pluginImageQueuePtr_->releaseLock();
            partnerPluginImageQueuePtr_ -> releaseLock();         
            end_process = gettime_->getPCtime();
#if DEBUG            
            if (testConfigEnabled_ && !testConfig_->imagegrab_prefix.empty()
                && testConfig_->plugin_prefix == "jaaba_plugin")
            {

                if (nidaq_task_ != nullptr) {

                    if (frameCount_ <= testConfig_->numFrames) {

                        nidaq_task_->acquireLock();
                        DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task_->taskHandle_grab_in, 10.0, &read_ondemand, NULL));
                        nidaq_task_->releaseLock();

                    }

                }

                if (!testConfig_->nidaq_prefix.empty()) {

                    if (cameraNumber_ == 0)
                        ts_nidaq[frameCount_][0] = nidaq_task_->cam_trigger[frameCount_];
                    else
                        ts_nidaq[frameCount_][0] = 0;

                    ts_nidaq[frameCount_][1] = read_ondemand;
                }

                if (!testConfig_->f2f_prefix.empty()) {

                    pc_time = gettime_->getPCtime();

                    if (frameCount_ <= testConfig_->numFrames)
                        ts_pc[frameCount_] = pc_time;
                }

                if (!testConfig_->queue_prefix.empty()) {

                    if (frameCount_ <= testConfig_->numFrames)
                        queue_size[frameCount_] = pluginImageQueuePtr_->size();

                }

                if (process_frame_time) {

                    if (frameCount_ <= testConfig_->numFrames)
                        ts_nidaqThres[frameCount_] = (end_process - start_process);
                }

                if (frameCount_ == testConfig_->numFrames - 1
                    && !testConfig_->f2f_prefix.empty())
                {

                    std::string filename = testConfig_->dir_list[0] + "/"
                        + testConfig_->f2f_prefix + "/" + testConfig_->cam_dir
                        + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                        + testConfig_->plugin_prefix
                        + "_" + testConfig_->f2f_prefix + "cam"
                        + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";
                    
                    gettime_->write_time_1d<int64_t>(filename, testConfig_->numFrames, ts_pc);

                }

                if (frameCount_ == testConfig_->numFrames - 1
                    && !testConfig_->nidaq_prefix.empty())
                {

                    std::string filename = testConfig_->dir_list[0] + "/"
                        + testConfig_->nidaq_prefix + "/" + testConfig_->cam_dir
                        + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                        + testConfig_->plugin_prefix
                        + "_" + testConfig_->nidaq_prefix + "cam"
                        + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";

                    gettime_->write_time_2d<uInt32>(filename, testConfig_->numFrames, ts_nidaq);

                }

                if (frameCount_ == testConfig_->numFrames - 1
                    && !testConfig_->queue_prefix.empty()) {

                    string filename = testConfig_->dir_list[0] + "/"
                        + testConfig_->queue_prefix + "/" + testConfig_->cam_dir
                        + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                        + testConfig_->plugin_prefix
                        + "_" + testConfig_->queue_prefix + "cam"
                        + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";

                    gettime_->write_time_1d<unsigned int>(filename, testConfig_->numFrames, queue_size);

                }

                if (frameCount_ == testConfig_->numFrames - 1
                    && process_frame_time) {

                    string filename = testConfig_->dir_list[0] + "/"
                        + testConfig_->f2f_prefix + "/" + testConfig_->cam_dir
                        + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                        + testConfig_->plugin_prefix
                        + "_" + "jaaba_process_time" + "cam"
                        + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";
                    std::cout << filename << std::endl;
                    gettime_->write_time_1d<float>(filename, testConfig_->numFrames, ts_nidaqThres);

                }
            }
#endif
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
   
        numMessageSent_ = 0;
        numMessageReceived_ = 0;
        frameCount_ = 0;
        laserOn = false;
        mesPass = false;
        score_calculated_ = 0;
        scoreCount = 0;
        frameSkip = 5;
        process_frame_time = 1; 

        processScoresPtr_side = new ProcessScores(this, mesPass, gettime_);   
        processScoresPtr_front = new ProcessScores(this, mesPass, gettime_);  

        visplots = new VisPlots(livePlotPtr,this);
 
        updateWidgetsOnLoad();
        setupHOGHOF();
        setupClassifier();

    }


    /*void JaabaPlugin::initiateVidSkips(priority_queue<int, vector<int>, greater<int>>& skip_frames)
                                      
    {

        //srand(time(NULL));
        int framenumber;

        for (int j = 0; j < no_of_skips; j++)
        {
            framenumber = rand() % nframes_;
            skip_frames.push(framenumber);
            std::cout << "frame skipped " << framenumber << std::endl;
        }

    }*/


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

        connect(gpuInitButtonPtr_,
            SIGNAL(clicked()),
            this,
            SLOT(gpuInit())
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
             
#if isVidInput

            if (isReceiver())
                processScoresPtr_side->isVid = 1;

#endif
            
            HOGHOF *hoghofside = new HOGHOF(this);

            if (processScoresPtr_side != nullptr)
            {
                printf("processScores Side allocated");
                acquireLock();
                processScoresPtr_side->HOGHOF_frame = hoghofside;
                processScoresPtr_side->HOGHOF_frame->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
                processScoresPtr_side->HOGHOF_frame->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
                processScoresPtr_side->HOGHOF_frame->CropParam_file = pathtodir_->placeholderText() + CropSideParamFilePtr_->placeholderText();
                processScoresPtr_side->HOGHOF_frame->loadHOGParams();
                processScoresPtr_side->HOGHOF_frame->loadHOFParams();
                processScoresPtr_side->HOGHOF_frame->loadCropParams();
                releaseLock();

            }else {

                printf("processScores Side not allocated");
            }

        }

        if(frontRadioButtonPtr_->isChecked()) 
        {

#if isVidInput
//DEVEL
           
            if (isSender())
                processScoresPtr_front->isVid = 1;

#endif

            HOGHOF *hoghoffront = new HOGHOF(this);  
            if (processScoresPtr_front != nullptr)
            {
                printf("processScores Front allocated");
                acquireLock();
                processScoresPtr_front->HOGHOF_partner = hoghoffront;
                processScoresPtr_front->HOGHOF_partner->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
                processScoresPtr_front->HOGHOF_partner->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
                processScoresPtr_front->HOGHOF_partner->CropParam_file = pathtodir_->placeholderText() + CropFrontParamFilePtr_->placeholderText();
                processScoresPtr_front->HOGHOF_partner->loadHOGParams();
                processScoresPtr_front->HOGHOF_partner->loadHOFParams();
                processScoresPtr_front->HOGHOF_partner->loadCropParams();
                releaseLock();

            } else {
                printf("processScores Front not allocated");
            }
        }
    }


    void JaabaPlugin::setupClassifier() 
    {
        if (frontRadioButtonPtr_->isChecked() || sideRadioButtonPtr_->isChecked()) {
            
            if (mesPass && isReceiver())
            {
                beh_class *cls = new beh_class(this);
                processScoresPtr_side->classifier = cls;
                processScoresPtr_side->classifier->classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
                //qDebug()  << classifier->classifier_file;
                processScoresPtr_side->classifier->allocate_model();
                processScoresPtr_side->classifier->loadclassifier_model();

            }

            if (!mesPass)
            {
                if (isReceiver()) {

                    beh_class *cls = new beh_class(this);
                    processScoresPtr_side->classifier = cls;
                    processScoresPtr_side->classifier->classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
                    //qDebug()  << classifier->classifier_file;
                    processScoresPtr_side->classifier->allocate_model();
                    processScoresPtr_side->classifier->loadclassifier_model();
                }

                if (isSender()) {

                    beh_class *cls = new beh_class(this);
                    processScoresPtr_front->classifier = cls;
                    processScoresPtr_front->classifier->classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
                    //qDebug()  << classifier->classifier_file;
                    processScoresPtr_front->classifier->allocate_model();
                    processScoresPtr_front->classifier->loadclassifier_model();

                }
            }
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

        if(isSender() && mesPass)
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

                if (!mesPass) {

                    if(isSender())
                    {
                        processScoresPtr_front -> acquireLock();
                        processScoresPtr_front -> isFront = true;
                        processScoresPtr_front -> releaseLock();
                    }

                    if (isReceiver())
                    {
                        processScoresPtr_side->acquireLock();
                        processScoresPtr_side->isSide = true;
                        processScoresPtr_side->releaseLock();
                    }
                       
                }else {

                    if (isReceiver())
                    {
                        processScoresPtr_side->acquireLock();
                        processScoresPtr_side->isSide = true;
                        processScoresPtr_side->releaseLock();
                    
                        processScoresPtr_front->acquireLock();
                        processScoresPtr_front->isFront = true;
                        processScoresPtr_front->releaseLock();
                    }
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
        if (frontRadioButtonPtr_->isChecked() && sideRadioButtonPtr_->isChecked())
        {
            
            if (processScoresPtr_side->HOGHOF_frame->isHOGPathSet
                && processScoresPtr_side->HOGHOF_frame->isHOFPathSet
                && processScoresPtr_front->HOGHOF_partner->isHOGPathSet
                && processScoresPtr_front->HOGHOF_partner->isHOFPathSet
                && processScoresPtr_side->classifier->isClassifierPathSet)
            {

                detectButtonPtr_->setEnabled(true);
                saveButtonPtr_->setEnabled(true);

            }

        }else if (frontRadioButtonPtr_->isChecked()) {

            if (processScoresPtr_front->HOGHOF_partner->isHOGPathSet
                && processScoresPtr_front->HOGHOF_partner->isHOFPathSet)
            {
               
                detectButtonPtr_->setEnabled(true);
                saveButtonPtr_->setEnabled(true);
                
            }

        }else if (sideRadioButtonPtr_->isChecked()) {

            if (processScoresPtr_side->HOGHOF_frame->isHOGPathSet
                && processScoresPtr_side->HOGHOF_frame->isHOFPathSet)
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
        if (frontRadioButtonPtr_->isChecked() || sideRadioButtonPtr_->isChecked())
        {
            if (isReceiver()) {

                if (processScoresPtr_side->classifier == nullptr)
                {

                    setupClassifier();

                } else {

                    processScoresPtr_side->classifier->classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
                    processScoresPtr_side->classifier->allocate_model();
                    processScoresPtr_side->classifier->loadclassifier_model();

                }
            }

            if (isSender()) {

                if (processScoresPtr_front->classifier == nullptr)
                {

                    setupClassifier();

                }
                else {

                    processScoresPtr_front->classifier->classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
                    processScoresPtr_front->classifier->allocate_model();
                    processScoresPtr_front->classifier->loadclassifier_model();

                }
            }
        }
        detectEnabled();
       
    }


    void JaabaPlugin::triggerLaser()
    {

        /*int num_beh = static_cast<int>(classifier->beh_present.size()); 
        for(int nbeh =0;nbeh < num_beh;nbeh++)
        {
            if (classifier->score[nbeh] > 0)  
                laserRead[nbeh] = 1;
            else
                laserRead[nbeh] = 0;
        }*/

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

    void JaabaPlugin::onFrameHOFShape(QPointer<HOGHOF> partner_hofshape)
    {

        partner_hofshape_ = partner_hofshape->hof_shape;
        std::cout << "cameraNumber: " << cameraNumber_ << partner_hofshape_.x << "" <<
            partner_hofshape_.y << " " << partner_hofshape_.bin << std::endl;

        if (!mesPass && isSender())
        {
            if (processScoresPtr_front->isHOGHOFInitialised)
            {

                std::cout << "translated from mat to c" << std::endl;
                processScoresPtr_front->classifier->translate_mat2C(&partner_hofshape->hog_shape,
                    &processScoresPtr_front->HOGHOF_partner->hog_shape);

            }
        }
    }

    void JaabaPlugin::onFrameHOGShape(QPointer<HOGHOF> partner_hogshape)
    {

        partner_hogshape_ = partner_hogshape->hog_shape;
        std::cout << "cameraNumber: " <<  cameraNumber_ << partner_hogshape_.x << "" << 
           partner_hogshape_.y << " " << partner_hogshape_.bin << std::endl;

        if (!mesPass && isReceiver())
        {
            if (processScoresPtr_side->isHOGHOFInitialised) 
            {

                std::cout << "translated from mat to c" << std::endl;
                processScoresPtr_side->classifier->translate_mat2C(&processScoresPtr_side->HOGHOF_frame->hog_shape, &partner_hogshape->hog_shape);    
             
            }
        }
    }

    void JaabaPlugin::scoreCompute(PredData predScore)
    {
        
        if (isReceiver())
        {
            processScoresPtr_side->isProcessed_front = 1;
            processScoresPtr_side->acquireLock();
            processScoresPtr_side->frontScoreQueue.push_back(predScore);
            processScoresPtr_side->releaseLock();
           
        } 

    }

    void JaabaPlugin::receiveFrameRead(int64_t frameReadtime, int frameCount)
    {

        if (isReceiver())
        {
            acquireLock();
            processScoresPtr_side->partner_frame_read_stamps[frameCount] = frameReadtime;
            releaseLock();
        }

        if (isSender())
        {
            acquireLock();
            processScoresPtr_front->partner_frame_read_stamps[frameCount] = frameReadtime;
            releaseLock();
        }

    }

    /*void JaabaPlugin::receiveFrameNum(unsigned int frameReadNum)
    {
        if (isReceiver())
        {
            partner_frameCount_ = frameReadNum;
            processScoresPtr_side->predScore.frameCount = frameReadNum;
            processScoresPtr_side->partner_frameCount_ = frameReadNum;
            //std::cout << frameNum << std::endl;
        }
    }*/

    void JaabaPlugin::scoreCalculated(bool score_cal)
    {
        if (isSender())
            score_calculated_ = score_cal;
    }

    void JaabaPlugin::setSkipFrameProcess(unsigned int frameCount)
    {
        if (isSender())
        {
            acquireLock();
            processScoresPtr_front->skipSide = 1;   
            releaseLock();

        }

        if (isReceiver())
        {
            acquireLock();
            processScoresPtr_side->skipFront = 1;
            releaseLock();

        }
    }

    void JaabaPlugin::setupNIDAQ(std::shared_ptr <Lockable<NIDAQUtils>> nidaq_task,
                                    bool testConfigEnabled, string trial_info,
                                    std::shared_ptr<TestConfig> testConfig) 
    {
        std::cout << "nidaq setup for plugin " << std::endl;
        nidaq_task_ = nidaq_task;
        testConfig_ = testConfig;
        testConfigEnabled_ = testConfigEnabled;
        trial_num_ = trial_info;

        if (testConfigEnabled_)
            allocate_testVec();
          
    }

    void JaabaPlugin::allocate_testVec()
    {

        if (testConfigEnabled_) {

            if (!testConfig_->f2f_prefix.empty()) {

                ts_pc.resize(testConfig_->numFrames, 0.0);
            }

            if (!testConfig_->nidaq_prefix.empty()) {
                
                ts_nidaq.resize(testConfig_->numFrames, std::vector<uInt32>(2, 0));
                ts_nidaqThres.resize(testConfig_->numFrames, 0.0);
                
            }

            if (!testConfig_->queue_prefix.empty()) {

                queue_size.resize(testConfig_->numFrames, 0);
            }

            if (process_frame_time)
            {
                ts_gpuprocess_time.resize(testConfig_->numFrames, 0);
                ts_pc.resize(testConfig_->numFrames, 0.0);
            }
        }

#if DEBUG     
        time_stamps1.resize(testConfig_->numFrames, 0.0);
        time_stamps5.resize(testConfig_->numFrames, 0);
#endif        
    }

    void JaabaPlugin::setImageQueue(std::shared_ptr<LockableQueue<StampedImage>> pluginImageQueuePtr,
                                    std::shared_ptr<LockableQueue<unsigned int>> skippedFramesPluginPtr)
    {
        //std::cout << "inside setImageQueue" << std::endl;
        pluginImageQueuePtr_ = pluginImageQueuePtr;
        skippedFramesPluginPtr_ = skippedFramesPluginPtr;

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
 
}

