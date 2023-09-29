#include "jaaba_plugin.hpp"
#include <iostream>
#include <QDebug>
#include <cuda_runtime.h>
#include <string>
#include <algorithm>
#include <functional>
#include <Windows.h>


//#define DEBUG 1 
//#define compute 1
//#define isVidInput 1
//#define visualize 0
//#define savefeat 0

//
//Camera 1 should always be front view
//Camera 0 should always be side view
//

namespace bias {

    //Public static variables 
    const QString JaabaPlugin::PLUGIN_NAME = QString("jaabaPlugin");
    const QString JaabaPlugin::PLUGIN_DISPLAY_NAME = QString("Jaaba Plugin");

    // Public Methods
    JaabaPlugin::JaabaPlugin(string camera_id, 
                             QPointer<QThreadPool> threadPoolPtr, 
                             std::shared_ptr<Lockable<GetTime>> gettime,
                             CmdLineParams& cmdlineparams,
                             QWidget *parent) : BiasPlugin(parent)
    {

        //nviews_ = numberOfCameras;
        camera_serial_id = stoi(camera_id);
        threadPoolPtr_ = threadPoolPtr;
        gettime_ = gettime;
        nidaq_task_ = nullptr;

        //initialize cmdline arguments
        cmdlineparams_ = cmdlineparams;
        output_feat_directory = cmdlineparams.output_dir;
        isVideo = cmdlineparams.isVideo;
        saveFeat = cmdlineparams.saveFeat;
        compute_jaaba = cmdlineparams.compute_jaaba;
        classify_scores = cmdlineparams.classify_scores;
        visualize = cmdlineparams.visualize;
        numframes_ = cmdlineparams.numframes;
        isDebug = cmdlineparams.debug;

        print(cmdlineparams);

        cudaError_t err = cudaGetDeviceCount(&nDevices_);
        if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
        setupUi(this);
        connectWidgets();
        initialize(cmdlineparams);

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

/*#if isVidInput   
        //DEVEL
        if(processScoresPtr_self->isVid)
        {
            std::cout << "inside side " << std::endl;
            image_height = IMAGE_HEIGHT; //height of frame from video input
            image_width = IMAGE_WIDTH;   //width of frame from video input 
            
        }

        if (processScoresPtr_self->isVid)
        {
            std::cout << "inside front" << std::endl;
            image_height = IMAGE_HEIGHT; //height of frame from video input
            image_width = IMAGE_WIDTH;   //width of frame from video input 
            
        }
#else 
        Format7Settings settings;
        cameraPtr_->acquireLock();
        settings = cameraPtr_->getFormat7Settings();
        image_height = settings.width;
        image_width = settings.height;
        cameraPtr_->releaseLock();

#endif*/

        Format7Settings settings;
        cameraPtr_->acquireLock();
        settings = cameraPtr_->getFormat7Settings();
        image_height = settings.width;
        image_width = settings.height;
        cameraPtr_->releaseLock();

    }


    bool JaabaPlugin::isSender()
    {

        if (view_ == "viewB")
        {
            return true;
        }
        else {
            return false;
        }
    }


    bool JaabaPlugin::isReceiver()
    {

        if (view_ == "viewA")
        {
            return true;
        }
        else {
            return false;
        }
    }


    void JaabaPlugin::stop()
    {
        gpuInitialized = false;
        
        if (isReceiver())
        {
            if (visualize) {
                processScoresPtr_self->visplots->stop();
                delete processScoresPtr_self->visplots;
            }

            delete processScoresPtr_self->classifier;
        }


        if (!HOGHOF_self.isNull())
        {
            delete HOGHOF_self;
        }

        if (!processScoresPtr_self.isNull())
        {
            delete processScoresPtr_self;
        }
        
    }


    void JaabaPlugin::reset()
    {
        
        if (isReceiver())
        {
            if ((threadPoolPtr_ != nullptr) && (processScoresPtr_self != nullptr))
            {
                if (classify_scores) {
                    threadPoolPtr_->start(processScoresPtr_self);
                }

                if (visualize)
                {
                    threadPoolPtr_->start(processScoresPtr_self->visplots);
                }

            }
            
        }

    }

    void JaabaPlugin::stopThread()
    {
        if (isReceiver())
        {
            if ((threadPoolPtr_ != nullptr) && !processScoresPtr_self.isNull())
            {
                if (classify_scores)
                {

                    processScoresPtr_self->acquireLock();
                    processScoresPtr_self->stop();
                    processScoresPtr_self->releaseLock();
                    if (processScoresPtr_self.isNull())
                        std::cout << "ProcessScores is deleted" << std::endl;
                }
            }
        }
    }


    void JaabaPlugin::gpuInit()
    {

        // intitialize the HOGHOF context for the GPU memory
        getFormatSettings();
        std::cout << "DEBUG:: Image height: " << image_height 
                  << "Image Width: " << image_width << std::endl;

        if (!HOGHOF_self->isHOGHOFInitialised)
        {

            if (!(HOGHOF_self.isNull()))
            {
                if (nDevices_ >= 2)
                {
                    cudaSetDevice(cuda_device);

                    HOGHOF_self->initHOGHOF(image_width, image_height);
                    hog_num_elements = HOGHOF_self->hog_shape.x*
                        HOGHOF_self->hog_shape.y*
                        HOGHOF_self->hog_shape.bin;
                    hof_num_elements = HOGHOF_self->hof_shape.x*
                       HOGHOF_self->hof_shape.y*
                        HOGHOF_self->hof_shape.bin;

                }
                else {

                    HOGHOF_self->initHOGHOF(image_width, image_height);
                    hog_num_elements = HOGHOF_self->hog_shape.x*
                        HOGHOF_self->hog_shape.y*
                        HOGHOF_self->hog_shape.bin;
                    hof_num_elements = HOGHOF_self->hof_shape.x*
                        HOGHOF_self->hof_shape.y*
                        HOGHOF_self->hof_shape.bin;

                }
                hoghof_feat.resize(numframes_, std::vector<float>());
                hoghof_feat_avg.resize(numframes_, std::vector<float>());
            }
            else {
                QString errMsgTitle = QString("gpuInit");
                QString errMsgText = QString("HOGHOF_self is NULL in cameraNumber ") + QString::number(cameraNumber_);
                QMessageBox::critical(this, errMsgTitle, errMsgText);
            }
        }

        // initialize side gpu context
        if (isReceiver()){

            //acquireLock();
            detectStarted = true;
            processScoresPtr_self->isSide = true;
            //releaseLock();

            setgpuInitializeFlag();

        }
        
        if (isSender()){

            //acquireLock();
            detectStarted = true;
            processScoresPtr_self->isFront = true;
            //releaseLock();

            setgpuInitializeFlag();

        }

        // histogram parameters initialize - obsolete
        /*if (mesPass) {

            if (HOGHOF_self->isHOGHOFInitialised && HOGHOF_partner->isHOGHOFInitialised)
            {

                processScoresPtr_self->classifier->translate_mat2C(&HOGHOF_self->hog_shape,
                    &HOGHOF_self->hog_shape);
                std::cout << HOGHOF_self->hog_shape.x
                    << HOGHOF_self->hog_shape.y
                    << std::endl;
            
            }
        }*/
    }

    void JaabaPlugin::gpuDeinit()
    {
        
        HOGHOF_self->isHOGHOFInitialised = false;
        gpuInitialized = false;
        if (HOGHOF_self != nullptr)
        {
            if (nDevices_ >= 2)
            {
                /*if (view_ == "viewA"){
                    cudaSetDevice(0);
                    HOFTeardown(HOGHOF_self->hof_ctx);
                    HOGTeardown(HOGHOF_self->hog_ctx);
                    cudaDeviceReset();

                }else if (view_ == "viewB") {
                    cudaSetDevice(1);
                    HOFTeardown(HOGHOF_self->hof_ctx);
                    HOGTeardown(HOGHOF_self->hog_ctx);
                    cudaDeviceReset();
                }*/

                cudaSetDevice(cuda_device);
                HOFTeardown(HOGHOF_self->hof_ctx);
                HOGTeardown(HOGHOF_self->hog_ctx);
                cudaDeviceReset();
                std::cout << "gpu ctx deInit in " << view_ << std::endl;
                
            }

        }

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
            //connect(partnerPluginPtr, SIGNAL(passHOGShape(QPointer<HOGHOF>)),
            //    this, SLOT(receiveHOGShape(QPointer<HOGHOF>)));
            //connect(partnerPluginPtr, SIGNAL(processSide(bool)),
            //    this, SLOT(initialize_classifier()));
            connect(this, SIGNAL(passHOGShape(QPointer<HOGHOF>)),
                    partnerPluginPtr, SLOT(receiveHOGShape(QPointer<HOGHOF>)));
            connect(this, SIGNAL(processSide(bool)),
                    partnerPluginPtr, SLOT(setgpuInitializeFlag()));

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


    /*void JaabaPlugin::resetTrigger()
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

    }*/

    void JaabaPlugin::processFrames(StampedImage stampedImage)
    {
        
        if(mesPass)
        {
            //processFramePass();

        } else {

            processFrame_inPlugin(stampedImage);

        }

    }

    void JaabaPlugin::processFrame_inPlugin(StampedImage stampedImage)
    {

        cv::Mat pluginImage;
        uInt32 read_buffer = 0, read_ondemand = 0;
        cv::Mat greyImage;
        float scaling_factor = 1.0 / 255.0;
        bool isskip;
        QString errorMsg(" ");


        //DEVEL
        int64_t pc_time;
        uint64_t start_process = 0, end_process = 0;
        uint64_t front_read_time, side_read_time, time_now;
        uint64_t start_delay, end_delay;
        string filename;
        uint64_t expTime = 0, curTime = 0;
        uint64_t curTime_vid=0, expTime_vid=0;
        uint64_t frameGrabAvgTime, max_jaaba_compute_time, avg_frameLatSinceFirstFrame=0;
        int64_t wait_thres, avgwait_time;
        double vis_ts = 0.0;
        string prefix;
      
        frameCount_ = 0; // Incoming frame framecount

        if (view_ == "viewA")
        {
            prefix = "side";
        }
        else if(view_ == "viewB") 
        {
            prefix = "front";
        }

        if (nDevices_ >= 2)
            cudaSetDevice(cuda_device);

        if (isVideo) {
            frameGrabAvgTime = 2500;
            max_jaaba_compute_time = 2000;
            wait_thres = 2000;
            avgwait_time = 0;
        }
        else {

            frameGrabAvgTime = 2500;
            wait_thres = static_cast<int64_t>(1500);
            max_jaaba_compute_time = 2000;
            avgwait_time = 0;
        }


        if (pluginImageQueuePtr_ != nullptr)
        {
            acquireLock();
            pluginImage = stampedImage.image;
            frameCount_ = stampedImage.frameCount;
            fstfrmtStampRef_ = stampedImage.fstfrmtStampRef;
            timeStamp_ = stampedImage.timeStamp;
            releaseLock();

            // ensuring hoghof averaging vectors are empty before starting processing
            if (frameCount_ == 0) {
                std::cout << " FrameCount on Start " << processedFrameCount << std::endl;

                waitForEmptyHOGHOFAvgQueue(HOGHOF_self->hog_out_past);
                waitForEmptyHOGHOFAvgQueue(HOGHOF_self->hof_out_past);
            }

            /*if (!isVideo) {

                if (isDebug) {
                    if (testConfigEnabled_ && nidaq_task_ != nullptr) {

                        if (frameCount_ < testConfig_->numFrames) {

                            nidaq_task_->getNidaqTimeNow(read_ondemand);
                        }
                    }
                }
            }*/

            start_process = gettime_->getPCtime();

            // skip frame if process time on gpu is higher than thres
            /*if (frameCount_ == 0)
                start_prev = start_process;

            if (cameraNumber_ == 0) {
                if (jaabaSkipFrame(start_process, start_prev,
                    processScoresPtr_self->processedFrameCount, max_jaaba_compute_time))
                    return;
            }
            else if (cameraNumber_ == 1){
                if (jaabaSkipFrame(start_process, start_prev,
                    processScoresPtr_self->processedFrameCount, max_jaaba_compute_time))
                    return;
            }*/

            if (fstfrmtStampRef_ != 0)
            {
                // estimate curr time and exp time 
                /*if (isVideo) {
                    time_now = gettime_->getPCtime();
                    expTime_vid = fstfrmtStampRef_ + (frameGrabAvgTime * (frameCount_ + 1)) + max_jaaba_compute_time;
                    curTime_vid = static_cast<uint64_t>(time_now);
                    avgwait_time = curTime_vid - expTime_vid;

                    // this is for visualizer code to run accurately
                    if (isReceiver() && processScoresPtr_self != nullptr &&
                        processScoresPtr_self->processedFrameCount == 0)
                        processScoresPtr_self->fstfrmStampRef = fstfrmtStampRef_;
                }
                else {
                   
                    avg_frameLatSinceFirstFrame = (((frameGrabAvgTime) * (frameCount_ + 1))
                                                    + max_jaaba_compute_time ); // nidaq ts frame of reference

                    expTime = (static_cast<uint64_t>(fstfrmtStampRef_) * 20) + avg_frameLatSinceFirstFrame;

                    //expTime = nidaq_task_->cam_trigger[frameCount_]*0.02;
                    curTime = (static_cast<uint64_t>(read_ondemand) * 20);

                    avgwait_time = curTime - expTime;
                }*/


                //match to see if incoming frame frameCount matches the currently being processed 
                //frameCount, otherwise consider it skipped frame
                while (processedFrameCount < frameCount_)
                {
                    //std::cout << view_ << " skipped in Jaaba plugin " <<
                    //    processedFrameCount << std::endl;
                    if (isDebug && testConfigEnabled_) {
                        end_process = gettime_->getPCtime();
                        ts_nidaqThres[processedFrameCount] = 1;
                        ts_gpuprocess_time[processedFrameCount] = (end_process - start_process);
                        ts_jaaba_start[processedFrameCount] = start_process;
                        ts_jaaba_end[processedFrameCount] = end_process;
                        //time_cur[processScoresPtr_self->processedFrameCount] = curTime;
                    }

                    if (saveFeat) {

                        saveFeatures(output_feat_directory + "hoghof_" + prefix + "_biasjaaba.csv",
                            HOGHOF_self->hog_out_skip,
                            HOGHOF_self->hof_out_skip,
                            hog_num_elements, hof_num_elements);
                    }

                    HOGHOF_self->averageWindowFeatures(window_size, processedFrameCount, 1);

                    if (saveFeat) {
                        saveFeatures(output_feat_directory + "hoghof_avg_" + prefix + "_biasjaaba.csv",
                            HOGHOF_self->hog_out_avg,
                            HOGHOF_self->hof_out_avg,
                            hog_num_elements, hof_num_elements);
                    }

                    processedFrameCount++;
                }
                
                // if frameCount from image metadata does not match current processing framecount throw error
                if(processedFrameCount != frameCount_) 
                {
                    errorMsg = QString::fromStdString("Camera framecount does not match Jaaba framecount ");
                    errorMsg += QString::number(frameCount_);
                    emit framecountMatchError(0, errorMsg);
                }


                if (pluginImage.rows != 0 && pluginImage.cols != 0)
                {

                    if (pluginImage.channels() == 3)
                    {
                        cv::cvtColor(pluginImage, pluginImage, cv::COLOR_BGR2GRAY);
                    }

                    // convert the frame into float32
                    pluginImage.convertTo(greyImage, CV_32FC1, scaling_factor);

                    if (processScoresPtr_self != nullptr)
                    {

                        if (frameCount_ == 0)
                            std::cout << view_ << " image received" << std::endl;

                        if (compute_jaaba) {                 

                            //std::cout << view_ << " image received in " << cameraNumber_ << std::endl;
                            
                            HOGHOF_self->img.buf = greyImage.ptr<float>(0);
                            HOGHOF_self->genFeatures(frameCount_);

                            if (saveFeat) {
                                    
                                saveFeatures(output_feat_directory + "hoghof_" + prefix + "_biasjaaba.csv", 
                                    HOGHOF_self->hog_out,
                                    HOGHOF_self->hof_out,
                                    hog_num_elements, hof_num_elements);
                            }

                            //average window features
                            HOGHOF_self->averageWindowFeatures(window_size, processedFrameCount,0);
                                
                            if(saveFeat){

                                saveFeatures(output_feat_directory + "hoghof_avg_" + prefix + "_biasjaaba.csv",
                                    HOGHOF_self->hog_out_avg,
                                    HOGHOF_self->hof_out_avg,
                                    hog_num_elements, hof_num_elements);
                      
                            }
                  
                        }

                        processScoresPtr_self->classifier->boost_classify(processScoresPtr_self->classifier->predScore.score,
                            HOGHOF_self->hog_out_avg, HOGHOF_self->hof_out_avg,
                            &HOGHOF_self->hog_shape, &HOGHOF_self->hof_shape,
                            processScoresPtr_self->classifier->model,
                            processedFrameCount, view_);

                        processScoresPtr_self->classifier->predScore.frameCount = processedFrameCount;
                        
                        if (nidaq_task_ != nullptr) {

                            /*if (frameCount_ < numframes_) {

                                nidaq_task_->getNidaqTimeNow(read_ondemand);
                            }*/

                        }

                        if(!isVideo) 
                        {
                            if (nidaq_task_ != nullptr) {

                                if (frameCount_ < numframes_) {

                                    nidaq_task_->getNidaqTimeNow(read_ondemand);
                                }
                                time_now = static_cast<uint64_t>(read_ondemand);
                            }
                            else {

                                time_now = gettime_->getPCtime();
                            }
                        }
                        else if(isVideo)
                        {
                            time_now = gettime_->getPCtime();
                        }

                        if (processScoresPtr_self->classifier->isClassifierPathSet && isReceiver())
                        {

                            processScoresPtr_self->classifier->predScore.view = 1;
                            processScoresPtr_self->classifier->predScore.fstfrmtStampRef_ = fstfrmtStampRef_;
                            processScoresPtr_self->classifier->predScore.score_viewA_ts = time_now;

                            selfScoreQueuePtr_->acquireLock();
                            selfScoreQueuePtr_->push(processScoresPtr_self->classifier->predScore);
                            selfScoreQueuePtr_->releaseLock();
                        }

                        if (processScoresPtr_self->classifier->isClassifierPathSet && isSender())
                        {
                            processScoresPtr_self->classifier->predScore.view = 2;
                            processScoresPtr_self->classifier->predScore.fstfrmtStampRef_ = fstfrmtStampRef_;
                            processScoresPtr_self->classifier->predScore.score_viewB_ts = time_now;

                            partnerScoreQueuePtr_->acquireLock();
                            partnerScoreQueuePtr_->push(processScoresPtr_self->classifier->predScore);
                            partnerScoreQueuePtr_->releaseLock();
                        }

                        // obsolete code
                        /*if (processScoresPtr_self->classifier->isClassifierPathSet && isSender())
                        {

                            processScoresPtr_self->classifier->boost_classify_front(processScoresPtr_self->classifier->predScoreFront.score,
                                HOGHOF_self->hog_out_avg, HOGHOF_self->hof_out_avg,
                                &HOGHOF_self->hog_shape, &HOGHOF_self->hof_shape,
                                processScoresPtr_self->classifier->model,
                                processedFrameCount);
                            //std::cout << "Scr in Jaaba " << processScoresPtr_self->classifier->predScoreFront.score[0] << std::endl;

                            time_now = gettime_->getPCtime();
                            processScoresPtr_self->classifier->predScoreFront.frameCount = processedFrameCount;
                            processScoresPtr_self->classifier->predScoreFront.score_viewB_ts = time_now;
                            processScoresPtr_self->classifier->predScoreFront.view = 2;

                            //std::cout << "Pushing to Front queue" << std::endl;
                            frontScoreQueuePtr_->acquireLock();
                            frontScoreQueuePtr_->push(processScoresPtr_self->classifier->predScoreFront);
                            frontScoreQueuePtr_->releaseLock();

                        }


                        if (processScoresPtr_self->classifier->isClassifierPathSet && isReceiver())
                        {

                            processScoresPtr_self->classifier->boost_classify_side(processScoresPtr_self->classifier->predScoreSide.score,
                                HOGHOF_self->hog_out_avg, HOGHOF_self->hof_out_avg,
                                &HOGHOF_self->hog_shape, &HOGHOF_self->hof_shape,
                                processScoresPtr_self->classifier->model,
                                processedFrameCount);
                            std::cout << "Scr in Jaaba " << processScoresPtr_self->classifier->predScoreSide.score[0] << std::endl;

                            time_now = gettime_->getPCtime();
                            processScoresPtr_self->classifier->predScoreSide.frameCount = processedFrameCount;
                            processScoresPtr_self->classifier->predScoreSide.score_viewA_ts = time_now;
                            processScoresPtr_self->classifier->predScoreSide.view = 1;
                            processScoresPtr_self->isProcessed_side = 1;

                            //processScoresPtr_self->write_score(output_feat_directory + "classifier_side.csv", processScoresPtr_self->classifier->predScoreSide);
                            sideScoreQueuePtr_->acquireLock();
                            sideScoreQueuePtr_->push(processScoresPtr_self->classifier->predScoreSide);
                            sideScoreQueuePtr_->releaseLock();

                        }*/

                        processedFrameCount++;
                    }
                }
            }  
            end_process = gettime_->getPCtime();
        }

        if(isDebug){
            if (testConfigEnabled_ && frameCount_ < testConfig_->numFrames)
            {

                if (!testConfig_->nidaq_prefix.empty()) {

                    ts_nidaq[frameCount_][0] = nidaq_task_->cam_trigger[frameCount_];
                    ts_nidaq[frameCount_][1] = read_ondemand;
                    imageTimeStamp[frameCount_] = timeStamp_;
                    
                }

                if (!testConfig_->f2f_prefix.empty()) {

                    pc_time = gettime_->getPCtime();
                    ts_pc[frameCount_] = pc_time;
                }

                if (!testConfig_->queue_prefix.empty()) {

                    //queue_size[frameCount_] = 2;// pluginImageQueuePtr_->size();
                    if (cameraNumber_ == 0)
                        queue_size[frameCount_] = side_skip_count;
                    else if (cameraNumber_ == 1)
                        queue_size[frameCount_] = front_skip_count;

                }

                if (process_frame_time) {

                    ts_gpuprocess_time[frameCount_] = (end_process - start_process);
                    ts_jaaba_start[frameCount_] = start_process;
                    ts_jaaba_end[frameCount_] = end_process;
                    //ts_nidaqThres[frameCount_] = expTime;
                    time_cur[frameCount_] = curTime;

                }

                if (frameCount_ == (testConfig_->numFrames - 1)
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

                if (frameCount_ == (testConfig_->numFrames - 1)
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

                    std::string filename2 = testConfig_->dir_list[0] + "/"
                        + testConfig_->nidaq_prefix + "/" + testConfig_->cam_dir
                        + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                        + testConfig_->plugin_prefix
                        + "_" + "imagetimestamp_" + "cam"
                        + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";

                    gettime_->write_time_1d<int64_t>(filename1, testConfig_->numFrames, ts_nidaqThres);
                    gettime_->write_time_2d<uInt32>(filename, testConfig_->numFrames, ts_nidaq);
                    gettime_->write_time_1d<double>(filename2, testConfig_->numFrames, imageTimeStamp);
                }

                if (frameCount_ == (testConfig_->numFrames - 1)
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

                if (frameCount_ == (testConfig_->numFrames - 1)
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

                    string filename2 = testConfig_->dir_list[0] + "/"
                        + testConfig_->nidaq_prefix + "/" + testConfig_->cam_dir
                        + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                        + testConfig_->plugin_prefix
                        + "_" + "end_time_" + "cam"
                        + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";

                    string filename3 = testConfig_->dir_list[0] + "/"
                        + testConfig_->nidaq_prefix + "/" + testConfig_->cam_dir
                        + "/" + testConfig_->git_commit + "_" + testConfig_->date + "/"
                        + testConfig_->plugin_prefix
                        + "_" + "cur_time" + "cam"
                        + std::to_string(cameraNumber_) + "_" + trial_num_ + ".csv";

                    gettime_->write_time_1d<int64_t>(filename, testConfig_->numFrames, ts_gpuprocess_time);
                    gettime_->write_time_1d<int64_t>(filename1, testConfig_->numFrames, ts_jaaba_start);
                    gettime_->write_time_1d<int64_t>(filename2, testConfig_->numFrames, ts_jaaba_end);
                    //gettime_->write_time_1d<int64_t>(filename3, testConfig_->numFrames, time_cur);

                }
            }
        }else{}

    }

    //void JaabaPlugin::processFrames(QList<StampedImage> frameList)
    /*void JaabaPlugin::processFramePass()
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
        if (isReceiver() && ((!processScoresPtr_self->isHOGHOFInitialised) || (!processScoresPtr_partner->isHOGHOFInitialised)))
        {
            gpuInit();
        }

        // Send frames from front plugin to side
        if(pluginImageQueuePtr_ != nullptr && isSender())
        {    
            emit(partnerImageQueue(pluginImageQueuePtr_)); //causes the preview images to be slow
            processScoresPtr_self -> processedFrameCount += 1;
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

                sideImage = stampedImage0.image;
                frontImage = stampedImage1.image;

                if((sideImage.rows != 0) && (sideImage.cols != 0) 
                    && (frontImage.rows != 0) && (frontImage.cols != 0))
                {

                    acquireLock();
                    currentImage_ = sideImage;
                    frameCount_ = stampedImage0.frameCount;
                    releaseLock();

                    // Test
                //    if(sideImage.ptr<float>(0) != nullptr && frameCount == 1000)
                //    {
                //
                //        imwrite("out_feat/side_" + std::to_string(frameCount) + ".jpg", sideImage);
                //        imwrite("out_feat/front_" + std::to_string(frameCount) + ".jpg", frontImage);
                //        //sideImage.convertTo(sideImage, CV_32FC1);
                //        //frontImage.convertTo(frontImage,CV_32FC1);
                //        //sideImage = sideImage / 255;
                //        //std::cout << sideImage.rows << " " << sideImage.cols << std::endl;
                //        //write_output("out_feat/side" + std::to_string(frameCount_) + ".csv" , sideImage.ptr<float>(0), sideImage.rows, sideImage.cols);
                //        //write_output("out_feat/front" + std::to_string(frameCount_) + ".csv" , frontImage.ptr<float>(0), frontImage.rows , frontImage.cols);
                //    }
                    

                    if( stampedImage0.frameCount == stampedImage1.frameCount)
                    {
                        
                        if(frameCount_  == (processScoresPtr_self -> processedFrameCount+1))
                        {
                            // Test - Uncomment to perform preprocesing of video frames
                            //if (processScoresPtr_self->capture_sde.isOpened())
                            //{
                            //    if (processScoresPtr_self->processedFrameCount < nframes_)
                            //    {

                            //        sideImage = processScoresPtr_self->vid_sde->getImage(processScoresPtr_self->capture_sde);
                            //        processScoresPtr_self->vid_sde->convertImagetoFloat(sideImage);
                            //        greySide = sideImage;

                            //    } else {

                            //        return;
                            //    }

                            //}

                            //if (processScoresPtr_self->capture_front.isOpened())
                            //{
                            //    if (processScoresPtr_self->processedFrameCount < nframes_)
                            //    {

                            //        frontImage = processScoresPtr_self->vid_front->getImage(processScoresPtr_self->capture_front);
                            //        processScoresPtr_self->vid_front->convertImagetoFloat(frontImage);
                            //        greyFront = frontImage;

                            //    } else {

                            //        return;
                            //    }

                            //}
                            //else {

                            //    processScoresPtr_self->vid_front->releaseCapObject(processScoresPtr_self->capture_front);
                            //    
                            //}

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
                                    
                                if(processScoresPtr_self -> isSide)
                                {

                                    cudaSetDevice(0);
                                    processScoresPtr_self -> HOGHOF_self->img.buf = greySide.ptr<float>(0);
                                    processScoresPtr_self -> onProcessSide();
                                  
                                }

                                if(processScoresPtr_self -> isFront)
                                {

                                    cudaSetDevice(1);
                                    processScoresPtr_self -> HOGHOF_self->img.buf = greyFront.ptr<float>(0);
                                    processScoresPtr_self -> genFeatures(processScoresPtr_self -> HOGHOF_self, frameCount_);
                                    //processScoresPtr_self->onProcessFront();
                                    
                                }

                                while (!processScoresPtr_self->isProcessed_side) {}

                            } else {

                                processScoresPtr_self -> HOGHOF_self -> img.buf = greySide.ptr<float>(0);
                                processScoresPtr_self -> genFeatures(processScoresPtr_self -> HOGHOF_self, frameCount_);
                                processScoresPtr_self -> HOGHOF_self -> img.buf = greyFront.ptr<float>(0);
                                processScoresPtr_self -> genFeatures(processScoresPtr_self -> HOGHOF_self, frameCount_);

                            }
                            

                            // Test
                            //if(processScoresPtr_->save && frameCount_ == 2000)
                            //{

                            //    QPointer<HOGHOF> HOGHOF_side = processScoresPtr_->HOGHOF_self;
                            //    processScoresPtr_ -> write_histoutput("./out_feat/hog_side_" + std::to_string(frameCount_) 
                            //    + ".csv", HOGHOF_side->hog_out.data(), HOGHOF_side->hog_shape.x, HOGHOF_side->hog_shape.y, HOGHOF_side->hog_shape.bin);
                            //    processScoresPtr_ -> write_histoutput("./out_feat/hof_side_" + std::to_string(frameCount_) 
                            //    + ".csv", HOGHOF_side->hof_out.data(), HOGHOF_side->hof_shape.x, HOGHOF_side->hof_shape.y, HOGHOF_side->hof_shape.bin);


                            //    QPointer<HOGHOF> HOGHOF_front = processScoresPtr_->HOGHOF_self;
                            //    processScoresPtr_ -> write_histoutput("./out_feat/hog_front_" + std::to_string(frameCount_) 
                            //    + ".csv", HOGHOF_front->hog_out.data(), HOGHOF_front->hog_shape.x, HOGHOF_front->hog_shape.y, 
                            //    HOGHOF_front->hog_shape.bin);
                            //        
                            //    processScoresPtr_ -> write_histoutput("./out_feat/hof_front_" + std::to_string(frameCount_) 
                            //    + ".csv", HOGHOF_front->hof_out.data(), HOGHOF_front->hof_shape.x, HOGHOF_front->hof_shape.y, 
                            //    HOGHOF_front->hof_shape.bin);
                            //}


                            // compute scores
                            if(processScoresPtr_self->classifier->isClassifierPathSet & processScoresPtr_self->processedFrameCount >= 0)
                            {

                                //std::fill(laserRead.begin(), laserRead.end(), 0);
                                processScoresPtr_self->classifier->boost_classify_side(processScoresPtr_self->classifier->predScoreSide.score, 
                                 processScoresPtr_self->HOGHOF_self->hog_out, processScoresPtr_self -> HOGHOF_self->hof_out, 
                                 &processScoresPtr_self->HOGHOF_self->hog_shape, &processScoresPtr_partner ->HOGHOF_self->hof_shape, 
                                    processScoresPtr_self->classifier-> nframes, processScoresPtr_self->classifier->model, processScoresPtr_self->processedFrameCount);

                                processScoresPtr_self->classifier->boost_classify_front(processScoresPtr_self->classifier->predScoreFront.score, 
                                    processScoresPtr_partner->HOGHOF_self->hog_out, processScoresPtr_partner->HOGHOF_self->hof_out, 
                                    &processScoresPtr_self->HOGHOF_self->hog_shape, &processScoresPtr_partner->HOGHOF_self->hof_shape, 
                                    processScoresPtr_self->classifier->nframes, processScoresPtr_self->classifier->model,processScoresPtr_partner->processedFrameCount);
                                
                                processScoresPtr_self->classifier->addScores(processScoresPtr_self->classifier->predScoreSide.score,
                                                                             processScoresPtr_self->classifier->predScoreFront.score);
                                processScoresPtr_self->write_score("classifierscr.csv",
                                    processScoresPtr_self->classifier->finalscore);

                                //triggerLaser();
                                //visplots -> livePlotTimeVec_.append(stampedImage0.timeStamp);
                                //visplots -> livePlotSignalVec_Lift.append(double(classifier->score[0]));
                                //visplots -> livePlotSignalVec_Handopen.append(double(classifier->score[1]));
                                //visplots -> livePlotSignalVec_Grab.append(double(classifier->score[2]));
                                //visplots -> livePlotSignalVec_Supinate.append(double(classifier->score[3]));
                                //visplots -> livePlotSignalVec_Chew.append(double(classifier->score[4]));
                                //visplots -> livePlotSignalVec_Atmouth.append(double(classifier->score[5]));
                                //visplots->livePlotPtr_->show();
                                

                            }
                           
                            //std::cout << frameCount_ << " "  << processScoresPtr_self->processedFrameCount << std::endl;
                            processScoresPtr_self -> processedFrameCount = frameCount_;
                            processScoresPtr_partner -> processedFrameCount = frameCount_; 
                            processScoresPtr_self->isProcessed_side = false;
                            //processScoresPtr_partner->isProcessed_front = false;

                        }else{ std::cout << "skipped 1 " << frameCount_ << std::endl; }

                    }else{ std::cout << "skipped 2 " << frameCount_ << std::endl; }
                     
                }else { std::cout << "skipped 3" << frameCount_ << std::endl;}
                    
                pluginImageQueuePtr_->pop();
                partnerPluginImageQueuePtr_ -> pop();
             
            }else { std::cout << "skipped 4 " << frameCount_ << std::endl; }

            pluginImageQueuePtr_->releaseLock();
            partnerPluginImageQueuePtr_ -> releaseLock();         
            end_process = gettime_->getPCtime();
//#if DEBUG            
            if (testConfigEnabled_ && !testConfig_->imagegrab_prefix.empty()
                && testConfig_->plugin_prefix == "jaaba_plugin" && isDebug)
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
                    gettime_->write_time_1d<int64_t>(filename, testConfig_->numFrames, ts_nidaqThres);

                }
            }
//#endif
        }        
    }*/


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
    
        
    void JaabaPlugin::initialize(CmdLineParams& cmdlineparams)
    {

        QPointer<CameraWindow> cameraWindowPtr = getCameraWindow();
        cameraNumber_ = cameraWindowPtr->getCameraNumber();
        partnerCameraNumber_ = getPartnerCameraNumber();
        cameraWindowPtrList_ = cameraWindowPtr->getCameraWindowPtrList();
        cameraPtr_ = cameraWindowPtr->getCameraPtr();

        frameCount_ = 0;
        laserOn = false;
        mesPass = false;
        score_calculated_ = 0;
        process_frame_time = 1;
        gpuInitialized = false;
        
        processScoresPtr_self = new ProcessScores(this, mesPass, gettime_ , cmdlineparams);
        //processScoresPtr_partner = new ProcessScores(this, mesPass, gettime_, cmdlineparams);

        // need to manually call delete to delete the above objects
        processScoresPtr_self->setAutoDelete(false);
        //processScoresPtr_partner->setAutoDelete(false);

        if (visualize)
        {
            if (processScoresPtr_self != nullptr && cameraNumber_ == 0)
                processScoresPtr_self->visplots = new VisPlots(livePlotPtr, this);
        }

        //updateWidgetsOnLoad();
        //setupHOGHOF();
        //setupClassifier();

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

        /*connect(
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
            tabWidgetPtr,
            SIGNAL(currentChanged(currentIndex())),
            this,
            SLOT(setCurrentIndex(currentIndex()))
        );*/

        /*connect(
            reloadPushButtonPtr_,
            SIGNAL(clicked()),
            this,
            SLOT(reloadButtonPressed())
        );

        connect(
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
    
        if (isVideo) {
            processScoresPtr_self->isVid = 1;
        }
      
        HOGHOF *hoghofside = new HOGHOF();//(this);

        if (processScoresPtr_self != nullptr)
        {

            /*acquireLock();
            processScoresPtr_self->HOGHOF_self = hoghofside;
            processScoresPtr_self->HOGHOF_self->HOGParam_file = config_file_dir + hog_file;
            processScoresPtr_self->HOGHOF_self->HOFParam_file = config_file_dir + hof_file;
            processScoresPtr_self->HOGHOF_self->CropParam_file = config_file_dir + crop_file;
            processScoresPtr_self->HOGHOF_self->initialize_HOGHOFParams();
            releaseLock();*/

            HOGHOF_self = hoghofside;
            HOGHOF_self->HOGParam_file = config_file_dir + hog_file;
            HOGHOF_self->HOFParam_file = config_file_dir + hof_file;
            HOGHOF_self->CropParam_file = config_file_dir + crop_file;
            HOGHOF_self->isHOGHOFInitialised = false;
            HOGHOF_self->initialize_HOGHOFParams();
            printf("processScores allocated in %s\n" , view_);
                         
        }else {

            printf("processScores not allocated in %s\n", view_);
        }

    }


    void JaabaPlugin::setupClassifier() 
    {
       
        /*if (mesPass && isReceiver())
        {
            beh_class *cls = new beh_class(this);
            processScoresPtr_self->classifier = cls;
            processScoresPtr_self->classifier->classifier_file = config_file_dir + classifier_filename;
            //processScoresPtr_self->classifier->num_behs = num_behs;
            //processScoresPtr_self->classifier->beh_names = beh_names;
            processScoresPtr_self->classifier->allocate_model();
            processScoresPtr_self->classifier->loadclassifier_model();
                          
        }*/

        float classifier_thres = jab_conf.classsifer_thres;
        bool output_trigger = jab_conf.output_trigger;
        int baudRate = jab_conf.baudRate;
        int perFrameLat = jab_conf.perFrameLat;

        std::cout <<  "Classifier Threshold " << classifier_thres  
                  <<  "output Trigger " << output_trigger  
                  <<  "BaudRate " << baudRate
                  << "Perframe latency " << perFrameLat 
                  << std::endl;
        
        //extract behavior names
        string behavior_names = jab_conf.beh_names;
        stringstream behnamestream(behavior_names);
        string cur_beh;
        while (!behnamestream.eof())
        {
            getline(behnamestream, cur_beh, ',');
            beh_names.push_back(cur_beh);
        }
        num_behs = jab_conf.num_behs;

        std::cout << "num behs " << num_behs << std::endl;

        if (!mesPass)
        {
            beh_class *cls = new beh_class(this);
            processScoresPtr_self->classifier = cls;
            processScoresPtr_self->classifier->classifier_file = config_file_dir + classifier_filename;
            processScoresPtr_self->classifier->num_behs = num_behs;
            processScoresPtr_self->classifier->beh_names = beh_names;
            processScoresPtr_self->classifier->allocate_model();
            processScoresPtr_self->classifier->loadclassifier_model();
            processScoresPtr_self->classifierThres = classifier_thres;
            processScoresPtr_self->outputTrigger = output_trigger;
            processScoresPtr_self->baudRate = baudRate;
            processScoresPtr_self->perFrameLat = perFrameLat;
    
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

   
    /*void JaabaPlugin::updateWidgetsOnLoad() 
    {

        if(isSender() && mesPass)
        {
 
            this -> setEnabled(false);   

        } else {

            sideRadioButtonPtr_ -> setChecked(false);
            frontRadioButtonPtr_ -> setChecked(false);
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
        //setupHOGHOF();
        //setupClassifier();
        //detectEnabled();
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
        //setupHOGHOF();
        //setupClassifier();
        //detectEnabled();

    }


    void JaabaPlugin::reloadButtonPressed()
    {

        pathtodir_->setPlaceholderText(pathtodir_->displayText());
        // load side HOGHOFParams if side view checked
        if(sideRadioButtonPtr_->isChecked())
        {
            if(processScoresPtr_self -> HOGHOF_self == nullptr) 
            {
                setupHOGHOF();

            } else {
            
                readPluginConfig(jab_conf);
                pathtodir_->placeholderText() = processScoresPtr_self->HOGHOF_self->plugin_file;
                processScoresPtr_self -> HOGHOF_self->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
                processScoresPtr_self -> HOGHOF_self->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
                processScoresPtr_self -> HOGHOF_self->CropParam_file = pathtodir_->placeholderText() + CropSideParamFilePtr_->placeholderText();
                processScoresPtr_self -> HOGHOF_self->loadHOGParams();
                processScoresPtr_self -> HOGHOF_self->loadHOFParams();
                processScoresPtr_self -> HOGHOF_self->loadCropParams();            
            }
        }

        // load front HOGHOFParams if front view checked
        if(frontRadioButtonPtr_->isChecked())
        {
            if(processScoresPtr_self -> HOGHOF_self == nullptr)
            {

                setupHOGHOF();        
  
            } else {

                readPluginConfig(jab_conf);
                pathtodir_->placeholderText() = processScoresPtr_self->HOGHOF_self->plugin_file;
                processScoresPtr_self -> HOGHOF_self->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
                processScoresPtr_self -> HOGHOF_self->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
                processScoresPtr_self -> HOGHOF_self->CropParam_file = pathtodir_->placeholderText() + CropFrontParamFilePtr_->placeholderText();
                processScoresPtr_self -> HOGHOF_self->loadHOGParams();
                processScoresPtr_self -> HOGHOF_self->loadHOFParams();
                processScoresPtr_self -> HOGHOF_self->loadCropParams();
            }
        }

        //load classifier
        if (frontRadioButtonPtr_->isChecked() || sideRadioButtonPtr_->isChecked())
        {
            if (isReceiver()) {

                if (processScoresPtr_self->classifier == nullptr)
                {

                    setupClassifier();

                } else {

                    processScoresPtr_self->classifier->classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
                    processScoresPtr_self->classifier->allocate_model();
                    processScoresPtr_self->classifier->loadclassifier_model();

                }
            }

            if (isSender()) {

                if (processScoresPtr_self->classifier == nullptr)
                {

                    setupClassifier();

                }
                else {

                    processScoresPtr_self->classifier->classifier_file = pathtodir_->placeholderText() + ClassFilePtr_->placeholderText();
                    processScoresPtr_self->classifier->allocate_model();
                    processScoresPtr_self->classifier->loadclassifier_model();

                }
            }
        }
        //detectEnabled(); 
    }


    void JaabaPlugin::triggerLaser()
    {

        int num_beh = static_cast<int>(processScoresPtr_self->classifier->beh_present.size());
        for(int nbeh =0;nbeh < num_beh;nbeh++)
        {
            if (processScoresPtr_self->classifier->beh_present[nbeh] > 0)  
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
        }
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

    }*/

    
    void JaabaPlugin::onPartnerPlugin(std::shared_ptr<LockableQueue<StampedImage>> partnerPluginImageQueuePtr)
    {

        if(partnerPluginImageQueuePtr != nullptr)
        { 
            partnerPluginImageQueuePtr_ = partnerPluginImageQueuePtr;
        }
    }


    void JaabaPlugin::receiveHOGShape(QPointer<HOGHOF> partner_hoghof)
    {
        
        if (!mesPass)
        {
            if (isReceiver()) 
            {

                std::cout << "translated from mat to c Side" << std::endl;
                partner_hogshape_ = partner_hoghof->hog_shape;
                
                //partner_hogshape_.x = 20; partner_hogshape_.y = 10; partner_hogshape_.bin = 8;
                //hogshape_.x = 30; hogshape_.y = 10; hogshape_.bin = 8;

                //processScoresPtr_self->classifier->translate_mat2C(&processScoresPtr_self->HOGHOF_self->hog_shape, 
                //    &partner_hogshape_);
				//processScoresPtr_self->classifier->translate_featureIndexes(&processScoresPtr_self->HOGHOF_self->hog_shape,
				//	&partner_hogshape_, true);
                processScoresPtr_self->classifier->getviewandfeature(&HOGHOF_self->hog_shape,
                        &partner_hogshape_, view_);
            }

            if (isSender())
            {
                std::cout << "translated from mat to c front" << std::endl;
                partner_hogshape_ = partner_hoghof->hog_shape;
                //processScoresPtr_self->classifier->translate_mat2C(&partner_hogshape_,
                //    &processScoresPtr_self->HOGHOF_self->hog_shape);
				//processScoresPtr_self->classifier->translate_featureIndexes(&partner_hogshape_,
				//	&processScoresPtr_self->HOGHOF_self->hog_shape, false);
                processScoresPtr_self->classifier->getviewandfeature(&partner_hogshape_,
                        &HOGHOF_self->hog_shape, view_);
            }
        }
    }

    void JaabaPlugin::setgpuInitializeFlag()
    {

        gpuInitialized = true;
    }


    void JaabaPlugin::scoreCompute(PredData predScore)
    {
        
        if (isReceiver())
        {
            processScoresPtr_self->isProcessed_front = 1;
            processScoresPtr_self->acquireLock();
            processScoresPtr_self->partnerScoreQueuePtr_->push(predScore);
            processScoresPtr_self->releaseLock();
           
        } 

    }


    void JaabaPlugin::receiveFrameRead(int64_t frameReadtime, int frameCount)
    {

        if (isReceiver())
        {
            acquireLock();
            processScoresPtr_self->partner_frame_read_stamps[frameCount] = frameReadtime;
            releaseLock();
        }

        if (isSender())
        {
            acquireLock();
            processScoresPtr_self->partner_frame_read_stamps[frameCount] = frameReadtime;
            releaseLock();
        }

    }


    /*void JaabaPlugin::receiveFrameNum(unsigned int frameReadNum)
    {
        if (isReceiver())
        {
            partner_frameCount_ = frameReadNum;
            processScoresPtr_self->predScore.frameCount = frameReadNum;
            processScoresPtr_self->partner_frameCount_ = frameReadNum;
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
            processScoresPtr_self->skipSide = 1;   
            releaseLock();
        }

        if (isReceiver())
        {
            acquireLock();
            processScoresPtr_self->skipFront = 1;
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

        if(nidaq_task != nullptr)
            ts_nidaq.resize(numframes_, std::vector<uInt32>(2, 0));

        // test vectors allocated for scores
        if (processScoresPtr_self != nullptr && isReceiver())
        {
            std::cout << "Scores && " << numframes_ << std::endl;
            processScoresPtr_self->scores.resize(numframes_);
            processScoresPtr_self->numFrames = numframes_;
            processScoresPtr_self->nidaq_task_= nidaq_task_;
			//translated_indexes.resize(numframes_, std::vector<float>());
        }
        else if (isSender()) {}
        else{

            QString errMsgTitle = QString("ProcessScores Initialization");
            QString errMsgText = QString("ProcessScores params not initialized");
            QMessageBox::critical(this, errMsgTitle, errMsgText);
        }

        if (isDebug)
            allocate_testVec();
          
    }


    void JaabaPlugin::allocate_testVec()
    {

        if (testConfigEnabled_) {
            std::cout << "allocated testVec " << cameraNumber_ <<  std::endl;

            if (!testConfig_->f2f_prefix.empty()) {

                ts_pc.resize(testConfig_->numFrames, 0);
            }

            if (!testConfig_->nidaq_prefix.empty()) {
                
                //ts_nidaq.resize(testConfig_->numFrames, std::vector<uInt32>(2, 0));
                ts_nidaqThres.resize(testConfig_->numFrames, 0);
                //scores.resize(testConfig_->numFrames);
                imageTimeStamp.resize(testConfig_->numFrames, 0.0);
            }

            if (!testConfig_->queue_prefix.empty()) {

                queue_size.resize(testConfig_->numFrames, 0);
            }

            if (process_frame_time)
            {
                ts_gpuprocess_time.resize(testConfig_->numFrames, 0);
                ts_jaaba_start.resize(testConfig_->numFrames, 0);
                ts_jaaba_end.resize(testConfig_->numFrames, 0);
                time_cur.resize(testConfig_->numFrames, 0);
           
            }
            
        }
       
    }


    void JaabaPlugin::refill_testVec()
    {
        std::cout << "Refill testVec " << cameraNumber_ << std::endl;
        if (testConfigEnabled_) {

            if (!testConfig_->f2f_prefix.empty()) {

                std::fill(ts_pc.begin(), ts_pc.end(), 0);
            }

            if (!testConfig_->nidaq_prefix.empty()) {

                //std::fill(ts_nidaq.begin(), ts_nidaq.end(), vector<uInt32>(2, 0)); // this segfaults ?? 
                if (!ts_nidaq.empty()) {
                    for (unsigned int i = 0; i < ts_nidaq.size(); i++)
                    {
                        ts_nidaq[i][0] = 0;
                        ts_nidaq[i][1] = 0;
                    }
                }
                else {
                    std::cout << "ts nidaq empty " <<  ts_nidaq.size() << std::endl;
                }
                std::fill(ts_nidaqThres.begin(), ts_nidaqThres.end(), 0);
                std::fill(imageTimeStamp.begin(),imageTimeStamp.end(),0.0);
            }

            if (!testConfig_->queue_prefix.empty()) {

                std::fill(queue_size.begin(),queue_size.end(),0);
            }

            if (process_frame_time)
            {
                std::fill(ts_gpuprocess_time.begin(),ts_gpuprocess_time.end(), 0);
                std::fill(ts_jaaba_start.begin(), ts_jaaba_start.end(), 0);
                std::fill(ts_jaaba_end.begin(), ts_jaaba_end.end(), 0);
                std::fill(time_cur.begin(), time_cur.end(), 0);

            }

        }
    }


    void JaabaPlugin::setImageQueue(std::shared_ptr<LockableQueue<StampedImage>> pluginImageQueuePtr,
                                    std::shared_ptr<LockableQueue<unsigned int>> skippedFramesPluginPtr)
    {
        
        pluginImageQueuePtr_ = pluginImageQueuePtr;
        skippedFramesPluginPtr_ = skippedFramesPluginPtr;
        
    }


    void JaabaPlugin::setScoreQueue(std::shared_ptr<LockableQueue<PredData>> selfScoreQueuePtr,
                                    std::shared_ptr<LockableQueue<PredData>> partnerScoreQueuePtr)
    {

        if (selfScoreQueuePtr != nullptr) 
        {
            selfScoreQueuePtr_ = selfScoreQueuePtr;
        }
        else 
        {
            QString errMsgTitle = QString("setScore Queue");
            QString errMsgText = QString("score queue self is null ");
            QMessageBox::critical(this, errMsgTitle, errMsgText);
        }

        if (partnerScoreQueuePtr != nullptr)
        {
            partnerScoreQueuePtr_ = partnerScoreQueuePtr;
        }
        else 
        {
            QString errMsgTitle = QString("setScore Queue");
            QString errMsgText = QString("score queue partner is null ");
            QMessageBox::critical(this, errMsgTitle, errMsgText);
        }
        std::cout << "Score Queue set in JAABA for cameranumber: \n" << cameraNumber_ <<  std::endl;
        
        if (processScoresPtr_self != nullptr)
        {

            // setQueue for processScores thread 
            processScoresPtr_self->setScoreQueue(selfScoreQueuePtr_, partnerScoreQueuePtr_);
            
            if(isReceiver())
                processScoresPtr_self->initSerialOutputPort();
        }
    }


    void JaabaPlugin::initializeParamsProcessScores()
    {

        std::queue<vector<float>> empty_que;
        processedFrameCount = 0;

        if (isDebug) {
            refill_testVec();
        }

        if (processScoresPtr_self != nullptr)
        {
            processScoresPtr_self->initialize(mesPass, gettime_, cmdlineparams_);
            /*processScoresPtr_self->HOGHOF_self->resetHOGHOFVec();
            processScoresPtr_self->HOGHOF_self->hog_out_past.clear();
            processScoresPtr_self->HOGHOF_self->hof_out_past.clear();*/

            HOGHOF_self->resetHOGHOFVec();
            HOGHOF_self->hog_out_past.clear();
            HOGHOF_self->hof_out_past.clear();
            std::cout << "reset hoghof vectors" << cameraNumber_ << std::endl;
        }

    }


    void JaabaPlugin::loadConfig()
    {
        std::cout << "Jaaba load Config reached\n" << std::endl;
        unordered_map<string,unsigned int>::iterator camera_it;
        unordered_map<unsigned int, string>::iterator crop_file_it;

        //jab_conf.readPluginConfig(conf_filename.toStdString());      
        config_file_dir = jab_conf.config_file_dir;
        hog_file = jab_conf.hog_file;
        hof_file = jab_conf.hof_file;
        classifier_filename = jab_conf.classifier_filename;
        camera_list = jab_conf.camera_serial_id;
        jab_crop_list = jab_conf.crop_file_list;
        window_size = jab_conf.window_size;
        cuda_device = jab_conf.cuda_device;

        std::cout << "cuda device set in " << view_ << " " << cuda_device << std::endl;

        camera_it = camera_list.begin();

        // map camera guid names to camera view
        while (camera_it != camera_list.end())
        {
            
            if (camera_serial_id == camera_it->second && camera_it->first == "viewA") {
                sideRadioButtonPtr_->setChecked(true);
                view_ = "viewA";
            }
            else if (camera_serial_id == camera_it->second && camera_it->first == "viewB") {
                frontRadioButtonPtr_->setChecked(true);
                view_ = "viewB";
            }else{}

           camera_it++;
        }

        // map camera guid names to crop file name
        crop_file_it = jab_crop_list.begin();
        while (crop_file_it != jab_crop_list.end())
        {
            
            if (crop_file_it->first == camera_serial_id)
                crop_file = crop_file_it->second;
            crop_file_it++;
        }

        //extract behavior names
        /*string behavior_names = jab_conf.beh_names;
        stringstream behnamestream(behavior_names);
        string cur_beh;
        while(!behnamestream.eof())
        {
            getline(behnamestream , cur_beh , ',');
            beh_names.push_back(cur_beh);
        }*/
          
        setupHOGHOF();
        setupClassifier();
        if (cameraNumber_ == 0)
            std::cout << "all side setup done\n" << std::endl;
        else if (cameraNumber_ == 1)
            std::cout << "all front setup done\n" << std::endl;
    }


    QVariantMap JaabaPlugin::getConfigAsMap()
    {
        QVariantMap configMap = jab_conf.toMap();
        return configMap;
    }


    RtnStatus JaabaPlugin::setConfigFromMap(QVariantMap configMap)
    {

        RtnStatus rtnStatus = jab_conf.fromMap(configMap);
        if (rtnStatus.success) {
            std::cout << "Load Jaaba config Plugin" << std::endl;
            loadConfig();
        }
        return rtnStatus;
    }


    void JaabaPlugin::average_windowFeatures(vector<float>& hog_feat, vector<float>& hof_feat, 
                                             vector<float>& hog_feat_avg, vector<float>& hof_feat_avg, int window_size)
    {
        int feat_size = hog_feat.size();
        transform(hog_feat.begin(), hog_feat.end(), hog_feat.begin(), [window_size](float &c) {return (c / window_size); });
        transform(hof_feat.begin(), hof_feat.end(), hof_feat.begin(), [window_size](float &c) {return (c / window_size); });
        
        for (int i = 0; i < feat_size; i++) {

            hog_feat_avg[i] = hog_feat_avg[i] + hog_feat[i];
            hof_feat_avg[i] = hof_feat_avg[i] + hof_feat[i];
        }

    }


    void JaabaPlugin::subtractLastwindowfeature(vector<float>& hog_past, vector<float>& hof_past,
                                                vector<float>& hog_feat_avg, vector<float>& hof_feat_avg)
    {
        int feat_size = hog_past.size();
        for (int i = 0; i < feat_size; i++) {

            hog_feat_avg[i] = hog_feat_avg[i] - hog_past[i];
            hof_feat_avg[i] = hof_feat_avg[i] - hof_past[i];
         }

    }


    void JaabaPlugin::setTrialNum(string trialnum)
    {
        trial_num_ = trialnum;
        if (processScoresPtr_self != nullptr && isReceiver())
        {
            processScoresPtr_self->setTrialNum(trialnum);
        }
    }


    bool JaabaPlugin::jaabaSkipFrame(uint64_t& ts_cur, uint64_t& ts_prev,
        int& frameCount, uint64_t wait_thres)
    {
        if (frameCount == 1) {
            std::cout << "prev ts before " << ts_prev <<
                "cur ts before " << ts_cur << std::endl;
            std::cout << "start diff " << (ts_cur - ts_prev) << std::endl;
        }
        if ((ts_cur - ts_prev) > wait_thres)
        {
            std::cout << "Frame skipped " << frameCount << std::endl;
            frameCount++;       
            ts_prev = ts_cur;
            return 1;
        }
        ts_prev = ts_cur;
        if (frameCount == 0)
            std::cout << "ts prev after " << ts_prev
            << "ts cur after " << ts_cur << std::endl;
        return 0;
    }


    void JaabaPlugin::waitForEmptyHOGHOFAvgQueue(LockableQueue<vector<float>>& avg_que)
    {
        avg_que.acquireLock();
        avg_que.waitIfNotEmpty();
        if (!avg_que.empty())
        {
            std::cout << "Averaging hog queue not empty" << std::endl;
            avg_que.releaseLock();
            return;
        }
        avg_que.releaseLock();

        if (!avg_que.empty())
            std::cout << " HOGHOF avg Que not empty ***" << std::endl;

    }


    void JaabaPlugin::setHOGHOFShape()
    {
        if (!HOGHOF_self.isNull()) {

            if (isReceiver()){
                if (gpuInitialized) {

                    emit(passHOGShape(HOGHOF_self));
                }
                else {
                    QString errMsgTitle = QString("setHOGHOFShape");
                    QString errMsgText = QString("GPU not initialized in cameraNumber ") + QString::number(cameraNumber_);
                    QMessageBox::critical(this, errMsgTitle, errMsgText);
                }   
            }
                
            if (isSender()) {
                if (gpuInitialized) {
                    emit(passHOGShape(HOGHOF_self));
                }
                else {
                    QString errMsgTitle = QString("setHOGHOFShape");
                    QString errMsgText = QString("GPU not initialized in cameraNumber ") + QString::number(cameraNumber_);
                    QMessageBox::critical(this, errMsgTitle, errMsgText);
                }
            }
            
        }
        else {
            std::cout << " HOGHOF is NULL in setHOGHOFShape" << std::endl;
        }
    }


    /*void JaabaPlugin::saveAvgwindowfeatures(vector<vector<float>>& hoghof_feat, QPointer<HOGHOF> hoghof_obj,
                                            int frameCount, string filename) {

        std::ofstream x_out;
        x_out.open(filename.c_str(), std::ios_base::app);


        for (int j = 0; j < hog_num_elements; j++)
        {
            x_out << setprecision(6) << hoghof_obj->hog_out_avg[j] << ",";
        }

        for (int k = 0; k < hof_num_elements; k++)
        {
            if (k == (hof_num_elements - 1))
                x_out << setprecision(6) << hoghof_obj->hof_out_avg[k] << "\n";
            else
                x_out << setprecision(6) << hoghof_obj->hof_out_avg[k] << ",";
        }


        x_out.close();

    }*/


	/*void JaabaPlugin::paintEvent(QPainter& painter)
	{
		setAttribute(Qt::WA_OpaquePaintEvent);
		
		QPen linepen(Qt::red);
		linepen.setCapStyle(Qt::RoundCap);
		linepen.setWidth(30);
		painter.setRenderHint(QPainter::Antialiasing, true);
		painter.setPen(linepen);
		painter.drawPoint(50,50);
	}
       
    void JaabaPlugin::saveFeatures(string filename, QPointer<HOGHOF> hoghof_obj,//vector<float>& feat_out,
        int hog_num_elements, int hof_num_elements) 
    {
        std::ofstream x_out;
        x_out.open(filename.c_str(), std::ios_base::app);


        for (int j = 0; j < hog_num_elements; j++)
        {
            x_out << setprecision(6) << hoghof_obj->hog_out[j] << ",";
        }

        for (int k = 0; k < hof_num_elements; k++)
        {
            if (k == (hof_num_elements - 1))
                x_out << setprecision(6) << hoghof_obj->hof_out[k] << "\n";
            else
                x_out << setprecision(6) << hoghof_obj->hof_out[k] << ",";
        }

        x_out.close();

    }*/
       

/*********************************************************************************************************/

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

    
    void JaabaPlugin::write_score_final(std::string file, unsigned int numFrames,
        vector<PredData>& pred_score)
    {
        std::ofstream x_out;
        x_out.open(file.c_str(), std::ios_base::app);
        std::cout << "once" << std::endl;
        x_out << "Score ts," << "Score ts side," << "Score ts front," << "Score," << " FrameNumber," << "View" << "\n";

        for (unsigned int frm_id = 0; frm_id < numFrames; frm_id++)
        {
            x_out << pred_score[frm_id].score_ts << "," << pred_score[frm_id].score_viewA_ts
                << "," << pred_score[frm_id].score_viewB_ts << "," << pred_score[frm_id].score[0]
                << "," << pred_score[frm_id].frameCount << "," << pred_score[frm_id].view <<
                "\n";
        }
        x_out.close();
    }
 
}

