#include "process_scores.hpp"
#include <cuda_runtime_api.h>


//#define isVidInput 1
//#define visualize 0

//string output_score_dir = "Y:/hantman_data/jab_experiments/STA14/STA14/20230503/STA14_20230503_142341/";

namespace bias {

    // public 
    ProcessScores::ProcessScores(QObject *parent, bool mesPass,
                                 std::shared_ptr<Lockable<GetTime>> getTime,
                                 CmdLineParams& cmdlineparams) : QObject(parent)
    {

        initialize(mesPass, getTime, cmdlineparams);
        
        // this is defined separately here because it is initialised when constructor
        // is called but some variables need to be reinitialized when multiple trials of the plugin
        //are run. this is the variable that does not need to be reinitialized for every trial
        
        //isHOGHOFInitialised = false;
        stopped_ = true;
        isSide = false;
        isFront = false;
        

    }

    void ProcessScores::initialize(bool mesPass, std::shared_ptr<Lockable<GetTime>> getTime,
        CmdLineParams& cmdlineparams)
    {

        //stopped_ = true;
        //detectStarted_ = false;
        save = false;
        //processedFrameCount = 0;
        processSide = false;
        processFront = false;
        isProcessed_side = false;
        isProcessed_front = false;
        mesPass_ = mesPass;
        frameCount_ = 0;
        partner_frameCount_ = -1;
        scoreCount = 0;
        gettime = getTime;
        skipFront = 0;
        skipSide = 0;
        side_read_time_ = 0;
        front_read_time_ = 0;
        fstfrmStampRef = 0;
        //isHOGHOFInitialised = false;

        output_score_dir = cmdlineparams.output_dir;
        isVideo = cmdlineparams.isVideo;
        visualize = cmdlineparams.visualize;
        wait_threshold = cmdlineparams.wait_thres;
        portName = cmdlineparams.comport;
        
        frame_triggered = 0;

        clearQueues();

    }
   
    /*void ProcessScores::initHOGHOF(QPointer<HOGHOF> hoghof, int img_height, int img_width)
    {

        hoghof->initHOGHOF(img_height,img_width);
        std::cout << "In initHOGHOF, isSide = " << isSide << ", isFront = " << isFront << std::endl;
        isHOGHOFInitialised = true;

    }

    void ProcessScores::genFeatures(QPointer<HOGHOF> hoghof,int frame)
    {

        hoghof->genFeatures(frame);

    }*/


    void ProcessScores::onProcessSide()
    {

        processSide = true;      
 
    }


    void ProcessScores::onProcessFront()
    {

        processFront = true;

    }


    void ProcessScores::stop()
    {

        stopped_ = true;

    }


    /*void ProcessScores::detectOn()
    {

        detectStarted_ = true;

    }


    void ProcessScores::detectOff()
    {

        detectStarted_ = false;

    }*/


    void ProcessScores::setScoreQueue(std::shared_ptr<LockableQueue<PredData>> selfScoreQueuePtr,
        std::shared_ptr<LockableQueue<PredData>> partnerScoreQueuePtr)
    {
        selfScoreQueuePtr_ = selfScoreQueuePtr;
        partnerScoreQueuePtr_ = partnerScoreQueuePtr;
        std::cout << "Score Queue set in ProcessScore " <<  std::endl;
    }


    void ProcessScores::visualizeScores(vector<float>& scr_vec)
    {
        uint64_t time_now = 0;
        double vis_ts = 0.0;
        time_now = gettime->getPCtime();
        vis_ts = (time_now - fstfrmStampRef)*(1.0e-6);

        visplots->livePlotTimeVec_.append(vis_ts);
        visplots->livePlotSignalVec_Lift.append(double(scr_vec[0]));
        visplots->livePlotSignalVec_Handopen.append(double(scr_vec[1]));
        visplots->livePlotSignalVec_Grab.append(double(scr_vec[2]));
        visplots->livePlotSignalVec_Supinate.append(double(scr_vec[3]));
        visplots->livePlotSignalVec_Chew.append(double(scr_vec[4]));
        visplots->livePlotSignalVec_Atmouth.append(double(scr_vec[5]));
        visplots->livePlotPtr_->show();
    }


    void ProcessScores::setTrialNum(string trialnum)
    {
        trial_num_ = trialnum;
        testConfigEnabled_ = 1;

        scores_filename = output_score_dir + "classifier_trial" + trial_num_.back() + ".csv";

    }

    void ProcessScores::initSerialOutputPort()
    {
        // initialize the serial port output

        if (outputTrigger) {
            std::cout << "In ProcessScores constructor, calling initPort.\n";
            if (!portOutput.initPort(portName).success) {
                std::cout << "Error initializing serial port\n";
            }

            // set baudrate
            portOutput.setBaudRate(baudRate);
        }
    }


    /// Private methods
    void ProcessScores::triggerOnClassifierOutput(PredData& classifierPredScore, int frameCount)
    {

        int numBehs = classifier->num_behs;

        //reset the output every frame
        std::fill(classifier->behavior_output_signal.begin(),
            classifier->behavior_output_signal.end(), '0');

        for (auto classifierNum = 0; classifierNum < numBehs; classifierNum++)
        {
            if (classifierPredScore.score[classifierNum] > classifierThres) {

                classifier->behavior_output_signal[classifierNum] = '1';        
            }
        }

        // binary string output signal
        string output_binary_signal(classifier->behavior_output_signal.begin(), 
                                    classifier->behavior_output_signal.end());

        
        //convert binary string to int
        int output_int_val = stoi(output_binary_signal, 0, 2);
        
        if (output_int_val > 0) {

            //convert output from int to char to send over the serial port
            char output_char_signal = (char)(output_int_val + '0');
            portOutput.trigger(output_char_signal);
            frame_triggered += 1;

            //std::cout << "binary signal " <<   output_binary_signal 
            //    << "char signal " << output_char_signal 
            //    << "int signal " << output_int_val << "FrameCount " << frameCount << std::endl;
        }

    }
        

    void ProcessScores::run()
    {

        const bool DEBUGTRIGGER = false;
        const int debugTriggerSkip = 100;

        bool done = false;
        uint64_t time_now;
        double score_ts;
        uint64_t ts_last_score = INT_MAX, cur_time = 0;
        
        if (testConfigEnabled_)
            scores_filename = output_score_dir + "classifier_trial" + trial_num_.back() + ".csv";
        else
            scores_filename = output_score_dir + "classifier_score.csv";

        // Set thread priority to idle - only run when no other thread are running
        QThread *thisThread = QThread::currentThread();
        thisThread -> setPriority(QThread::NormalPriority);
          

        acquireLock();
        stopped_ = false;
        releaseLock();

        while (!done)
        {
            acquireLock();
            done = stopped_;
            releaseLock();

            if (mesPass_) 
            {

                /*if (processSide)
                {

                    cudaSetDevice(0);
                    genFeatures(HOGHOF_self, processedFrameCount + 1);
                    acquireLock();
                    processSide = false;
                    isProcessed_side = true;
                    releaseLock();

                }

                if (processFront) {

                    cudaSetDevice(1);
                    genFeatures(HOGHOF_self, processedFrameCount + 1);
                    acquireLock();
                    processFront = false;
                    isProcessed_front = true;
                    releaseLock();
                }*/

            }
            else {
                skipFront = false;
                skipSide = false;

                if (selfScoreQueuePtr_->empty() && partnerScoreQueuePtr_->empty()) {

                    continue;

                } 
                else if (!selfScoreQueuePtr_->empty() && !partnerScoreQueuePtr_->empty()) {


                    partnerScoreQueuePtr_->acquireLock();
                    predScorePartner = partnerScoreQueuePtr_->front();
                    partnerScoreQueuePtr_->releaseLock();

                    selfScoreQueuePtr_->acquireLock();
                    predScore = selfScoreQueuePtr_->front();
                    selfScoreQueuePtr_->releaseLock();

                    //std::cout << "PredScore FrameCount " << predScore.frameCount
                    //    << "PredScore Partner FrameCount " << predScorePartner.frameCount
                    //    << std::endl;

                    // to keep up with frame where both views are skipped 
                    if (scoreCount < predScore.frameCount)
                    {
                        //if (predScore.frameCount > predScorePartner.frameCount)
                        //    partnerScoreQueuePtr_->pop();
                        scoreCount++;
                        continue;
                    }

                    // to keep up with frames where both views are skipped
                    if (scoreCount < predScorePartner.frameCount)
                    {
                        //if (predScorePartner.frameCount > predScore.frameCount)
                        //    selfScoreQueuePtr_->pop();
                        scoreCount++;
                        continue;
                    }
                    
                    if (scoreCount > predScore.frameCount) {
                        selfScoreQueuePtr_->pop();
                        continue;
                    }

                    if (scoreCount > predScorePartner.frameCount){
                        partnerScoreQueuePtr_->pop();
                        continue;
                    }
                    
                    if (predScore.frameCount == predScorePartner.frameCount)
                    {

                        classifier->addScores(predScore.score, predScorePartner.score);

                        skipSide = false;
                        skipFront = false;

                        selfScoreQueuePtr_->acquireLock();
                        selfScoreQueuePtr_->pop();
                        selfScoreQueuePtr_->releaseLock();

                        partnerScoreQueuePtr_->acquireLock();
                        partnerScoreQueuePtr_->pop();
                        partnerScoreQueuePtr_->releaseLock();
         

                        //KB add output code here
                        if (outputTrigger) {
                            if (DEBUGTRIGGER) {
                                std::cout << scoreCount << std::endl;
                                if ((scoreCount%debugTriggerSkip) == 0) {
                                    portOutput.trigger('1');
                                }
                            }
                            else {

                                triggerOnClassifierOutput(classifier->finalscore, scoreCount);
                            }
                        }

                        if (isVideo) {
                            time_now = gettime->getPCtime();
                            scores[scoreCount].score_ts = time_now;

                        }
                        else {
                            nidaq_task_->getNidaqTimeNow(read_ondemand_);
                            scores[scoreCount].score_ts = read_ondemand_;

                        }

                        scores[scoreCount].score = classifier->finalscore.score;
                        scores[scoreCount].frameCount = predScore.frameCount;
                        scores[scoreCount].view = 3;
                        
                        scores[scoreCount].score_viewA_ts = predScore.score_viewA_ts;
                        scores[scoreCount].score_viewB_ts = predScorePartner.score_viewB_ts;
                                              // - max(predScore.score_ts, predScorePartner.score_ts);

                        if(visualize)
                        {
                            visualizeScores(classifier->finalscore.score);
                        }

                        scoreCount++;
                                             
                    }
                    
                }else if (!partnerScoreQueuePtr_->empty()) {

                    partnerScoreQueuePtr_->acquireLock();
                    predScorePartner = partnerScoreQueuePtr_->front();
                    partnerScoreQueuePtr_->releaseLock();

                    // check if this is not already a processed scoreCount
                    //if (scoreCount > predScorePartner.frameCount) {
                    //    partnerScoreQueuePtr_->pop();
                    //    continue;
                    //}

                    if (predScorePartner.frameCount > scoreCount)
                        scoreCount++;

                    time_now = gettime->getPCtime();
                    score_ts = predScorePartner.score_viewB_ts;
                    
                    if ((time_now - score_ts) > wait_threshold)
                    {
                        skipSide = true;
                    }

                }
                else if (!selfScoreQueuePtr_->empty()) {

                    selfScoreQueuePtr_->acquireLock();
                    predScore = selfScoreQueuePtr_->front();
                    selfScoreQueuePtr_->releaseLock();

                    // check if this is not already a processed scoreCount
                    //if (scoreCount > predScore.frameCount) {
                    //    selfScoreQueuePtr_->pop();
                    //    continue;
                    //}

                    if (predScore.frameCount > scoreCount)
                        scoreCount++;

                    time_now = gettime->getPCtime();
                    score_ts = predScore.score_viewA_ts;
                    
                    if ((time_now - score_ts) > wait_threshold)
                    {                       
                        skipFront = true;                    
                    }

                }

                if (skipFront)
                {
                    if (!skipSide)
                    {
                        skipFront = false;
                        if (scoreCount == predScore.frameCount)
                        {

                            selfScoreQueuePtr_->acquireLock();
                            selfScoreQueuePtr_->pop();
                            selfScoreQueuePtr_->releaseLock();

                            if (outputTrigger) {
                                if (DEBUGTRIGGER) {
                                    std::cout << scoreCount << std::endl;
                                    if ((scoreCount%debugTriggerSkip) == 0) {
                                        portOutput.trigger('1');
                                    }
                                }
                                else {
                                    triggerOnClassifierOutput(predScore, scoreCount);
                                }
                            }

                            if (isVideo) {
                                time_now = gettime->getPCtime();
                                scores[scoreCount].score_ts = time_now;
                            }
                            else {
                                nidaq_task_->getNidaqTimeNow(read_ondemand_);
                                scores[scoreCount].score_ts = read_ondemand_;
                            }

                            scores[scoreCount].score = predScore.score;
                            scores[scoreCount].frameCount = predScore.frameCount;
                            scores[scoreCount].view = 1;
                            scores[scoreCount].score_viewA_ts = predScore.score_viewA_ts;

                            if (visualize) 
                            {
                                visualizeScores(predScore.score);
                            }
                            
                            scoreCount++;
                        }
                        
                    }

                }else if (skipSide) {

                    if (!skipFront)
                    {
                        skipSide = false;
                        if (scoreCount == predScorePartner.frameCount)
                        {

                            partnerScoreQueuePtr_->acquireLock();
                            partnerScoreQueuePtr_->pop();
                            partnerScoreQueuePtr_->releaseLock();  

                            if (outputTrigger) {
                                if (DEBUGTRIGGER) {
                                    std::cout << scoreCount << std::endl;
                                    if ((scoreCount%debugTriggerSkip) == 0) {
                                        portOutput.trigger('1');
                                    }
                                }
                                else {
                                    triggerOnClassifierOutput(predScorePartner, scoreCount);
                                }
                            }

                            if (isVideo) {
                                time_now = gettime->getPCtime();
                                scores[scoreCount].score_ts = time_now;
                            }
                            else {

                                nidaq_task_->getNidaqTimeNow(read_ondemand_);
                                scores[scoreCount].score_ts = read_ondemand_;
                            }

                            time_now = gettime->getPCtime();
                            scores[scoreCount].score = predScorePartner.score;
                            scores[scoreCount].frameCount = predScorePartner.frameCount;
                            scores[scoreCount].view = 2;
                            scores[scoreCount].score_viewB_ts = predScorePartner.score_viewB_ts;

                            if (visualize)
                            {
                                visualizeScores(predScorePartner.score);
                            }
       
                            scoreCount++;

                        }       
                    }
                }
            }

            //std::cout << scoreCount << std::endl;
            acquireLock();
            done = stopped_;
            releaseLock();

            if (scoreCount == (numFrames)) {
                std::cout << "Score file name " << scores_filename << std::endl;
                std::cout << "Writing score...." << std::endl;
                write_score_final(scores_filename,numFrames, scores);
                std::cout << "frames triggered ...." << frame_triggered << std::endl;
              
                /*acquireLock();
                done = true;
                releaseLock();*/

                clearQueues();

                scoreCount = 0;
            }

        }
     
        std::cout << "ProcessScores run method exited " << std::endl;
        if (outputTrigger) {
            portOutput.disconnectTriggerDev();
        }

    }

    void ProcessScores::clearQueues()
    {
        
        if (selfScoreQueuePtr_ != nullptr) {
            if (!selfScoreQueuePtr_->empty()) {
                selfScoreQueuePtr_->acquireLock();
                selfScoreQueuePtr_->clear();
                selfScoreQueuePtr_->releaseLock();
                std::cout << "Side Queue Cleared " << std::endl;
            }
            else {
                std::cout << "Side queue is empty" << std::endl;
            }
        }
        else {
            std::cout << "Side Score NULL" << std::endl;
        }

        if (partnerScoreQueuePtr_ != nullptr) {
            if (!partnerScoreQueuePtr_->empty()) {
                partnerScoreQueuePtr_->acquireLock();
                partnerScoreQueuePtr_->clear();
                partnerScoreQueuePtr_->releaseLock();
                std::cout << "Front Queue Cleared " << std::endl;
            }
            else {
                std::cout << "Front queue is empty" << std::endl;
            }

        }
        else {
        std:cout << "Front Score NULL" << std::endl;
        }
    }


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //Test

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void ProcessScores::write_histoutput(std::string file,float* out_img, unsigned w, unsigned h,unsigned nbins)
    {

        std::ofstream x_out;
        x_out.open(file.c_str());

        // write hist output to csv file
        for(unsigned int k = 0;k < nbins; k++){
            for(unsigned int i = 0;i < h; i++){
                for(unsigned int j = 0; j < w; j++){
                    x_out << out_img[k*w*h +i*w + j]; // i*h +j
                    if(j != w-1 || i != h-1 || k != nbins -1)
                        x_out << ",";
                }
            }
        }
    }


    void ProcessScores::write_score(std::string file,PredData& score)
    {

        std::ofstream x_out;
        x_out.open(file.c_str(), std::ios_base::app);

        //x_out << "Score ts," << "Score," << " FrameNumber," << "View" << "\n";
        
        // write score to csv file
        x_out << score.score_ts << "," << score.score[1] << "," 
            << score.frameCount << "," << score.view << "\n";

        x_out.close();

    }


    void ProcessScores::write_score_final(std::string file, unsigned int numFrames,
                                          vector<PredData>& pred_score)
    {
        std::ofstream x_out;
        x_out.open(file.c_str(), std::ios_base::app);
        std::cout << "once" << std::endl;
        x_out << "Score ts," << "Score ts side," << "Score ts front," << "Score," << " FrameNumber," << "View" << "\n";

        for (unsigned int frm_id = 0; frm_id < numFrames; frm_id++)
        {
            x_out << pred_score[frm_id].score_ts << "," << pred_score[frm_id].score_viewA_ts 
                << "," << pred_score[frm_id].score_viewB_ts << "," 
                << setprecision(6) << pred_score[frm_id].score[0]
				<< "," << setprecision(6) << pred_score[frm_id].score[1] 
                << "," << setprecision(6) << pred_score[frm_id].score[2]
				<< "," << setprecision(6) << pred_score[frm_id].score[3] 
                << "," << setprecision(6) << pred_score[frm_id].score[4]
				<< "," << setprecision(6) << pred_score[frm_id].score[5]
                << "," << pred_score[frm_id].frameCount << "," << pred_score[frm_id].view <<
                "\n";
        }
        x_out.close();
    }


    void ProcessScores::write_frameNum(std::string filename, vector<int>& frame_vec, int numSkips) {

        std::ofstream x_out;
        x_out.open(filename.c_str());

        for (int i = 0; i < numSkips; i++)
        {
            x_out << frame_vec[i] << "\n";
        }
        x_out.close();

    }

 ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // copied from GrabDetectorPlugin::refreshPortList
    // todo: refactor code to share
    void SerialPortOutput::refreshPortList()
    {
        serialInfoList_.clear();
        // Get list of serial ports and populate comports 
        QList<QSerialPortInfo> serialInfoListTmp = QSerialPortInfo::availablePorts();

        for (QSerialPortInfo serialInfo : serialInfoListTmp)
        {
            if (serialInfo.portName().contains("ttyS"))
            {
                continue;
            }
            else
            {
                serialInfoList_.append(serialInfo);
            }
        }
    }


    RtnStatus SerialPortOutput::initPort(string portName) {

        RtnStatus rtnStatus;

        bool portFound = false;
        QSerialPortInfo portInfo;
        refreshPortList();
        portName_ = QString::fromStdString(portName);
        std::cout << "Portname: " <<  portName  << std::endl;

        for (QSerialPortInfo serialInfo : serialInfoList_)
        {
            //std::cout << serialInfo.portName().toStdString() << std::endl;
            if (portName_ == serialInfo.portName())
            {
                portFound = true;
                portInfo = serialInfo;
                break;
            }
        }

        if (portFound)
        {
            rtnStatus = connectTriggerDev(portInfo);

            if (!rtnStatus.success)
            {
                rtnStatus.success = false;
                rtnStatus.appendMessage(QString("unable to connect to port %1").arg(portName_));
            }
        }
        else
        {
            rtnStatus.success = false;
            rtnStatus.appendMessage(QString("port %1 not found").arg(portName_));
        }

        return rtnStatus;
    }


    RtnStatus SerialPortOutput::connectTriggerDev(QSerialPortInfo portInfo)
    {
        RtnStatus rtnStatus;
        std::cout << "Connecting to output trigger device...." << std::endl;

        if (pulseDevice_.isOpen())
        {
            rtnStatus.success = true;
            rtnStatus.message = QString("device already connected");
            std::cout << "Device already connected." << std::endl;
            return rtnStatus;
        }

        std::cout << "Connecting ...\n";
        if (serialInfoList_.size() > 0)
        {
            pulseDevice_.setPort(portInfo);
            pulseDevice_.open();
        }

        // Check to see if device is opene or closed and set status string accordingly
        if (pulseDevice_.isOpen())
        {
            std::cout << "Connected\n";

            // Get list of allowed output pins
            bool ok;
            QVector<int> allowedOutputPin = pulseDevice_.getAllowedOutputPin(&ok);
            if (ok)
            {
                allowedOutputPin_ = allowedOutputPin;
            }

            // Set output pin
            ok = pulseDevice_.setOutputPin(allowedOutputPin_[outputPinIndex_]);
            // If bad response or unable to set output pin 
            if (!ok)
            {
                std::cout << "Could not set output pin index, hopefully this will be ok!\n";
                allowedOutputPin_.clear();
                outputPinIndex_ = -1;
            }
            else{
                std::cout << "Set output pin.\n";
            }

            // Set pulse length 
            ok = pulseDevice_.setPulseLength(1.0e6*pulseDuration_);
            if (ok) {
                std::cout << "Set pulse length\n";
            }
            else {
                std::cout << "Could not set pulse length, hopefully this will be ok!!\n";
            }

        }
        else
        {
            std::cout << "Failed to connect.\n";
            rtnStatus.success = false;
            rtnStatus.message = QString("failed to open device");
            return rtnStatus;
        }

        rtnStatus.success = true;
        rtnStatus.message = QString("");
        return rtnStatus;
    }


    void SerialPortOutput::trigger(char output_signal) 
    {
        bool writeSuccess = false;
        if (pulseDevice_.isOpen())
        {
            
            writeSuccess = pulseDevice_.startPulse(output_signal);
            if (!writeSuccess) {

                std::cout << "Write to serial failed " << std::endl;
            }

        }
        else {
            std::cout << "Port not open, not sending pulse\n";
        }
    }


    void SerialPortOutput::disconnectTriggerDev()
    {
        RtnStatus rtnStatus;

        if (pulseDevice_.isOpen())
        {
            std::cout << "Disconnecting...\n";
            pulseDevice_.close();
        }
        else
        {
            std::cout << "Already disconnected\n";
        }

        allowedOutputPin_.clear();

    }

    void SerialPortOutput::setBaudRate(int baudRate)
    {
        bool ok = pulseDevice_.setBaudRate(qint32(baudRate));
        if (ok) {
            std::cout << "Set baudrate\n";
        }
        else {
            std::cout << "Could not set baud rate\n";
        }
    }

}

