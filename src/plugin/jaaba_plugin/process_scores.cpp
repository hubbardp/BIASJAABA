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

        stopped_ = true;
        //detectStarted_ = false;
        save = false;
        isSide= false;
        isFront = false;
        processedFrameCount = 0;
        processSide = false;
        processFront = false;
        isProcessed_side = false;
        isProcessed_front = false;
        isHOGHOFInitialised = false;
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

        output_score_dir = cmdlineparams.output_dir;
        isVideo = cmdlineparams.isVideo;
        visualize = cmdlineparams.visualize;
        wait_threshold = cmdlineparams.wait_thres;

        //frame_read_stamps.resize(2798,0);
        //predScoreFront_ = &classifier->predScoreFront;
        //predScoreSide_ = &classifier->predScoreSide;

    }

   
    void ProcessScores::initHOGHOF(QPointer<HOGHOF> hoghof, int img_height, int img_width)
    {

        hoghof->initHOGHOF(img_height,img_width);
        isHOGHOFInitialised = true;

    }

    void ProcessScores::genFeatures(QPointer<HOGHOF> hoghof,int frame)
    {

        hoghof->genFeatures(frame);

    }


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

    void ProcessScores::setScoreQueue(std::shared_ptr<LockableQueue<PredData>> sideScoreQueuePtr,
        std::shared_ptr<LockableQueue<PredData>> frontScoreQueuePtr)
    {
        sideScoreQueuePtr_ = sideScoreQueuePtr;
        frontScoreQueuePtr_ = frontScoreQueuePtr;
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
          
    void ProcessScores::run()
    {

        bool done = false;
        uint64_t time_now;
        double score_ts;
        
//#if isVidInput
        /*((if (isVideo) {
             wait_threshold = 1500;
//#else if
        }else {
            wait_threshold = 1500;
        }*/
//#endif
        uint64_t ts_last_score = INT_MAX, cur_time=0;
        string filename = output_score_dir + "classifier_score.csv";
        // Set thread priority to idle - only run when no other thread are running
        QThread *thisThread = QThread::currentThread();
        thisThread -> setPriority(QThread::NormalPriority);
          
        acquireLock();
        stopped_ = false;
        releaseLock();
        
        while (!done)
        {
            if (mesPass_) 
            {

                if (processSide)
                {

                    cudaSetDevice(0);
                    genFeatures(HOGHOF_frame, processedFrameCount + 1);
                    acquireLock();
                    processSide = false;
                    isProcessed_side = true;
                    releaseLock();

                }

                if (processFront) {

                    cudaSetDevice(1);
                    genFeatures(HOGHOF_partner, processedFrameCount + 1);
                    acquireLock();
                    processFront = false;
                    isProcessed_front = true;
                    releaseLock();
                }

            }
            else {
                skipFront = false;
                skipSide = false;

                if (sideScoreQueuePtr_->empty() && frontScoreQueuePtr_->empty()) {

                    continue;

                } 
                else if (!sideScoreQueuePtr_->empty() && !frontScoreQueuePtr_->empty()) {


                    frontScoreQueuePtr_->acquireLock();
                    predScorePartner = frontScoreQueuePtr_->front();
                    frontScoreQueuePtr_->releaseLock();

                    sideScoreQueuePtr_->acquireLock();
                    predScore = sideScoreQueuePtr_->front();
                    sideScoreQueuePtr_->releaseLock();

                    // to keep up with frame where both views are skipped 
                    if (scoreCount < predScore.frameCount)
                    {
                        //if (predScore.frameCount > predScorePartner.frameCount)
                        //    frontScoreQueuePtr_->pop();
                        scoreCount++;
                        continue;
                    }

                    // to keep up with frames where both views are skipped
                    if (scoreCount < predScorePartner.frameCount)
                    {
                        //if (predScorePartner.frameCount > predScore.frameCount)
                        //    sideScoreQueuePtr_->pop();
                        scoreCount++;
                        continue;
                    }
                   
                    
                    if (scoreCount > predScore.frameCount) {
                        sideScoreQueuePtr_->pop();
                        continue;
                    }


                    if (scoreCount > predScorePartner.frameCount){
                        frontScoreQueuePtr_->pop();
                        continue;
                    }
                    
                    
                    if (predScore.frameCount == predScorePartner.frameCount)
                    {

                        classifier->addScores(predScore.score, predScorePartner.score);

                        skipSide = false;
                        skipFront = false;

                        sideScoreQueuePtr_->acquireLock();
                        sideScoreQueuePtr_->pop();
                        sideScoreQueuePtr_->releaseLock();

                        frontScoreQueuePtr_->acquireLock();
                        frontScoreQueuePtr_->pop();
                        frontScoreQueuePtr_->releaseLock();
         
                        if (isVideo) {
                            time_now = gettime->getPCtime();
                            scores[scoreCount].score_ts = time_now;

                        }
                        else {
                            nidaq_task_->getNidaqTimeNow(read_ondemand_);
                            scores[scoreCount - 1].score_ts = read_ondemand_;
                        }


                        scores[scoreCount].score = classifier->finalscore.score;
                        scores[scoreCount].frameCount = predScore.frameCount;
                        scores[scoreCount].view = 3;
                        
                        scores[scoreCount].score_side_ts = predScore.score_side_ts;
                        scores[scoreCount].score_front_ts = predScorePartner.score_front_ts;
                                              // - max(predScore.score_ts, predScorePartner.score_ts);
                        //write_score("classifierscr.csv", scoreCount, scores[scoreCount-1]);


                        if(visualize){

                            visualizeScores(classifier->finalscore.score);
                
                        }
                        scoreCount++;
                                             
                    }
                    
                }else if (!frontScoreQueuePtr_->empty()) {

                    frontScoreQueuePtr_->acquireLock();
                    predScorePartner = frontScoreQueuePtr_->front();
                    frontScoreQueuePtr_->releaseLock();

                    // check if this is not already a processed scoreCount
                    //if (scoreCount > predScorePartner.frameCount) {
                    //    frontScoreQueuePtr_->pop();
                    //    continue;
                    //}

                    if (predScorePartner.frameCount > scoreCount)
                        scoreCount++;

                    time_now = gettime->getPCtime();
                    score_ts = predScorePartner.score_front_ts;
                    
                    if ((time_now - score_ts) > wait_threshold)
                    {
                        skipSide = true;
                    }

                }
                else if (!sideScoreQueuePtr_->empty()) {

                    sideScoreQueuePtr_->acquireLock();
                    predScore = sideScoreQueuePtr_->front();
                    sideScoreQueuePtr_->releaseLock();

                    // check if this is not already a processed scoreCount
                    //if (scoreCount > predScore.frameCount) {
                    //    sideScoreQueuePtr_->pop();
                    //    continue;
                    //}

                    if (predScore.frameCount > scoreCount)
                        scoreCount++;

                    time_now = gettime->getPCtime();
                    score_ts = predScore.score_side_ts;
                    
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

                            sideScoreQueuePtr_->acquireLock();
                            sideScoreQueuePtr_->pop();
                            sideScoreQueuePtr_->releaseLock();

                            //write_score("classifierscr.csv", scoreCount, predScore);

                            if (isVideo) {
                                time_now = gettime->getPCtime();
                                scores[scoreCount].score_ts = time_now;
                            }
                            else {
                                //#else
                                nidaq_task_->getNidaqTimeNow(read_ondemand_);
                                scores[scoreCount].score_ts = read_ondemand_;
                            }
                            //#endif
                            scores[scoreCount].score[0] = predScore.score[0];
                            scores[scoreCount].frameCount = predScore.frameCount;
                            scores[scoreCount].view = 1;
                            scores[scoreCount].score_side_ts = predScore.score_side_ts;

                            if (visualize) {
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

                            frontScoreQueuePtr_->acquireLock();
                            frontScoreQueuePtr_->pop();
                            frontScoreQueuePtr_->releaseLock();
                            //#if isVidInput                         
                            if (isVideo) {
                                time_now = gettime->getPCtime();
                                scores[scoreCount].score_ts = time_now;
                            }
                            else {
                                //#else
                                nidaq_task_->getNidaqTimeNow(read_ondemand_);
                                scores[scoreCount].score_ts = read_ondemand_;
                            }
                            //#endif

                            //write_score("classifierscr.csv", scoreCount, predScorePartner);
                            time_now = gettime->getPCtime();
                            scores[scoreCount].score[0] = predScorePartner.score[0];
                            scores[scoreCount].frameCount = predScorePartner.frameCount;
                            scores[scoreCount].view = 2;
                            scores[scoreCount].score_front_ts = predScorePartner.score_front_ts;

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

            if (scoreCount == (numFrames-1)) {
                std::cout << "Score file name " << filename << std::endl;
                std::cout << "Writing score...." << std::endl;
                write_score_final(filename,numFrames-1, scores);
                std::cout << "Written ...." << std::endl;
                
            }

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
            x_out << pred_score[frm_id].score_ts << "," << pred_score[frm_id].score_side_ts 
                << "," << pred_score[frm_id].score_front_ts << "," 
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



}

