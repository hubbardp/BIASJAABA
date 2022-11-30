#include "process_scores.hpp"
#include <cuda_runtime_api.h> 

#define DEBUG 1

namespace bias {


    // public 
 
    ProcessScores::ProcessScores(QObject *parent, bool mesPass, 
                                 std::shared_ptr<Lockable<GetTime>> getTime) : QObject(parent)
    {

        stopped_ = true;
        detectStarted_ = false; 
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
        scoreCount = 1;
        getTime_ = getTime;
        skipFront = 0;
        skipSide = 0;
        side_read_time_ = 0;
        front_read_time_ = 0;

#if DEBUG
        scores.resize(10000);
#endif
        //frame_read_stamps.resize(2798,0);
        //predScoreFront_ = &classifier->predScoreFront;
        //predScoreSide_ = &classifier->predScoreSide;

    }

   
    void ProcessScores::initHOGHOF(QPointer<HOGHOF> hoghof, int img_height, int img_width)
    {

        int nDevices;
        cudaError_t err = cudaGetDeviceCount(&nDevices); 
        if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
        hoghof->loadImageParams(img_width, img_height);
        struct HOGContext hogctx = HOGInitialize(logger, hoghof->HOGParams, img_width, img_height, hoghof->Cropparams);
        struct HOFContext hofctx = HOFInitialize(logger, hoghof->HOFParams, hoghof->Cropparams);
        hoghof->hog_ctx = (HOGContext*)malloc(sizeof(hogctx));
        hoghof->hof_ctx = (HOFContext*)malloc(sizeof(hofctx));
        memcpy(hoghof->hog_ctx, &hogctx, sizeof(hogctx));
        memcpy(hoghof->hof_ctx, &hofctx, sizeof(hofctx));
        //hoghof->startFrameSet = false;

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
    
        isHOGHOFInitialised = true;

    }


    void ProcessScores::genFeatures(QPointer<HOGHOF> hoghof,int frame)
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


    void ProcessScores::detectOn()
    {

        detectStarted_ = true;

    }


    void ProcessScores::detectOff()
    {

        detectStarted_ = false;

    }

          
    void ProcessScores::run()
    {

        bool done = false;
        uint64_t time_now;
        double score_ts;
        double wait_threshold = 10000;
        unsigned int numFrames = 2498;
        uint64_t ts_last_score = INT_MAX, cur_time=0;
        string filename = "C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/plugin_latency/nidaq/multi/2c5ba_9_8_2022/classifier_trial1.csv";
    
        // Set thread priority to idle - only run when no other thread are running
        QThread *thisThread = QThread::currentThread();
        thisThread -> setPriority(QThread::TimeCriticalPriority);
          
        acquireLock();
        stopped_ = false;
        releaseLock();
        
        while (!done)
        {
            /*if (mesPass_) 
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
                
                if (sideScoreQueue.empty() && frontScoreQueue.empty()) {

                    //cur_time = getTime_->getPCtime();
                    //if ((cur_time - ts_last_score) > wait_threshold) {
                    //    scoreCount++;
                    //    ts_last_score = cur_time;
                    //}
                    //else {
                    //    continue;
                    //}
                    continue;

                }
                else if (!sideScoreQueue.empty() && !frontScoreQueue.empty()) {

                    acquireLock();
                    predScorePartner = frontScoreQueue.front();
                    predScore = sideScoreQueue.front();
                    releaseLock();

                    // not sure if this condition occurs
                    if (scoreCount < predScore.frameCount)
                    {
                        if (predScore.frameCount > predScorePartner.frameCount)
                            frontScoreQueue.pop_front();
                        scoreCount++;
                        continue;
                    }

                    // not sure if this condition occurs
                    if (scoreCount < predScorePartner.frameCount)
                    {
                        if (predScorePartner.frameCount > predScore.frameCount)
                            sideScoreQueue.pop_front();
                        scoreCount++;
                        continue;
                    }
                   
                    if (scoreCount > predScore.frameCount) {
                        sideScoreQueue.pop_front();
                        continue;
                    }


                    if (scoreCount > predScorePartner.frameCount){
                        frontScoreQueue.pop_front();
                        continue;
                    }
                    
                    
                    if (predScore.frameCount == predScorePartner.frameCount)
                    {

                        classifier->addScores(predScore.score, predScorePartner.score);

                        skipSide = false;
                        skipFront = false;
                        sideScoreQueue.pop_front();
                        frontScoreQueue.pop_front();
                        time_now = getTime_->getPCtime();
                        
                        scores[scoreCount - 1].score[0] = classifier->finalscore.score[0];
                        scores[scoreCount-1].frameCount = predScore.frameCount;
                        scores[scoreCount-1].view = 3;
                        scores[scoreCount - 1].score_ts = time_now;
                                              // - max(predScore.score_ts, predScorePartner.score_ts);
                        //write_score("classifierscr.csv", scoreCount, scores[scoreCount-1]);
                        scoreCount++;
                        
                    }
                    
                }
                else if (!frontScoreQueue.empty()) {

                    acquireLock();
                    predScorePartner = frontScoreQueue.front();
                    releaseLock();

                    // check if this is not already a processed scoreCount
                    //if (scoreCount > predScorePartner.frameCount) {
                    //    frontScoreQueue.pop_front();
                    //    continue;
                    //}

                    //if (predScorePartner.frameCount > scoreCount)
                    //    std::cout << "Front ahead of score" << std::endl;

                    time_now = getTime_->getPCtime();
                    score_ts = predScorePartner.score_ts;
                    scores[scoreCount - 1].score_front_ts = score_ts;
                    //if ((time_now - score_ts) > wait_threshold)
                    //{
                    //    skipSide = true;
                    //}

                }
                else if (!sideScoreQueue.empty()) {

                    acquireLock();
                    predScore = sideScoreQueue.front();
                    releaseLock();

                    // check if this is not already a processed scoreCount
                    //if (scoreCount > predScore.frameCount) {
                    //    sideScoreQueue.pop_front();
                    //    continue;
                    //}

                    //if (predScore.frameCount > scoreCount)
                    //    std::cout << "side is ahead of score" << std::endl;

                    time_now = getTime_->getPCtime();
                    score_ts = predScore.score_ts;
                    scores[scoreCount - 1].score_side_ts = score_ts;
                    //if ((time_now - score_ts) > wait_threshold)
                    //{
                    //    skipFront = true;
                       
                    //}

                }

                if (skipFront)
                {
                    if (!skipSide)
                    {
                        skipFront = false;
                        if (scoreCount == predScore.frameCount)
                        {

                            acquireLock();
                            sideScoreQueue.pop_front();
                            releaseLock();

                            //write_score("classifierscr.csv", scoreCount, predScore);

                            scores[scoreCount-1].score[0] = predScore.score[0];
                            scores[scoreCount-1].frameCount = predScore.frameCount;
                            scores[scoreCount-1].view = 1;
                            scores[scoreCount-1].score_ts = predScore.score_ts;
                           
                            scoreCount++;
                        }
                        
                    }

                }else if (skipSide) {

                    if (!skipFront)
                    {
                        skipSide = false;
                        if (scoreCount == predScorePartner.frameCount)
                        {

                            acquireLock();
                            frontScoreQueue.pop_front();
                            releaseLock();
                          
                            //write_score("classifierscr.csv", scoreCount, predScorePartner);

                            scores[scoreCount-1].score[0] = predScorePartner.score[0];
                            scores[scoreCount-1].frameCount = predScorePartner.frameCount;
                            scores[scoreCount-1].view = 2;
                            scores[scoreCount-1].score_ts = predScorePartner.score_ts;
                            
                            scoreCount++;

                        }
                        
                    }
                }
                
            }*/

            acquireLock();
            done = stopped_;
            releaseLock();

            /*if (scoreCount >= (numFrames)) {

                std::cout << "Writing ...." << std::endl;
                write_score_final(filename,numFrames, scores);
                std::cout << "Written ...." << std::endl;
                break;
            }*/

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

    void ProcessScores::write_score(std::string file, int framenum, PredData& score)
    {

        std::ofstream x_out;
        x_out.open(file.c_str(), std::ios_base::app);
        if (framenum == 0)
            x_out << "Score ts," << "Score," << " FrameNumber," << "View" << "\n";
        
        // write score to csv file
        x_out << score.score_ts << "," << score.score[0] << "," 
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
                << "," << pred_score[frm_id].score_front_ts << "," << pred_score[frm_id].score[0]
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

