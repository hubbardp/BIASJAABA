#include "process_scores.hpp"
#include <cuda_runtime_api.h> 

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
        frameCount_ = -1;
        partner_frameCount_ = -1;
        scoreCount = 1;
        getTime_ = getTime;
        skip_frameFront = 0;
        skip_frameSide = 0;
        side_read_time_ = 0;
        front_read_time_ = 0;
        frame_read_stamps.resize(2798,0);

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
        int64_t time_now;
 
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

            } else { 
                
                /*if (!sideScoreQueue.empty() && !frontScoreQueue.empty())
                {

                    acquireLock();
                    predScoreFront_ = frontScoreQueue.front();
                    predScoreSide_ = sideScoreQueue.front();
                    releaseLock();

                    if (predScoreSide_.second == predScoreFront_.second) 
                    {

                        classifier->addScores(predScoreSide_.first, predScoreFront_.first);

                        isProcessed_front = 0;
                        isProcessed_side = 0;
                        score_calculated_ = 0;

                        acquireLock();
                        frontScoreQueue.pop_front();
                        sideScoreQueue.pop_front();
                        releaseLock();

                        std::cout << "side and front " << std::endl;
                        write_score("classifierscr.csv", scoreCount, classifier->score[0]);
                        scoreCount++;
                        std::cout << "ScoreCount " << scoreCount << std::endl;

                    } else if (scoreCount == predScoreFront_.second) {

                        predScoreSide_ = std::make_pair<vector<float>, int>({ 0.0,0.0,0.0,0.0,0.0, 0.0 }, 0);
                        classifier->addScores(predScoreSide_.first, predScoreFront_.first);

                        acquireLock();
                        frontScoreQueue.pop_front();
                        releaseLock();

                        std::cout << "ScoreCount front: " << scoreCount << std::endl;
                        write_score("classifierscr.csv", scoreCount, classifier->score[0]);
                        scoreCount++;

                    } else if (scoreCount == predScoreSide_.second) {

                        predScoreFront_ = std::make_pair<vector<float>, int>({ 0.0,0.0,0.0,0.0,0.0, 0.0 }, 0);
                        classifier->addScores(predScoreSide_.first, predScoreFront_.first);

                        acquireLock();
                        sideScoreQueue.pop_front();
                        releaseLock();

                        std::cout << "ScoreCount side: " << scoreCount << std::endl;
                        write_score("classifierscr.csv", scoreCount, classifier->score[0]);
                        scoreCount++;

                    } else {

                        std::cout << "ScoreCount skipped: " << scoreCount << std::endl;
                        predScoreFront_ = std::make_pair<vector<float>, int>({ 0.0,0.0,0.0,0.0,0.0, 0.0 }, 0);
                        predScoreSide_ = std::make_pair<vector<float>, int>({ 0.0,0.0,0.0,0.0,0.0, 0.0 }, 0);
                        classifier->addScores(predScoreSide_.first, predScoreFront_.first);
                        write_score("classifierscr.csv", scoreCount, classifier->score[0]);
                        scoreCount++;
                    }
                
                } else {

                    if (!frontScoreQueue.empty() && skip_frameSide == 1) 
                    {

                        acquireLock();
                        predScoreFront_ = frontScoreQueue.front();
                        predScoreSide_ = std::make_pair<vector<float>, int>({ 0.0,0.0,0.0,0.0,0.0, 0.0 }, 0);
                        releaseLock();

                        if (scoreCount == predScoreFront_.second)
                        {
                            classifier->addScores(predScoreSide_.first, predScoreFront_.first);

                            acquireLock();
                            frontScoreQueue.pop_front();
                            //skip_frameSide = 0;
                            releaseLock();

                            std::cout << "Only Front" << std::endl;
                            write_score("classifierscr.csv", scoreCount, classifier->score[0]);

                        }
                        scoreCount++;
                        std::cout << "ScoreCount " << scoreCount << std::endl;
                        
                    }else if (!sideScoreQueue.empty() && skip_frameFront == 1) {

                        acquireLock();
                        predScoreFront_ = std::make_pair<vector<float>, int>({ 0.0,0.0,0.0,0.0,0.0, 0.0 }, 0); 
                        predScoreSide_ = sideScoreQueue.front();
                        releaseLock();

                        if (scoreCount == predScoreSide_.second)
                        {
                            classifier->addScores(predScoreSide_.first, predScoreFront_.first);

                            acquireLock();
                            sideScoreQueue.pop_front();
                            //skip_frameFront = 0;
                            releaseLock();

                            std::cout << "Only Side" << std::endl;
                            write_score("classifierscr.csv", scoreCount, classifier->score[0]);
                        }
                        scoreCount++;
                        std::cout << "ScoreCount " << scoreCount << std::endl;

                    } else {}

                }*/
                
            }

            acquireLock();
            done = stopped_;
            releaseLock();

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


    void ProcessScores::write_score(std::string file, int framenum, float score)
    {

        std::ofstream x_out;
        x_out.open(file.c_str(), std::ios_base::app);

        // write score to csv file
        //for(int frame_id = 0;frame_id < framenum;frame_id++)
        x_out << framenum << "," << score << "\n";

        x_out.close();

    }

}

