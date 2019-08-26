#include "process_scores.hpp" 

namespace bias {


    // public 
 
    ProcessScores::ProcessScores(QObject *parent) : QObject(parent)
    {

        stopped_ = true;
        detectStarted_ = false; 
        save = false;
        frameCount = 0;

    }
   
 
    void ProcessScores::initHOGHOF(QPointer<HOGHOF> hoghof)
    {

        std::cout << "initside" << std::endl;
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
    
        
    void ProcessScores::enqueueFrameDataSender(FrameData frameData)
    {
        acquireLock();
        senderImageQueue_.enqueue(frameData);
        releaseLock();
    }


    void ProcessScores::enqueueFrameDataReceiver(FrameData frameData)
    {
        acquireLock();
        receiverImageQueue_.enqueue(frameData);
        releaseLock();
    }
    

    void ProcessScores::run()
    {

        bool done = false;
        cv::Mat curr_side;
        cv::Mat curr_front;
 
        // Set thread priority to idle - only run when no other thread are running
        QThread *thisThread = QThread::currentThread();
        thisThread -> setPriority(QThread::NormalPriority);
          
        acquireLock();
        stopped_ = false;
        releaseLock();

        bool haveDataCamera0 = false;
        bool haveDataCamera1 = false;
        FrameData frameDataCamera0;
        FrameData frameDataCamera1;
        long long lastProcessedCount = -1;

        while (!done)
        {

            //std::cout << stopped_ << std::endl;

            if (haveDataCamera0 && haveDataCamera1)
            {
                int sizeQueue0 = -1;
                int sizeQueue1 = -1;

                acquireLock();

                // Remove frames from queue #0 until caught up with queue #1
                while ((frameDataCamera0.count < frameDataCamera1.count) && !senderImageQueue_.isEmpty())
                {
                    frameDataCamera0 = senderImageQueue_.dequeue();
                }

                // Remove frames from queue #1 until caught up with queue #0
                while ((frameDataCamera1.count < frameDataCamera0.count) && !receiverImageQueue_.isEmpty())
                {
                    frameDataCamera1 = receiverImageQueue_.dequeue();
                }

                // Advance to the next frame (both queues) if we have already processed this one 
                if ((frameDataCamera0.count == lastProcessedCount) && !senderImageQueue_.isEmpty())
                {
                    frameDataCamera0 = senderImageQueue_.dequeue();
                }
                if ((frameDataCamera1.count == lastProcessedCount) && !receiverImageQueue_.isEmpty())
                {
                    frameDataCamera1 = receiverImageQueue_.dequeue();
                }

                // Get queue sizes - just for general info
                sizeQueue0 = senderImageQueue_.size();
                sizeQueue1 = receiverImageQueue_.size();

                // Check to see if stop has been called
                done = stopped_;
                releaseLock();

                // If frame counts match and are greater than last processed frame count then process the data
                if (frameDataCamera0.count == frameDataCamera1.count)
                {

                    if (((long long)(frameDataCamera0.count) > lastProcessedCount) && detectStarted_)
                    {
                        // --------------------------
                        // Process data here 
                        // --------------------------

                        //normalize frame
                        curr_side = vid_sde->getImage(capture_sde);
                        vid_sde->convertImagetoFloat(curr_side);
                        curr_front = vid_frt->getImage(capture_frt); 
                        vid_frt->convertImagetoFloat(curr_front);
           
                        HOGHOF_side->img.buf = curr_side.ptr<float>(0);
                        HOGHOF_front->img.buf = curr_front.ptr<float>(0);
 
                        genFeatures(HOGHOF_side,frameCount);
                        genFeatures(HOGHOF_front,frameCount);

                        if(save && frameCount == 200) 
                        {

                            std::cout << frameCount << std::endl;
                            write_histoutput("./out_feat/hog_side_" + std::to_string(frameCount) + ".csv", HOGHOF_side->hog_out.data(),
                                 HOGHOF_side->hog_shape.x, HOGHOF_side->hog_shape.y, HOGHOF_side->hog_shape.bin);
                            write_histoutput("./out_feat/hof_side_" + std::to_string(frameCount) + ".csv", HOGHOF_side->hof_out.data(),
                                 HOGHOF_side->hof_shape.x, HOGHOF_side->hof_shape.y, HOGHOF_side->hof_shape.bin);

                            write_histoutput("./out_feat/hog_front_" + std::to_string(frameCount) + ".csv", HOGHOF_front->hog_out.data()
                                         , HOGHOF_front->hog_shape.x, HOGHOF_front->hog_shape.y, HOGHOF_front->hog_shape.bin);
                            write_histoutput("./out_feat/hof_front_" + std::to_string(frameCount) + ".csv", HOGHOF_front->hof_out.data()
                                         , HOGHOF_front->hof_shape.x, HOGHOF_front->hof_shape.y, HOGHOF_front->hof_shape.bin);
                        }

                        if(classifier->isClassifierPathSet && frameCount > 1)
                        {

                            classifier->score = 0.0;
                            classifier->boost_classify(classifier->score, HOGHOF_side->hog_out, HOGHOF_front->hog_out, HOGHOF_side->hof_out,
                                                       HOGHOF_front->hof_out, &HOGHOF_side->hog_shape, &HOGHOF_front->hof_shape,
                                                       classifier->nframes, classifier->model);
                            write_score("classifierscr.csv", frameCount, classifier->score);

                        }

                      
                        frameCount++;

                        // Update last processed frame count 
                        lastProcessedCount = frameDataCamera0.count;

                        // Print some info
                        std::cout << "processed frame " << lastProcessedCount;
                        std::cout << ", queue0 size = " << sizeQueue0 << ", queue1 size = " << sizeQueue1 << std::endl;
                    }
                }
            }
            else
            {
                // Grab initial frame data from queues
                if (!haveDataCamera0)
                {
                    acquireLock();
                    if (!senderImageQueue_.isEmpty())
                    {
                        frameDataCamera0 = senderImageQueue_.dequeue();
                        haveDataCamera0 = true;
                    }
                    releaseLock();

                }
                if (!haveDataCamera1)
                {
                    acquireLock();
                    if (!receiverImageQueue_.isEmpty())
                    {
                        frameDataCamera1 = receiverImageQueue_.dequeue();
                        haveDataCamera1 = true;

                        if(!(HOGHOF_side.isNull()) && !(HOGHOF_front.isNull()))
                        {
                            initHOGHOF(HOGHOF_side);
                            initHOGHOF(HOGHOF_front);
                            classifier->translate_mat2C(&HOGHOF_side->hog_shape,&HOGHOF_front->hog_shape);
                        }
                    }
                    releaseLock();
                }
            }
        }
    }


    //Test
    void ProcessScores::write_histoutput(std::string file,float* out_img, unsigned w, unsigned h,unsigned nbins)
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


    void ProcessScores::write_score(std::string file, int framenum, float score)
    {

        std::ofstream x_out;
        x_out.open(file.c_str(), std::ios_base::app);

        // write score to csv file
        x_out << framenum << ","<< score << "\n";
        x_out.close();

    }

}


























