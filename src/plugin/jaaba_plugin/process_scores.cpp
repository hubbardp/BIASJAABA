#include "process_scores.hpp" 

namespace bias {


    ProcessScores::ProcessScores()
    {

        stopped_ = true;


    }
   
 
    void ProcessScores::initHOGHOF(QPointer<HOGHOF> hoghof)
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

    
    void ProcessScores::setupHOGHOF_side()
    {


        //Test DEvelopment
        QString file("/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_sde.avi");
        //vid_sde = new videoBackend(file);
        //capture_sde = vid_sde->videoCapObject();
        ///

        //HOGHOF *hoghofside = new HOGHOF(this);
        //HOGHOF_side = hoghofside;

     }


    void ProcessScores::setupHOGHOF_front()
    {


        QString file("/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_frt.avi");
        //vid_front = new videoBackend(file);
        //capture_front = vid_front->videoCapObject();


        //HOGHOF *hoghoffront = new HOGHOF(this);
        //HOGHOF_front = hoghoffront;
        //HOGHOF_front->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
        //HOGHOF_front->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
        //HOGHOF_front->CropParam_file = pathtodir_->placeholderText() + CropFrontParamFilePtr_->placeholderText();
 
    }


    void ProcessScores::run()
    {


        bool done = false;
 
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
                    if (((long long)(frameDataCamera0.count) > lastProcessedCount))
                    {
                        // --------------------------
                        // Process data here 
                        // --------------------------

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
                    std::cout << "inside framegrab" << std::endl;
                    if (!senderImageQueue_.isEmpty())
                    {
                        std::cout << "grabbing first frame " << std::endl; 
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
                        //initHOGHOF(HOGHOF_side);
                        //initHOGHOF(HOGHOF_front);
                    }
                    releaseLock();
                }
            }
        }
    }
}


























