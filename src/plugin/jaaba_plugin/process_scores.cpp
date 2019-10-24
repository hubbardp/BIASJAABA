#include "process_scores.hpp"
#include <cuda_runtime_api.h> 

namespace bias {


    // public 
 
    ProcessScores::ProcessScores(QObject *parent) : QObject(parent) 
    {

        stopped_ = true;
        detectStarted_ = false; 
        save = false;
        isSide= false;
        isFront = false;
        frameCount = 0;
        isHOGHOFInitialised = false;
        
        //partnerPluginPtr_ = partnerPluginPtr;
        //qRegisterMetaType<ShapeData>("ShapeData");
        //connect(partnerPluginPtr, SIGNAL(newShapeData(ShapeData)), this, SLOT(onNewShapeData(ShapeData)));          

    }
   
 
    void ProcessScores::initHOGHOF(QPointer<HOGHOF> hoghof, int img_height, int img_width)
    {

        //hoghof->loadImageParams(384, 260);
        int nDevices;
        cudaError_t err = cudaGetDeviceCount(&nDevices); 
        if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
        std::cout << img_height << " " << img_width << " " << nDevices << std::endl;
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
    
        
    void ProcessScores::enqueueFrameData(FrameData frameData)
    {
        acquireLock();
        frameQueue_.enqueue(frameData);
        releaseLock();
    }


    /*void ProcessScores::enqueueFrameDataReceiver(FrameData frameData)
    {
        acquireLock();
        receiverImageQueue_.enqueue(frameData);
        releaseLock();
    }*/
    

    void ProcessScores::run()
    {

        bool done = false;
 
        // Set thread priority to idle - only run when no other thread are running
        QThread *thisThread = QThread::currentThread();
        thisThread -> setPriority(QThread::NormalPriority);
          
        acquireLock();
        stopped_ = false;
        releaseLock();

        /*bool haveDataCamera0 = false;
        bool haveDataCamera1 = false;
        FrameData frameDataCamera0;
        FrameData frameDataCamera1;*/
        bool haveDataCamera = false;
        FrameData frameDataCamera;
        long long lastProcessedCount = -1;
        int sizeQueue = -1;

        while (!done)
        {

            //if (haveDataCamera0 && haveDataCamera1)
            if (haveDataCamera)
            {

                acquireLock();

                // Remove frames from queue #0 until caught up with queue #1
                /*while ((frameDataCamera0.count < frameDataCamera1.count) && !senderImageQueue_.isEmpty())
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
                }*/

                if(!frameQueue_.isEmpty() && (frameDataCamera.count == lastProcessedCount))
                {

                    frameDataCamera = frameQueue_.dequeue(); 
                
                    // Get queue sizes - just for general info               
                    sizeQueue = frameQueue_.size();

                    if(isSide)
                    {
                        write_score("buf0.csv", lastProcessedCount, sizeQueue);
                    }

                    if(isFront)
                    {
                        write_score("buf1.csv", lastProcessedCount, sizeQueue);
                    }
 
                }
                // Check to see if stop has been called
                done = stopped_;
                releaseLock();
  
                // If frame counts match and are greater than last processed frame count then process the data
                //if (frameDataCamera0.count == frameDataCamera1.count)
                //{

                 if (((long long)(frameDataCamera.count) > lastProcessedCount) && detectStarted_)
                 {
                        // --------------------------
                        // Process data here 
                        // --------------------------

                        //Test development capture framme and normalize frame
                        /*if(capture_.isOpened())
                        {
                            curr_frame = vid_->getImage(capture_);
                            vid_->convertImagetoFloat(curr_frame);
                            grey_frame = curr_frame;
                        }*/

                        curr_frame = frameDataCamera.image;
                         
                         
                        //curr_side = frameDataCamera0.image;
                        //curr_front = frameDataCamera1.image;

                        //std::cout << "rows " << curr_side.rows << "cols " << curr_side.cols << std::endl; 
                                                    
                        // convert the frame into RGB2GRAY
                        if(curr_frame.channels() == 3) 
                        {
                            cv::cvtColor(curr_frame, curr_frame, cv::COLOR_BGR2GRAY);
                        }

                        // convert the frame into float32
                        curr_frame.convertTo(grey_frame, CV_32FC1);
                        grey_frame = grey_frame / 255;
                        //cv::imwrite("out_feat/img" + std::to_string(frameCount) + ".bmp",curr_side);                     

                        int nDevices;
                        cudaError_t err = cudaGetDeviceCount(&nDevices);
                        if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

                        if(nDevices>=2)
                        {
                           
                            if(isSide)
                            {
                                //cudaSetDevice(0);
                                //GpuTimer timer2;
                                //timer2.Start();
                                HOGHOF_frame->img.buf = grey_frame.ptr<float>(0);
                                //genFeatures(HOGHOF_frame, frameCount);
                                genFeatures(HOGHOF_frame, lastProcessedCount);
                                //timer2.Stop();
                                //write_score("timing_gpu1.csv", lastProcessedCount, timer2.Elapsed()/1000);
                            }

                            if(isFront)
                            {                             
                                //cudaSetDevice(1);
                                HOGHOF_frame->img.buf = grey_frame.ptr<float>(0);     
                                //genFeatures(HOGHOF_frame, frameCount);                       
                                genFeatures(HOGHOF_frame, lastProcessedCount);
                            }
                            //timer1.Stop();
                            //write_score("timing_double.csv", lastProcessedCount, timer1.Elapsed()/1000);
                            //genFeatures(HOGHOF_front, frameCount);

                        } else {

                            GpuTimer timer1;
                            timer1.Start();
                            HOGHOF_frame->img.buf = grey_frame.ptr<float>(0);
                            genFeatures(HOGHOF_frame, lastProcessedCount);
                            //genFeatures(HOGHOF_side, frameCount);
                            HOGHOF_frame->img.buf = grey_frame.ptr<float>(0);
                            genFeatures(HOGHOF_frame, lastProcessedCount);
                            timer1.Stop();
                            //write_score("timing_single.csv", lastProcessedCount, timer1.Elapsed()/1000);
                            //genFeatures(HOGHOF_front, frameCount);

                        }


                        if(save && frameCount == 1000 && isSide) 
                        {

                            write_histoutput("./out_feat/hog_side_" + std::to_string(lastProcessedCount) + ".csv", HOGHOF_frame->hog_out.data(),
                                 HOGHOF_frame->hog_shape.x, HOGHOF_frame->hog_shape.y, HOGHOF_frame->hog_shape.bin);
                            write_histoutput("./out_feat/hof_side_" + std::to_string(lastProcessedCount) + ".csv", HOGHOF_frame->hof_out.data(),
                                 HOGHOF_frame->hof_shape.x, HOGHOF_frame->hof_shape.y, HOGHOF_frame->hof_shape.bin);
                        }


                        if(save && frameCount == 1000 && isFront)
                        {

                            write_histoutput("./out_feat/hog_front_" + std::to_string(lastProcessedCount) + ".csv", HOGHOF_frame->hog_out.data()
                                         , HOGHOF_frame->hog_shape.x, HOGHOF_frame->hog_shape.y, HOGHOF_frame->hog_shape.bin);
                            write_histoutput("./out_feat/hof_front_" + std::to_string(lastProcessedCount) + ".csv", HOGHOF_frame->hof_out.data()
                                         , HOGHOF_frame->hof_shape.x, HOGHOF_frame->hof_shape.y, HOGHOF_frame->hof_shape.bin);
                        }
                         

                        /*if(classifier->isClassifierPathSet && lastProcessedCount > 0)
                        {

                            classifier->score = 0.0;
                            classifier->boost_classify(classifier->score, HOGHOF_frame->hog_out, HOGHOF_frame->hog_out, HOGHOF_frame->hof_out,
                                                       HOGHOF_frame->hof_out, &HOGHOF_frame->hog_shape, &HOGHOF_frame->hof_shape,
                                                       classifier->nframes, classifier->model);
                            //write_score("classifierscr.csv", lastProcessedCount, classifier->score);
                            //write_score("buffer_0.csv", lastProcessedCount, sizeQueue0);
                            //write_score("buffer_1.csv", lastProcessedCount, sizeQueue1);

                        }*/

                     
                        // Update last processed frame count 
                        //frameCount++;
                        lastProcessedCount = frameDataCamera.count;

                        // Print some info
                        //std::cout << "processed frame " << lastProcessedCount;
                        //std::cout << ", queue0 size = " << sizeQueue0 << ", queue1 size = " << sizeQueue1 << std::endl;
                  }
               
            }
            else
            {
                // Grab initial frame data from queues
                if (!haveDataCamera)
                {
                    acquireLock();
                    if (!frameQueue_.isEmpty())
                    {
                        frameDataCamera = frameQueue_.dequeue();
                        haveDataCamera = true;
                        std::cout << "initializing" << std::endl; 
                        if(!(HOGHOF_frame.isNull()))
                        {
                            //cudaSetDevice(1);
                            if(isSide)
                            {
                                //cudaSetDevice(0);
                                initHOGHOF(HOGHOF_frame, frameDataCamera.image.rows, frameDataCamera.image.cols);
                                //initHOGHOF(HOGHOF_frame, 260, 384);
                            }
                            if(isFront)
                            {
                                //cudaSetDevice(1);
                                //initHOGHOF(HOGHOF_frame, 260, 384);
                                initHOGHOF(HOGHOF_frame, frameDataCamera.image.rows, frameDataCamera.image.cols);
                            }
                        }
                    }
                    releaseLock();

                }
                                
                //check if HOGHOF initialized on gpu and initialize classifier params 
                acquireLock();
                if(haveDataCamera)
                {
                    //classifier->translate_mat2C(&HOGHOF_side->hog_shape,&HOGHOF_front->hog_shape); //have to gifure this out 
                }
                releaseLock();
            }
        }
     
        // Some test code delete later

        /*bool done = false;
        int sizeQueue0 = -1;
        int sizeQueue1 = -1;
        long long lastProcessedCount;
        FrameData frameDataCamera0;
        FrameData frameDataCamera1;

        while(!done) 
        {
            acquireLock();

            if(!receiverImageQueue_.isEmpty())
            {
                          
                frameDataCamera1 = receiverImageQueue_.dequeue();
                sizeQueue1 = receiverImageQueue_.size();
                lastProcessedCount = frameDataCamera1.count;
                curr_side = frameDataCamera1.image;
                curr_side.convertTo(grey_sde, CV_32FC1);
                grey_sde = grey_sde / 255;
                HOGHOF_side->img.buf = grey_sde.ptr<float>(0);
                genFeatures(HOGHOF_side, lastProcessedCount);
                write_score("buffer_1.csv", lastProcessedCount, sizeQueue1);           

            }

            if(!senderImageQueue_.isEmpty())
            {
         
                frameDataCamera0 = senderImageQueue_.dequeue();
                sizeQueue0 = senderImageQueue_.size();
                lastProcessedCount = frameDataCamera0.count;
                write_score("buffer_0.csv", lastProcessedCount, sizeQueue0);
            }

            releaseLock();
        }*/
    }


    // Private slots
    /*void ProcessScores::onNewShapeData(ShapeData data)
    {
        std::cout << "called" << std::endl; 
        if(isSide)
        {

            //get frame from sender plugin
            HOGHOF_partner->hog_shape.x = data.shapex;
            HOGHOF_partner->hog_shape.y = data.shapey;
            HOGHOF_partner->hog_shape.bin = data.bins;
            std::cout << HOGHOF_partner->hog_shape.x << " " << HOGHOF_partner->hog_shape.bin << std::endl;
        }


        if(isFront)
        {
            //get frame from sender plugin
            HOGHOF_partner->hog_shape.x = data.shapex;
            HOGHOF_partner->hog_shape.y = data.shapey;
            HOGHOF_partner->hog_shape.bin = data.bins;
            std::cout << HOGHOF_partner->hog_shape.x << " " << HOGHOF_partner->hog_shape.bin << std::endl;

        }

    }*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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


























