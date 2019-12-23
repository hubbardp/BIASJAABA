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
        processedFrameCount = -1;
        processSide = false;
        processFront = false;
        isProcessed_side = false;
        isProcessed_front = false;
        isHOGHOFInitialised = false;

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
 
        // Set thread priority to idle - only run when no other thread are running
        QThread *thisThread = QThread::currentThread();
        thisThread -> setPriority(QThread::NormalPriority);
          
        acquireLock();
        stopped_ = false;
        releaseLock();
        
        while (!done)
        {

<<<<<<< HEAD
            /*std::cout << "running front " << processSide << " " << processFront << std::endl;
            if(processFront)
=======
            if(processSide)
>>>>>>> 2gpu_threading
            {

                cudaSetDevice(0);
                genFeatures(HOGHOF_frame, processedFrameCount+1);
                acquireLock();
                processSide = false;
                isProcessed_side = true;
                releaseLock();
                 
            }*/

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
        //for(int frame_id = 0;frame_id < framenum;frame_id++)
        x_out << framenum << "," << score << "\n";

        x_out.close();

    }

   
    void ProcessScores::write_time(std::string file, int framenum, std::vector<float> timeVec)
    {

        std::ofstream x_out;
        x_out.open(file.c_str(), std::ios_base::app);

        for(int frame_id= 0; frame_id < framenum; frame_id++)
        {

            x_out << frame_id << "," << timeVec[frame_id] << "\n";        

        }

    }


}


























