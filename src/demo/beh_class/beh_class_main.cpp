#include "HOGHOF.hpp"
#include "beh_class.hpp"
#include "spin_utils.hpp"
#include "utils_spin.hpp"
#include "SpinnakerC.h"
#include "cuda_runtime_api.h"

//#include <vector>
//#include "timer.h"
//#include <fstream>
//#include <iostream>
#include <queue>
#include <stdlib.h>
#include <QDebug>
#include <QThreadPool>
#include <QQueue>

#define isSkip 0

typedef std::pair<vector<float>, int> PredData;

//using namespace bias;

/*void parse_args(int nargs, char** arg_inputs, std::string& view) {

    std::string v = "side";

    // parse all the arguments from the command line
    while((opt = getopt(nargs,arg_inputs,"v:")) != -1) {

        options.insert(opt);

        switch(opt){

            case 'v': g = optarg;
                break;
            default: std::cout << "Missing arguments:<help>"
                exit(-1);
        }

    }

}*/

bool destroySpinImage(spinImage &hImage)
{
    bool rval = true;
    if (hImage != nullptr)
    {
        //std::cout << "destroy" << std::endl;
        spinError err = spinImageDestroy(hImage);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            rval = false;
        }
        //else
        //{
        //    hImage = nullptr;
        //}
    }
    return rval;
}

template <typename T>
void write_time(std::string filename, int framenum, std::vector<T>& timeVec)
{

    std::ofstream x_out;
    x_out.open(filename.c_str(), std::ios_base::app);

    for (int frame_id = 0; frame_id < framenum - 1; frame_id++)
    {

        x_out << timeVec[frame_id] << "\n";
    }

    x_out.close();

}

void initiateVidSkips(priority_queue<int, vector<int>, greater<int>>& skip_frames,
                      unsigned int numFrames)
{

    //srand(time(NULL));

    int no_of_skips = 1;
    int framenumber;

    for (int j = 0; j < no_of_skips; j++)
    {
        framenumber = rand() % numFrames;
        skip_frames.push(framenumber);
        std::cout << framenumber << std::endl;

    }

}

int main(int argc, char* argv[]) {

    //srand(time(NULL));

    //nviews - temp should be command line arg
    const int nviews = 2;
    int numFrames = 2498; //frames to process

    priority_queue<int, vector<int>, greater<int>>skipframes_view1; // frames to skip 
    priority_queue<int, vector<int>, greater<int>>skipframes_view2;

    QQueue<PredData> frontScoreQueue; // front score buffers
    QQueue<PredData> sideScoreQueue; // side score buffers

    //initiateVidSkips(skipframes_view1, numFrames);
    //initiateVidSkips(skipframes_view2, numFrames);
  
    int frameSkip = 5;
    bool isSkipFront = 0;
    bool isSkipSide = 0;

    //Initialize and load classifier model temporary , should be made command line arguments
#ifdef WIN32
    QString HOGParam_file_sde = "C:/Users/27rut/BIAS/BIASJAABA/src/plugin/jaaba_plugin/json_files/HOGparam.json";
    QString HOFParam_file_sde = "C:/Users/27rut/BIAS/BIASJAABA/src/plugin/jaaba_plugin/json_files/HOFparam.json";
    QString CropParam_file_sde = "C:/Users/27rut/BIAS/BIASJAABA/src/plugin/jaaba_plugin/json_files/Cropsde_param.json";

    QString HOGParam_file_frt = "C:/Users/27rut/BIAS/BIASJAABA/src/plugin/jaaba_plugin/json_files/HOGparam.json";
    QString HOFParam_file_frt = "C:/Users/27rut/BIAS/BIASJAABA/src/plugin/jaaba_plugin/json_files/HOFparam.json";
    QString CropParam_file_frt = "C:/Users/27rut/BIAS/BIASJAABA/src/plugin/jaaba_plugin/json_files/Cropfrt_param.json";

    // Video Capture
    QString vidFile[nviews] = { "C:/Users/27rut/BIAS/BIASJAABA_movies/movie_sde.avi",
                               "C:/Users/27rut/BIAS/BIASJAABA_movies/movie_frt.avi" };
    videoBackend vid_sde(vidFile[0]);
    videoBackend vid_frt(vidFile[1]);

    //Initialize and load classifier model
    QString classifier_file = "C:/Users/27rut/BIAS/BIASJAABA/src/plugin/jaaba_plugin/json_files/multiclassifier.mat";
#endif    

    // Output dir filepath
    string output_dir = "C:/Users/27rut/BIAS/misc/spinnaker_toy_example/bias_demo_example/cam2sys_lat/single_camera/";

    //HOG HOF Params
    HOGHOF* feat_side = new HOGHOF();
    HOGHOF* feat_frt = new HOGHOF();
    beh_class* classifier = new beh_class(classifier_file);
    Params param_sde;
    Params param_frt;

    param_sde = { HOGParam_file_sde , HOFParam_file_sde, CropParam_file_sde };
    param_frt = { HOGParam_file_frt , HOFParam_file_frt, CropParam_file_frt };
    
    // spinnaker camera configuration variables
    spinError err = SPINNAKER_ERR_SUCCESS;
    spinError errReturn = SPINNAKER_ERR_SUCCESS;
    spinSystem hSystem = NULL;
    spinCameraList hCameraList = NULL;
    SpinUtils spin_handle;

    // Multithreading for multi camera system
    //QPointer<QThreadPool> threadPoolPtr_ = new QThreadPool();

    // performance benchmark variables.
    std::vector<float> ts_nidaq(numFrames, 0.0);
    std::vector<int64_t>ts_pc(numFrames, 0);
    std::vector<int64_t>ts_pc_side(numFrames, 0);
    std::vector<int64_t>ts_pc_front(numFrames, 0);
    std::vector<int64_t>delay(10, 0);
    bias::NIDAQUtils* nidaq_task; // nidaq timing object 
    bias::GetTime* gettime; // pc time stamps object
    uInt32 read_buffer, read_ondemand;
    uInt32 read_start=0, read_end=0;
    int64_t start_delay = 0, end_delay = 0 ,avgWaitThres = 2500;
    int64_t start_process=0, end_process=0;
    bool isNIDAQ= 0;

    // Print out current library version
    spinLibraryVersion hLibraryVersion;
    spinSystemGetLibraryVersion(hSystem, &hLibraryVersion);
    printf(
        "Spinnaker library version: %d.%d.%d.%d\n\n",
        hLibraryVersion.major,
        hLibraryVersion.minor,
        hLibraryVersion.type,
        hLibraryVersion.build);


    // Retrieve list of cameras from the system
    bool hasValidInput = false;
    err = spin_handle.setupSystem(hSystem, hCameraList);
    if (err != SPINNAKER_ERR_SUCCESS) {

        printf("Aborting with error %d...\n\n", err);
        return err;
    }

    // Retrieve number of cameras
    size_t numCameras = 0;
    spinCamera hCamera = NULL;
    spinNodeMapHandle hNodeMapTLDevice = NULL;
    spinNodeMapHandle hNodeMap = NULL;
    
    int imageCnt = 0;    
    vector<vector<float>>score_cls(6,vector<float>(numFrames, 0.0));

    /*err = spinCameraListGetSize(hCameraList, &numCameras);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve number of cameras. Aborting with error %d...\n\n", err);
        return err;
    }*/

    if (isNIDAQ) {

        nidaq_task = new NIDAQUtils();
        if (nidaq_task != nullptr) {

            nidaq_task->startTasks();
        }
    }
    else {
        gettime = new bias::GetTime();

    }
    // initialize camera, nidaq, HOGHOF params 
    if (numCameras != 0) {

        std::cout << "Camera Initialized" << std::endl;
        hasValidInput = true;
        err = spinCameraListGet(hCameraList, 0, &hCamera);
        spin_handle.initialize_camera(hCamera, hNodeMap, hNodeMapTLDevice);

        //hNOdeMap.GetNode();

        feat_side->initialize_params(param_sde);
        feat_frt->initialize_params(param_frt);               
        int height = feat_side->HOFParams.input.h;
        int width = feat_side->HOFParams.input.w;
        feat_side->initializeHOGHOF(width, height, numFrames);
        feat_frt->initializeHOGHOF(width, height, numFrames);
        std::cout << "hoghof initalized " << std::endl;

        //have to load ImageParams from width and height information
        //printf("Number of cameras detected: %u\n\n", (unsigned int)numCameras);

    }else if (!vidFile->isEmpty()) {

        // Vid params
        std::cout << "Video File Input" << std::endl;
        hasValidInput = true;
        feat_side->initialize_vidparams(vid_sde, param_sde);
        feat_frt->initialize_vidparams(vid_frt , param_frt);
        feat_side->readVidFrames(vid_sde);
        feat_frt->readVidFrames(vid_frt);
        
        //HOGHOF params
        int height = feat_side->HOFParams.input.h;
        int width = feat_side->HOFParams.input.w;

        feat_side->initializeHOGHOF(width, height, numFrames);
        feat_frt->initializeHOGHOF(width, height, numFrames);
        classifier->translate_mat2C(&feat_side->hog_shape, &feat_frt->hog_shape);

    }
    
    bool isTriggered = false;
    // Finish if there are no cameras
    while (imageCnt < numFrames) {

        //std::cout << "frameCount: " << imageCnt <<  std::endl;
        isSkipFront = 0;
        isSkipSide = 0;

        if (hasValidInput && numCameras != 0)
        {
            spinImage hResultImage = NULL;
            cv::Mat image;

            if (!isTriggered && nidaq_task != nullptr) {

                printf("Started NIDAQ Trigger Signal");
                nidaq_task->start_trigger_signal();
                isTriggered = true;
            }

            err = spin_handle.getFrame_camera(hCamera, hResultImage);                

            if (err != SPINNAKER_ERR_SUCCESS)
            {
                printf("skip frame...\n\n", err);
            }

            spinError err = SPINNAKER_ERR_SUCCESS;
            spinImage hSpinImageConv = nullptr;

            err = spinImageCreateEmpty(&hSpinImageConv);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                printf("empty frame...\n\n", err);
            }

            spinPixelFormatEnums origPixelFormat = getImagePixelFormat_spin(hResultImage);
            spinPixelFormatEnums convPixelFormat = getSuitablePixelFormat(origPixelFormat);

            err = spinImageConvert(hResultImage, convPixelFormat, hSpinImageConv);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                printf("Unable to convert image...\n\n", err);
            }

            ImageInfo_spin imageInfo = getImageInfo_spin(hSpinImageConv);

            int opencvPixelFormat = getCompatibleOpencvFormat(convPixelFormat);

            cv::Mat imageTmp = cv::Mat(
                imageInfo.rows + imageInfo.ypad,
                imageInfo.cols + imageInfo.xpad,
                opencvPixelFormat,
                imageInfo.dataPtr,
                imageInfo.stride
            );

            imageTmp.copyTo(image);
            
            image.convertTo(image, CV_32FC1);
            image = image / 255;
            feat_side->img.buf = image.data;
            
            //printf("%d\n", imageCnt);
            if (nidaq_task != nullptr && isNIDAQ) {

                DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task->taskHandle_trigger_in, 10.0, &read_buffer, NULL));
                DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task->taskHandle_grab_in, 10.0, &read_ondemand, NULL));
                ts_nidaq[imageCnt] = static_cast<float>((read_ondemand - read_buffer)*0.02);

            }
          
            // ----------------------------------------------------------------------------

            if (!destroySpinImage(hSpinImageConv))
            {
                printf("Unable to release image. Non-fatal error %d...\n\n", err);
            }

            err = spinImageRelease(hResultImage);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                printf("Unable to release image. Non-fatal error %d...\n\n", err);
            }
            //std::cout << imageCnt << std::endl;

        }
        else if (hasValidInput && !vidFile->isEmpty()) {

            cv::Mat curImg_side;
            cv::Mat curImg_frt;
            cv::Mat greySide;
            cv::Mat greyFront;

            /*if (isNIDAQ && !isTriggered && nidaq_task != nullptr) {

                printf("Started NIDAQ Trigger Signal");
                nidaq_task->start_trigger_signal();
                isTriggered = true;
            }

            if (isNIDAQ)
            {
                DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task->taskHandle_grab_in, 10.0, &read_start, NULL));
                DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task->taskHandle_trigger_in, 10.0, &read_buffer, NULL));
               
            }
            else {
                start_process = gettime->getPCtime();
            }*/

            curImg_side = feat_side->vid_frames[imageCnt];
            feat_side->preprocess_vidFrame(vid_sde, curImg_side);
            curImg_frt = feat_frt->vid_frames[imageCnt];
            feat_frt->preprocess_vidFrame(vid_frt, curImg_frt);

            /*if(!isNIDAQ){

                //start_delay = gettime->getPCtime();
                //end_delay = start_delay;
                //while ((end_delay - start_delay) < avgWaitThres)
                //{
                //    end_delay = gettime->getPCtime();
                //}
                //write_time<int64_t>("./temp.csv", 1, delay);
                //write_time<int64_t>("./temp1.csv", 1, delay);
                end_process = gettime->getPCtime();
            }
            else if (isNIDAQ) {
                DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task->taskHandle_grab_in, 10.0, &read_end, NULL));
            }*/
            
            /*if (isNIDAQ)
                ts_nidaq[imageCnt] = (read_end - read_start)*0.02;
            else
                ts_pc[imageCnt] = end_process - start_process;*/

            start_process = gettime->getPCtime();
#if isSkip            
            if (!skipframes_view1.empty())
            {
                if (imageCnt < skipframes_view1.top())
                {

                    feat_side->process_vidFrame(imageCnt);

                }
                else if (imageCnt >= (skipframes_view1.top() + frameSkip)) {

                    feat_side->setLastInput();
                    feat_side->process_vidFrame(imageCnt);

                }
                else {

                    std::cout << "Framecount side view skipped: " << imageCnt << std::endl;
                    isSkipSide = 1;
                    classifier->score_side = { 0.0,0.0,0.0,0.0,0.0,0.0 };
                    classifier->score_front = { 0.0,0.0,0.0,0.0,0.0,0.0 };
                    classifier->addScores(classifier->score_side, classifier->score_front);
                    classifier->write_score("./lift_classifier.csv", imageCnt, classifier->score[0]);
                    imageCnt++;
                    continue;
                }
                if (imageCnt == (skipframes_view1.top() + frameSkip))
                    skipframes_view1.pop();
            }
#else            
            //feat_side->getvid_frame(vid_sde);
            feat_side->process_vidFrame(imageCnt);
#endif      
            end_process = gettime->getPCtime();
            ts_pc_side[imageCnt] = (end_process - start_process);

            start_process = gettime->getPCtime();
#if isSkip
            if (!skipframes_view2.empty())
            {
                if ((imageCnt < skipframes_view2.top()))
                {

                }else if((imageCnt >= (skipframes_view2.top() + frameSkip))) {

                    feat_frt->setLastInput();

                }else {

                    std::cout << "frameCnt: " << imageCnt << std::endl;
                    isSkipFront = 1;
                    classifier->score_side = { 0.0,0.0,0.0,0.0,0.0,0.0 };
                    classifier->score_front = { 0.0,0.0,0.0,0.0,0.0,0.0 };
                    classifier->addScores(classifier->score_side, classifier->score_front);
                    classifier->write_score("./lift_classifier.csv", imageCnt, classifier->score[0]);
                    imageCnt++;
                    continue;

                }
                if (imageCnt == (skipframes_view2.top() + frameSkip))
                    skipframes_view2.pop();
            }
#else
            //feat_frt->getvid_frame(vid_frt);
            feat_frt->process_vidFrame(imageCnt);
#endif
            

        }else {

            if (numCameras == 0) {

                err = spin_handle.ReleaseSystem(hSystem, hCameraList);
                if (err != SPINNAKER_ERR_SUCCESS)
                {
                    printf("Unable to release cameras. Aborting with error %d...\n\n", err);
                    return err;
                }
                printf("Not enough cameras!\n");
                break;
            }

            if (vidFile->isEmpty()) {

                break;
            }

        }

        if (imageCnt > 0) {

            /*if (!isSkipSide)
            {
                classifier->boost_classify_side(classifier->score_side,
                    feat_side->hog_out, feat_side->hof_out,
                    &feat_side->hog_shape, &feat_frt->hof_shape, classifier->nframes,
                    classifier->model);

            } else {

                classifier->score_side = {0.0,0.0,0.0,0.0,0.0,0.0};
            }*/

            if (!isSkipFront)
            {
                classifier->boost_classify_front(classifier->score_front,
                    feat_frt->hog_out, feat_frt->hof_out,
                    &feat_side->hog_shape, &feat_frt->hof_shape, classifier->nframes,
                    classifier->model);

            } else {

                std::cout << "score: " << imageCnt << std::endl;
                classifier->score_front = {0.0,0.0,0.0,0.0,0.0,0.0};
            }
            end_process = gettime->getPCtime();
            ts_pc_front[imageCnt] = (end_process - start_process);

            //classifier->addScores(classifier->score_side, classifier->score_front);
            //classifier->write_score(output_dir + "/lift_classifier.csv", imageCnt, classifier->score[0]);

        }
        //end_process = gettime->getPCtime();
        //ts_pc[imageCnt] = (end_process - start_process);
        
        if (isNIDAQ && imageCnt == numFrames-1){
            write_time<float>(output_dir + "/cam2sys_latency.csv", numFrames, ts_nidaq);
            break;
        }
        if(imageCnt == numFrames - 1) {
            write_time<int64_t>(output_dir + "/ts_pc_latency_vidread_processjaaba_side_.csv", numFrames, ts_pc_front);
            write_time<int64_t>(output_dir + "/ts_pc_latency_vidread_processjaaba_front_.csv", numFrames, ts_pc_side);
            break;
        }
        imageCnt++;

    }

    /*createh5("./hoghof", ".h5", 2498,
                 2400, 2400,
                 1600, 1600,
                 feat_side->hog_out, feat_frt->hog_out,
                 feat_side->hof_out, feat_frt->hof_out);*/

    if (numCameras != 0) {

        // End Acquisition
        err = spinCameraEndAcquisition(hCamera);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to end acquisition. Non-fatal error %d...\n\n", err);
        }

        spin_handle.deInitialize_camera(hCamera, hNodeMap);

        // Release camera
        err = spinCameraRelease(hCamera);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            errReturn = err;
        }

        if (isNIDAQ) {
            nidaq_task->Cleanup();
        }
        else {
            delete gettime;
        }
    }
        
    
    /*err = spin_handle.ReleaseSystem(hSystem, hCameraList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to release cameras. Aborting with error %d...\n\n", err);
        return err;
    }*/
   
}

 
