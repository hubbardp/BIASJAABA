#include "HOGHOF.hpp"
#include "beh_class.hpp"
#include "spin_utils.hpp"
#include "utils_spin.hpp"
#include "SpinnakerC.h"

//#include <vector>
//#include "timer.h"
//#include <fstream>
//#include <iostream>
#include <QDebug>

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
void write_time(std::string filename, int framenum, std::vector<std::vector<T>> timeVec)
{

    std::ofstream x_out;
    x_out.open(filename.c_str(), std::ios_base::app);

    for (int frame_id = 0; frame_id < framenum - 1; frame_id++)
    {

        x_out << timeVec[frame_id][1] << "\n";
    }

    x_out.close();

}

int main(int argc, char* argv[]) {

    //nviews - temp should be command line arg
    const int nviews = 2;

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
    //HOG HOF Params
    
    HOGHOF* feat_side = new HOGHOF();
    HOGHOF* feat_frt = new HOGHOF();
    beh_class* classifier = new beh_class(classifier_file);
    Params param_sde;
    Params param_frt;

    param_sde = { HOGParam_file_sde , HOFParam_file_sde, CropParam_file_sde };
    param_frt = { HOGParam_file_frt , HOFParam_file_frt, CropParam_file_frt };
    
    // Retrieve singleton reference to system
    
    spinError err = SPINNAKER_ERR_SUCCESS;
    spinError errReturn = SPINNAKER_ERR_SUCCESS;
    spinSystem hSystem = NULL;
    spinCameraList hCameraList = NULL;
    SpinUtils spin_handle;
    std::vector<std::vector<float>> timeStamps(500000, std::vector<float>(2, 0.0));
    bias::NIDAQUtils* nidaq_task = new NIDAQUtils();
    uInt32 read_buffer, read_ondemand;

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
    int numFrames = 5000;// classifier->nframes;
    vector<vector<float>>score_cls(6,vector<float>(numFrames, 0.0));

    err = spinCameraListGetSize(hCameraList, &numCameras);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve number of cameras. Aborting with error %d...\n\n", err);
        return err;
    }

    if (numCameras != 0) {

        hasValidInput = true;
        err = spinCameraListGet(hCameraList, 0, &hCamera);
        spin_handle.initialize_camera(hCamera, hNodeMap, hNodeMapTLDevice);

        //hNOdeMap.GetNode();

        if (nidaq_task != nullptr) {

            nidaq_task->startTasks();
            
        }

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

        hasValidInput = true;
        feat_side->initialize_vidparams(vid_sde, param_sde);
        feat_frt->initialize_vidparams(vid_frt , param_frt);
        int height = feat_side->HOFParams.input.h;
        int width = feat_side->HOFParams.input.w;
        feat_side->initializeHOGHOF(width, height, numFrames);
        feat_frt->initializeHOGHOF(width, height, numFrames);
        classifier->translate_mat2C(&feat_side->hog_shape, &feat_frt->hog_shape);

    }else {



    }
    
    bool isTriggered = false;
    // Finish if there are no cameras
    while (imageCnt < numFrames) {

        if (hasValidInput && numCameras != 0)
        {
            spinImage hResultImage = NULL;
            cv::Mat image;

            if (!isTriggered && nidaq_task != nullptr) {

                printf("Started tasks");
                nidaq_task->start_trigger_signal();
                isTriggered = true;
            }

            err = spin_handle.getFrame_camera(hCamera, hResultImage,
                nidaq_task, timeStamps, imageCnt);

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
            feat_side->process_camFrame();
            //feat_frt->img.buf = image.data;
            //feat_frt->process_camFrame();
            
            //printf("%d\n", imageCnt);
            if (nidaq_task != nullptr) {

                DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task->taskHandle_trigger_in, 10.0, &read_buffer, NULL));
                DAQmxErrChk(DAQmxReadCounterScalarU32(nidaq_task->taskHandle_grab_in, 10.0, &read_ondemand, NULL));
                timeStamps[imageCnt][1] = static_cast<float>((read_ondemand - read_buffer)*0.02);

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

        }else if (hasValidInput && !vidFile->isEmpty()) {

            feat_side->getvid_frame(vid_sde);
            feat_side->process_vidFrame(imageCnt);
            feat_frt->getvid_frame(vid_frt);
            feat_frt->process_vidFrame(imageCnt);

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

        /*if (imageCnt > 0) {

            classifier->boost_classify(classifier->score, feat_side->hog_out,
                feat_frt->hog_out, feat_side->hof_out,
                feat_frt->hof_out, &feat_side->hog_shape,
                &feat_frt->hof_shape, classifier->nframes, classifier->model);
            //classifier->write_score("./lift_classifier.csv", imageCnt, classifier->score[0]);
        }*/

        
        if (imageCnt == 4999){
            write_time<float>("./cam2sys_latency.csv", 4999, timeStamps);
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

        nidaq_task->Cleanup();
    }
        
    
    err = spin_handle.ReleaseSystem(hSystem, hCameraList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to release cameras. Aborting with error %d...\n\n", err);
        return err;
    }

        
}

 
