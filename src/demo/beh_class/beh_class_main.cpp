#include "HOGHOF.hpp"
#include "video_utils.hpp"
#include "logger.h"
#include "image_fcns.h"
#include <iostream>

//CONSTANTS
QString HOGParam_file = "/groups/branson/home/patilr/BIAS/BIASJAABA/src/demo/beh_class/json_files/HOGparam.json";
QString HOFParam_file = "/groups/branson/home/patilr/BIAS/BIASJAABA/src/demo/beh_class/json_files/HOFparam.json";
QString CropParam_file = "/groups/branson/home/patilr/BIAS/BIASJAABA/src/demo/beh_class/json_files/Cropparam.json";
QString classifier_file = "/nrs/branson/jab_experiments/M277PSAMBpn/FinalJAB/cuda_jabs/classifier_Lift.mat";

int main(int argc, char* argv[]) {

    //Input files
    HOGHOF feat;
    feat.HOGParam_file = "/groups/branson/home/patilr/BIAS/BIASJAABA/src/demo/beh_class/json_files/HOGparam.json";
    feat.HOFParam_file = "/groups/branson/home/patilr/BIAS/BIASJAABA/src/demo/beh_class/json_files/HOFparam.json";
    feat.CropParam_file = "/groups/branson/home/patilr/BIAS/BIASJAABA/src/demo/beh_class/json_files/Cropparam.json";
    //classifier_file = "/nrs/branson/jab_experiments/M277PSAMBpn/FinalJAB/cuda_jabs/feat_Lift.mat";


    // Video Capture
    QString vidFile = "/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_sde.avi" ;
    std::string hogout_file = "/groups/branson/home/patilr/BIAS/BIASJAABA/src/demo/beh_class/features/hog_";
    std:: string hofout_file = "/groups/branson/home/patilr/BIAS/BIASJAABA/src/demo/beh_class/features/hof_";
    bias::videoBackend vid(vidFile) ;
    cv::VideoCapture capture = vid.videoCapObject(vid);

    std::string fname = vidFile.toStdString();
    int num_frames = vid.getNumFrames(capture);
    int height = vid.getImageHeight(capture);
    int width =  vid.getImageWidth(capture);
    float fps = capture.get(cv::CAP_PROP_FPS);
    std::cout << "frame rate" << fps << std::endl;

    // Parse HOG/HOF/Crop Params
    feat.loadHOGParams();

    feat.loadHOFParams();
    feat.HOFParams.input.w = width;
    feat.HOFParams.input.h = height;
    feat.HOFParams.input.pitch = width;

    feat.loadImageParams(width,height);
    feat.loadCropParams();
    //feat.loadfeatmodel(,feat_file);

    // create input HOGContext / HOFConntext
    struct HOGContext hog_ctx = HOGInitialize(logger, feat.HOGParams, width, height, feat.Cropparams);
    struct HOFContext hof_ctx = HOFInitialize(logger, feat.HOFParams, feat.Cropparams);

    //allocate output HOG/HOF per frame 
    size_t hog_outputbytes = HOGOutputByteCount(&hog_ctx);
    size_t hof_outputbytes = HOFOutputByteCount(&hof_ctx);
    struct HOGFeatureDims hog_shape;
    HOGOutputShape(&hog_ctx, &hog_shape);
    struct HOGFeatureDims hof_shape;
    HOFOutputShape(&hof_ctx, &hof_shape);
    //feat.allocateHOGoutput(feat.hog_out, &hog_ctx);
    //feat.allocateHOFoutput(feat.hof_out, &hof_ctx);
    feat.hof_out  = (float*)malloc(hof_outputbytes);
    feat.hog_out = (float*)malloc(hog_outputbytes);

    cv::Mat cur_frame;
    int frame = 0;
    while(frame < num_frames) {

        //capture frame and convert to grey
        cur_frame = vid.getImage(capture);

        //convert to Float and normalize
        vid.convertImagetoFloat(cur_frame);
        feat.img.buf = cur_frame.ptr<float>(0);

        //Compute HOG/HOF         
        HOGCompute(&hog_ctx, feat.img);
        HOFCompute(&hof_ctx, feat.img.buf, hof_f32);

        //Copy results
        HOFOutputCopy(&hof_ctx, feat.hof_out, hof_outputbytes);
        HOGOutputCopy(&hog_ctx, feat.hog_out, hog_outputbytes);

        std::string hog_file = hogout_file + std::to_string(frame) + ".csv";
        write_histoutput(hog_file, feat.hog_out, hog_shape.x, hog_shape.y, hog_shape.bin);
        if(frame > 0) {
            std::string hof_file = hofout_file + std::to_string(frame-1) + ".csv";
            write_histoutput(hof_file, feat.hof_out, hof_shape.x, hof_shape.y, hof_shape.bin);
        }
        frame = frame + 1;

    }

    vid.releaseCapObject(capture);
}

