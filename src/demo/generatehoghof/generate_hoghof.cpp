#include "win_time.hpp"
#include "HOGHOF.hpp"
#include "utils_spin.hpp"
#include "video_utils.hpp"
#include "jaaba_utils.hpp"
#include "cuda_runtime_api.h"
#include "parser.hpp"
#include "getopt.h"

#include <opencv2/core/core.hpp>
#include <iostream>
#include <QApplication>

using namespace std;

namespace bias {

    // parser defined here because defining in parser giving link errors. Need to fix in future
    void parser(int argc, char *argv[], CmdLineParams& cmdlineparams) {
        int opt;

        // place ':' in the beginning of the string so that program can 
        //tell between '?' and ':' 
        while ((opt = getopt(argc, argv, ":o:i:s:c:l:v:f:k:w:n:d:")) != -1)
        {
            switch (opt)
            {
            case 'o':
                cmdlineparams.output_dir = optarg;
                break;
            case 'i':
                cmdlineparams.isVideo = stoi(optarg);
                break;
            case 's':
                cmdlineparams.saveFeat = stoi(optarg);
                break;
            case 'c':
                cmdlineparams.compute_jaaba = stoi(optarg);
                break;
            case 'l':
                cmdlineparams.classify_scores = stoi(optarg);
                break;
            case 'v':
                cmdlineparams.visualize = stoi(optarg);
                break;
            case 'f':
                cmdlineparams.numframes = stoi(optarg);
                break;
            case 'k':
                cmdlineparams.isSkip = stoi(optarg);
                break;
            case 'w':
                cmdlineparams.wait_thres = stoi(optarg);
                break;
            case 'n':
                cmdlineparams.window_size = stoi(optarg);
                break;
            case 'd':
                cmdlineparams.debug = stoi(optarg);
                break;
            case ':':
                printf("Required argument %c", opt);
                break;
            case '?':
                //printf("unknown option : %c\n", optopt);
                break;
            }
        }

        // optind is for the extra arguments
        // which are not parsed
        //for (; optind < argc; optind++) {
        //    printf(“extra arguments : %s\n”, argv[optind]);
        //}
    }
}

int main(int argc, char* argv[]) {

    QApplication app(argc, argv);
    const int nviews = 2;
    int numFrames;//frames to process
    bias::CmdLineParams cmdlineparams;
    bias::parser(argc, argv, cmdlineparams);
    cudaError err;

    string input_dir_path = cmdlineparams.output_dir;  //"C:/Users/27rut/BIAS/mouse_videos_0502/";
    string output_dir_path = cmdlineparams.output_dir;  //"C:/Users/27rut/BIAS/mouse_videos_0502/";
    string HOGParam_file_sde = input_dir_path + "json_files/HOGParam.json";
    string HOFParam_file_sde = input_dir_path + "json_files/HOFparam.json";
    string CropParam_file_sde = input_dir_path + "json_files/Cropsde_param.json";

    string HOGParam_file_frt = input_dir_path + "json_files/HOGparam.json";
    string HOFParam_file_frt = input_dir_path + "json_files/HOFparam.json";
    string CropParam_file_frt = input_dir_path + "json_files/Cropfrt_param.json";
    int window_size = cmdlineparams.window_size;
    int saveFeatures = cmdlineparams.saveFeat;
    int nframestocompute = cmdlineparams.numframes;
    int debug = cmdlineparams.debug;

    std::cout << "Command line arguments\n"
        << "Output Dir: " << cmdlineparams.output_dir
        << "\nis Video:" << cmdlineparams.isVideo
        << "\nsave features:" << cmdlineparams.saveFeat
        << "\ncompute jaaba: " << cmdlineparams.compute_jaaba
        << "\nclassify scores: " << cmdlineparams.classify_scores
        << "\nvisualize " << cmdlineparams.visualize
        << "\nnumframes " << cmdlineparams.numframes
        << "\nisskip " << cmdlineparams.isSkip
        << "\n wait threshold" << cmdlineparams.wait_thres // wait time between jaaba views for computing a score
        << "\ nwindow_size " << cmdlineparams.window_size // averaging window size hoghof features
        << "\n DEBUG " << cmdlineparams.debug
        << "\n com port" << cmdlineparams.comport
        << std::endl;

    // Video Capture
    QString vidFile[nviews] = { "movie_sde.avi" ,
                                "movie_frt.avi" };
    std::cout << "Movie filename" << input_dir_path <<
        std::endl;
    bias::videoBackend vid_sde(QString::fromStdString(input_dir_path) + vidFile[0]);
    bias::videoBackend vid_frt(QString::fromStdString(input_dir_path) + vidFile[1]);
    cv::VideoCapture cap_obj_sde = vid_sde.videoCapObject();
    cv::VideoCapture cap_obj_frt = vid_frt.videoCapObject();
    numFrames = vid_sde.getNumFrames(cap_obj_sde);
    int width = vid_sde.getImageWidth(cap_obj_sde);
    int height = vid_sde.getImageHeight(cap_obj_sde);
    std::cout << "Video has " << numFrames << " frames" << std::endl;
    std::cout << "Video height" << height << std::endl;
    std::cout << "Video width" << width << std::endl;
    std::cout << "window size" << window_size << std::endl;

    //HOG HOF Params
    bias::HOGHOF* feat_side = new bias::HOGHOF();
    bias::HOGHOF* feat_front = new bias::HOGHOF();
    //beh_class* classifier = new beh_class(classifier_file);

    feat_side->HOGParam_file = HOGParam_file_sde;
    feat_side->HOFParam_file = HOFParam_file_sde;
    feat_side->CropParam_file = CropParam_file_sde;

    feat_front->HOGParam_file = HOGParam_file_frt;
    feat_front->HOFParam_file = HOFParam_file_frt;
    feat_front->CropParam_file = CropParam_file_frt;

    int feat_dim_side = 0;
    int feat_dim_front = 0;

    float* feats_hog_out = nullptr;
    float* feats_hof_out = nullptr;
    float* featf_hog_out = nullptr;
    float* featf_hof_out = nullptr;

    feat_side->initialize_HOGHOFParams();
    feat_front->initialize_HOGHOFParams();

    // frame read buffer
    std::vector<cv::Mat>vid_frames_sde;
    std::vector<cv::Mat>vid_frames_frt;
    vid_sde.readVidFrames(cap_obj_sde, vid_frames_sde);
    vid_frt.readVidFrames(cap_obj_frt, vid_frames_frt);

    // test debug 
    int cuda_device_1 = 1;
    int cuda_device_2 = 0;

    err = cudaSetDevice(cuda_device_1);
    if (err != cudaSuccess)
    {
        std::cout << "Error setting gpu in init " << std::endl;
        exit(-1);
    }
    feat_side->initHOGHOF(height, width);

    //err = cudaSetDevice(cuda_device_2);
    if (err != cudaSuccess)
    {
        std::cout << "Error setting gpu in init " << std::endl;
        exit(-1);
    }
    feat_front->initHOGHOF(height, width);

    feat_dim_side = feat_side->hog_shape.x* feat_side->hog_shape.y*feat_side->hog_shape.bin;
    feat_dim_front = feat_front->hog_shape.x* feat_front->hog_shape.y*feat_front->hog_shape.bin;
    feats_hog_out = new float[nframestocompute * feat_dim_side];
    featf_hog_out = new float[nframestocompute * feat_dim_front];
    feats_hof_out = new float[nframestocompute * feat_dim_side];
    featf_hof_out = new float[nframestocompute * feat_dim_front];
    printf("Feature dims side and front - %d-%d", feat_dim_side, feat_dim_front);

    int nRuns = 3;
    int cur_run = 0;
    int imageCnt = 0;
    int start_time = 0, end_time = 0;
    string jaaba_process_time_file;  
    vector<int64_t> jaaba_process_time;
    jaaba_process_time.resize(nframestocompute);
    bias::GetTime* gettime_ = new bias::GetTime();

    while (cur_run < nRuns)
    {
        imageCnt = 0;
        if(debug)
            jaaba_process_time_file = input_dir_path + "jaaba_process_time_" + to_string(cur_run) + ".csv";
        while (imageCnt < nframestocompute) {

            start_time = gettime_->getPCtime();

            if (!vidFile->isEmpty()) {
                cv::Mat curImg_side;
                cv::Mat curImg_frt;
                cv::Mat greySide;
                cv::Mat greyFront;

                curImg_side = vid_frames_sde[imageCnt%numFrames];
                vid_sde.preprocess_vidFrame(curImg_side);
                feat_side->img.buf = curImg_side.ptr<float>(0);
                curImg_frt = vid_frames_frt[imageCnt%numFrames];
                vid_frt.preprocess_vidFrame(curImg_frt);
                feat_front->img.buf = curImg_frt.ptr<float>(0);

                //err = cudaSetDevice(cuda_device_1);
                if (err != cudaSuccess)
                {
                    std::cout << "Error setting gpu in process " << std::endl;
                    exit(-1);
                }
                feat_side->genFeatures(imageCnt);

                //err = cudaSetDevice(cuda_device_2);
                if (err != cudaSuccess)
                {
                    std::cout << "Error setting gpu in process " << std::endl;
                    exit(-1);
                }
                feat_front->genFeatures(imageCnt);


                if (saveFeatures) {
                   
                    bias::saveFeatures(output_dir_path + "hoghof_side_biasjaaba_offline.csv",
                        feat_side->hog_out, feat_side->hof_out,
                        feat_dim_side, feat_dim_side);
                    bias::saveFeatures(output_dir_path + "hoghof_front_biasjaaba_offline.csv",
                        feat_front->hog_out, feat_front->hof_out,
                        feat_dim_front, feat_dim_front);
                }
                feat_side->averageWindowFeatures(window_size, imageCnt, 0);
                feat_front->averageWindowFeatures(window_size, imageCnt, 0);

                if (saveFeatures) {
                    
                    bias::saveFeatures(output_dir_path + "hoghof_avg_side_biasjaaba_offline.csv",
                        feat_side->hog_out_avg, feat_side->hof_out_avg,
                        feat_dim_side, feat_dim_side);
                    bias::saveFeatures(output_dir_path + "hoghof_avg_front_biasjaaba_offline.csv",
                        feat_front->hog_out_avg, feat_front->hof_out_avg,
                        feat_dim_front, feat_dim_front);
                }
                end_time = gettime_->getPCtime();
                if(debug)
                    jaaba_process_time[imageCnt] = (end_time - start_time);
                imageCnt++;
            }
        } // finished reading video from movies 
            
        if (debug) {
            gettime_->write_time_1d<int64_t>(jaaba_process_time_file, nframestocompute, jaaba_process_time);
            std::cout << "Time elapsed to process video frames " << (end_time - start_time)*0.001 << std::endl;
        }
        cur_run++;
    }

    //destroy hog hof ctx on the gpu
    //cudaSetDevice(cuda_device_2);
    if (err != cudaSuccess)
    {
        std::cout << "Error setting gpu in deinit " << std::endl;
        exit(-1);
    }
    HOFTeardown(feat_front->hof_ctx);
    HOGTeardown(feat_front->hog_ctx);

    //if (cuda_device_1 != cuda_device_2)
    //    cudaDeviceReset();

    //cudaSetDevice(cuda_device_1);
    if (err != cudaSuccess)
    {
        std::cout << "Error setting gpu in deinit " << std::endl;
        exit(-1);
    }
    HOFTeardown(feat_side->hof_ctx);
    HOGTeardown(feat_side->hog_ctx);

    cudaDeviceReset();
   
    // delete hog/hof objects on the cpu
    delete feat_front;
    delete feat_side;

    delete[] feats_hog_out;
    delete[] featf_hog_out;
    delete[] feats_hof_out;
    delete[] featf_hof_out;

}
