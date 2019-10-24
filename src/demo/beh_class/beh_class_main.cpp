#include "HOGHOF.hpp"
#include "beh_class.hpp"
#include <vector>
#include "timer.h"
#include <fstream>
#include <iostream>
#include <QDebug>

/*void parse_args(int nargs, char** arg_inputs, std::string& view) {

    std::string v = "side";

    // parse all the arguments from the command line
    while((opt = getopt(nargs,arg_inputs,"v:")) != -1) {

        options.insert(opt);

        switch(opt){
iiiii
            case 'v': g = optarg;
                break;
            default: std::cout << "Missing arguments:<help>"
                exit(-1);
        }

    }

}*/


int main(int argc, char* argv[]) {

    //Input files
    //cudaSetDevice(1);
    HOGHOF feat_side;
    feat_side.HOGParam_file = "/groups/branson/home/patilr/BIAS/BIASJAABA/src/demo/beh_class/json_files/HOGparam.json";
    feat_side.HOFParam_file = "/groups/branson/home/patilr/BIAS/BIASJAABA/src/demo/beh_class/json_files/HOFparam.json";
    feat_side.CropParam_file = "/groups/branson/home/patilr/BIAS/BIASJAABA/src/demo/beh_class/json_files/Cropsde_param.json";
                           
    HOGHOF feat_frt;
    feat_frt.HOGParam_file = "/groups/branson/home/patilr/BIAS/BIASJAABA/src/demo/beh_class/json_files/HOGparam.json";
    feat_frt.HOFParam_file = "/groups/branson/home/patilr/BIAS/BIASJAABA/src/demo/beh_class/json_files/HOFparam.json"; 
    feat_frt.CropParam_file = "/groups/branson/home/patilr/BIAS/BIASJAABA/src/demo/beh_class/json_files/Cropfrt_param.json";

    // Video Capture
    int nviews = 2;
    QString vidFile[nviews] = {//"/groups/branson/home/patilr/bias_video_cam_0_date_2019_06_13_time_14_45_56_v001/image_%d.bmp"}; 
                               "/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_sde.avi",
                               "/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_frt.avi"};

    //Initialize and load classifier model
    beh_class classifier;
    classifier.classifier_file = "/nrs/branson/jab_experiments/M277PSAMBpn/FinalJAB/cuda_jabs/classifier_Lift.mat";
    classifier.allocate_model();
    classifier.loadclassifier_model();

 
    //compute Featues 
    //cudaSetDevice(0); 
    GpuTimer timer1;
    timer1.Start();
    feat_side.genFeatures(vidFile[0], feat_side.CropParam_file);
    timer1.Stop();

    //cudaSetDevice(1);
    GpuTimer timer2 ;
    timer2.Start();
    feat_frt.genFeatures(vidFile[1], feat_frt.CropParam_file);
    timer2.Stop();
    std::cout << timer1.Elapsed()/1000 << std::endl;
    std::cout << timer2.Elapsed()/1000 << std::endl;

    createh5("./hoghof", ".h5", 2498, 
             2400, 2400,
             1600, 1600,
             feat_side.hog_out, feat_frt.hog_out,
             feat_side.hof_out, feat_frt.hof_out); 

    //predict scores
    classifier.translate_mat2C(&feat_side.hog_shape, &feat_frt.hof_shape);
    classifier.scores.resize(classifier.nframes,0.0);
    
    for(int frame_id =1;frame_id < classifier.nframes; frame_id++) {

        //std::cout << frame_id << std::endl;
        classifier.boost_classify(classifier.scores, feat_side.hog_out, feat_frt.hog_out, feat_side.hof_out, feat_frt.hof_out,
                                  &feat_side.hog_shape, &feat_frt.hof_shape, classifier.nframes, frame_id-1, classifier.model);
        
    }

    std::string out_scores = "test_Grab.h5";
    H5::H5File file_scr(out_scores.c_str(), H5F_ACC_TRUNC);
    create_dataset(file_scr,"scores", classifier.scores, 1, classifier.nframes);
    file_scr.close();
    std::cout << "hi" << std::endl;

}


 
