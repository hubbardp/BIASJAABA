#include "win_time.hpp"
#include "HOGHOF.hpp"
#include "beh_class.hpp"
#include "utils_spin.hpp"
#include "video_utils.hpp"
#include "jaaba_utils.hpp"
#include "cuda_runtime_api.h"
#include "parser.hpp"
#include "getopt.h"

#include <opencv2/core/core.hpp>
#include <iostream>
#include <iomanip>
#include <QApplication>

using namespace std;

namespace bias {

    // parser defined here because defining in parser giving link errors. Need to fix in future
    void parser(int argc, char *argv[], CmdLineParams& cmdlineparams) {
        int opt;

        // place ':' in the beginning of the string so that program can 
        //tell between '?' and ':' 
        while ((opt = getopt(argc, argv, ":o:i:s:c:l:v:k:w:n:d:a:j:")) != -1)
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
            case 'a':
                cmdlineparams.movie_name_suffix = optarg;
                break;
            case 'j':
                cmdlineparams.jaaba_config_path = optarg;
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
        //    printf(�extra arguments : %s\n�, argv[optind]);
        //}
    }

    void write_score_final(std::string file, unsigned int numFrames,
        vector<PredData>& pred_score)
    {
        int num_behs = pred_score[0].score.size();
        std::ofstream x_out;
        x_out.open(file.c_str(), std::ios_base::app);
        std::cout << "once" << std::endl;
        x_out << "Score ts," << "Score ts side," << "Score ts front," << "Score," << " FrameNumber," << "View" << "\n";

        for (unsigned int frm_id = 0; frm_id < numFrames; frm_id++)
        {
            x_out << pred_score[frm_id].score_ts << "," << pred_score[frm_id].score_viewA_ts
                << "," << pred_score[frm_id].score_viewB_ts;
                for (int beh_id = 0; beh_id < num_behs; beh_id++) {
                    x_out << "," << setprecision(6) << pred_score[frm_id].score[beh_id];
                }
                x_out << "," << pred_score[frm_id].frameCount << "," << pred_score[frm_id].view <<
                "\n";
        }
        x_out.close();
    }
}

int main(int argc, char* argv[]) {

    QApplication app(argc, argv);

    const int nviews = 2;
    int numFrames;//frames to process
	int numFrames_frt,numFrames_sde;

    bias::CmdLineParams cmdlineparams;
    bias::parser(argc, argv, cmdlineparams);
    bias::JaabaConfig jab_conf;
    cudaError err;

    string input_dir_path = cmdlineparams.output_dir;
    string output_dir_path = cmdlineparams.output_dir; 
    string jaaba_config_path = cmdlineparams.jaaba_config_path; // path to jaaba config json file
    jab_conf.jaaba_config_file = jaaba_config_path + "/jaaba_config.json";
    std::cout << "Jaaba Config Dir" << jab_conf.jaaba_config_file << std::endl;
    
    //load jaaba config params
    jab_conf.loadJAABAConfigFile();
    string config_file_dir = jab_conf.config_file_dir; // path to the jaaba config files
    string classifier_filename = config_file_dir +  "/" + jab_conf.classifier_filename;
    int window_size = jab_conf.window_size;
    vector<string>classifier_concatenation_order;
    string classifier_order = jab_conf.classifier_concatenation_order;
    jab_conf.convertStringtoVector(classifier_order, classifier_concatenation_order);
    std::cout <<  "view order 0" << classifier_concatenation_order[0] << std::endl;
    std::cout << "view order 1 " << classifier_concatenation_order[1] << std::endl;

    unordered_map<string, unsigned int>::iterator camera_it;
    unordered_map<unsigned int, string>::iterator crop_list_it;
    unordered_map<string, unsigned int> camera_serial_id_list = jab_conf.camera_serial_id;
    unordered_map<unsigned int, string> jab_crop_list = jab_conf.crop_file_list;

    string HOGParam_file_sde = config_file_dir + "/" + jab_conf.hog_file;
    string HOFParam_file_sde = config_file_dir +  "/" + jab_conf.hof_file;
    string HOGParam_file_frt = config_file_dir + "/" + jab_conf.hog_file;
    string HOFParam_file_frt = config_file_dir + "/" + jab_conf.hof_file;
    string crop_viewA, crop_viewB;
    
    camera_it = camera_serial_id_list.begin();
      

    while (camera_it != camera_serial_id_list.end())
    {
        crop_list_it = jab_crop_list.begin();
        while (crop_list_it != jab_crop_list.end()) {

            if (camera_it->second == crop_list_it->first)
            {
                if (camera_it->first == "viewA")
                {
                    crop_viewA = crop_list_it->second;
                }
                else if (camera_it->first == "viewB") {

                    crop_viewB = crop_list_it->second;
                }
            }
            crop_list_it++;
        }

        camera_it++;
    }
    string CropParam_file_sde = config_file_dir +  "/" + crop_viewA;
    string CropParam_file_frt = config_file_dir + "/" + crop_viewB;

    string behavior_names_str = jab_conf.beh_names;
    vector<string>beh_names_vec;
    jab_conf.convertStringtoVector(behavior_names_str, beh_names_vec);
    int num_behs = beh_names_vec.size();
    jab_conf.print();

    std::cout << "HOG file viewA" << HOGParam_file_sde << "\n"
        << "HOF file viewA" << HOFParam_file_sde << "\n"
        << "Crop file ViewA " << CropParam_file_sde << std::endl;

    // set jaaba modes to run the classifier
    int saveFeatures = cmdlineparams.saveFeat;
    int debug = cmdlineparams.debug;
    int classify_scores = cmdlineparams.classify_scores;
    
    bias::PredData sde_scr;
    bias::PredData frt_scr;
    vector<bias::PredData> scores;

    std::cout << "Command line arguments\n"
        << "Output Dir: " << cmdlineparams.output_dir
        << "\nis Video:" << cmdlineparams.isVideo
        << "\nsave features:" << cmdlineparams.saveFeat
        << "\ncompute jaaba: " << cmdlineparams.compute_jaaba
        << "\nclassify scores: " << cmdlineparams.classify_scores
        << "\nvisualize " << cmdlineparams.visualize
        << "\nisskip " << cmdlineparams.isSkip
        << "\n wait threshold" << cmdlineparams.wait_thres // wait time between jaaba views for computing a score
        << "\ nwindow_size " << cmdlineparams.window_size // averaging window size hoghof features
        << "\n DEBUG " << cmdlineparams.debug
        << "\n com port" << cmdlineparams.comport
        << std::endl;

    // Video Capture 
    string movie_name_suffix = cmdlineparams.movie_name_suffix;
    QString vidFile[nviews] = { "/movie_sde" + QString::fromStdString(movie_name_suffix) + ".avi" ,
                                "/movie_frt" + QString::fromStdString(movie_name_suffix) + ".avi" };
    std::cout << "Movie filename" << input_dir_path << vidFile[0].toStdString() << 
        std::endl;
    bias::videoBackend vid_sde(QString::fromStdString(input_dir_path) + vidFile[0]);
    bias::videoBackend vid_frt(QString::fromStdString(input_dir_path) + vidFile[1]);
    cv::VideoCapture cap_obj_sde = vid_sde.videoCapObject();
    cv::VideoCapture cap_obj_frt = vid_frt.videoCapObject();
    numFrames = vid_sde.getNumFrames(cap_obj_sde);
    int width = vid_sde.getImageWidth(cap_obj_sde);
    int height = vid_sde.getImageHeight(cap_obj_sde);

	numFrames_frt = vid_frt.getNumFrames(cap_obj_frt);
	numFrames_sde = vid_sde.getNumFrames(cap_obj_sde);
	std::cout << "side video has " << numFrames_sde << std::endl;
	std::cout << "frt video has " << numFrames_frt << std::endl;
    std::cout << "Video has " << numFrames << " frames" << std::endl;
    std::cout << "Video height" << height << std::endl;
    std::cout << "Video width" << width << std::endl;

    //HOG HOF Params
    bias::HOGHOF* feat_side = new bias::HOGHOF();
    bias::HOGHOF* feat_front = new bias::HOGHOF();

    feat_side->HOGParam_file = HOGParam_file_sde;
    feat_side->HOFParam_file = HOFParam_file_sde;
    feat_side->CropParam_file = CropParam_file_sde;

    feat_front->HOGParam_file = HOGParam_file_frt;
    feat_front->HOFParam_file = HOFParam_file_frt;
    feat_front->CropParam_file = CropParam_file_frt;

    // classifier
    bias::beh_class* classifier_sde = new bias::beh_class();
    bias::beh_class* classifier_frt = new bias::beh_class();

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

    feat_side->initHOGHOF(height, width);
    feat_front->initHOGHOF(height, width);

    feat_dim_side = feat_side->hog_shape.x* feat_side->hog_shape.y*feat_side->hog_shape.bin;
    feat_dim_front = feat_front->hog_shape.x* feat_front->hog_shape.y*feat_front->hog_shape.bin;
    feats_hog_out = new float[numFrames * feat_dim_side];
    featf_hog_out = new float[numFrames * feat_dim_front];
    feats_hof_out = new float[numFrames * feat_dim_side];
    featf_hof_out = new float[numFrames * feat_dim_front];
    printf("Feature dims side and front - %d-%d\n", feat_dim_side, feat_dim_front);

    if (classify_scores)
    {
        classifier_sde->beh_names = beh_names_vec;
        classifier_sde->num_behs = num_behs;
        classifier_frt->beh_names = beh_names_vec;
        classifier_frt->num_behs = num_behs;
        classifier_sde->classifier_file = classifier_filename;
        classifier_frt->classifier_file = classifier_filename;
        classifier_sde->classifier_concatenation_order = classifier_concatenation_order;
        classifier_frt->classifier_concatenation_order = classifier_concatenation_order;

        classifier_sde->allocate_model();
        classifier_frt->allocate_model();

        classifier_sde->loadclassifier_model();
        classifier_frt->loadclassifier_model();

        // translate indexes from joint view to single views
        //std::cout << "hog shape self " << feat_side->hog_shape.x  << " " << feat_side->hog_shape.y 
        //          << "hog shape partner "  << feat_front->hog_shape.x  << " " << feat_side->hog_shape.y 
        //          << std::endl;
        classifier_sde->getviewandfeature(&feat_side->hog_shape, &feat_front->hog_shape, "viewA");
        classifier_frt->getviewandfeature(&feat_side->hog_shape, &feat_front->hog_shape, "viewB");
        //std::cout << classifier_sde->translation_index_map_hog[0].size() << std::endl;
        //std:cout << classifier_sde->translation_index_map_hof[0].size() << std::endl;
        
        //allocate scores for the classifier
        scores.resize(numFrames);

    }

    int nRuns = 1;
    int cur_run = 0;
    int imageCnt = 0;
    int start_time = 0, end_time = 0;
    string jaaba_process_time_file;  
    vector<int64_t> jaaba_process_time;
    jaaba_process_time.resize(numFrames);
    bias::GetTime* gettime_ = new bias::GetTime();

    while (cur_run < nRuns)
    {
        imageCnt = 0;
        
        if(debug)
            jaaba_process_time_file = input_dir_path + "jaaba_process_time_" + to_string(cur_run) + ".csv";
        
        while (imageCnt < numFrames) {

            start_time = gettime_->getPCtime();

            if (!vidFile->isEmpty()) {

                cv::Mat curImg_side;
                cv::Mat curImg_frt;
                cv::Mat greySide;
                cv::Mat greyFront;

                curImg_side = vid_frames_sde[imageCnt];
                vid_sde.preprocess_vidFrame(curImg_side);
                feat_side->img.buf = curImg_side.ptr<float>(0);
                curImg_frt = vid_frames_frt[imageCnt];
                vid_frt.preprocess_vidFrame(curImg_frt);
                feat_front->img.buf = curImg_frt.ptr<float>(0);

                feat_side->genFeatures(imageCnt);
                feat_front->genFeatures(imageCnt);

                if (saveFeatures) {
                   
                    bias::saveFeatures(output_dir_path + "/hoghof_side_biasjaaba.csv",
                        feat_side->hog_out, feat_side->hof_out,
                        feat_dim_side, feat_dim_side);
                    bias::saveFeatures(output_dir_path + "/hoghof_front_biasjaaba.csv",
                        feat_front->hog_out, feat_front->hof_out,
                        feat_dim_front, feat_dim_front);
                }
                feat_side->averageWindowFeatures(window_size, imageCnt, 0);
                feat_front->averageWindowFeatures(window_size, imageCnt, 0);

                if (saveFeatures) {
                    
                    bias::saveFeatures(output_dir_path + "/hoghof_avg_side_biasjaaba.csv",
                        feat_side->hog_out_avg, feat_side->hof_out_avg,
                        feat_dim_side, feat_dim_side);
                    bias::saveFeatures(output_dir_path + "/hoghof_avg_front_biasjaaba.csv",
                        feat_front->hog_out_avg, feat_front->hof_out_avg,
                        feat_dim_front, feat_dim_front);
					if (imageCnt == numFrames - 1)
						std::cout << "Writing the last frame " << std::endl;
                }
                ;
				if (debug) {
					end_time = gettime_->getPCtime();
					jaaba_process_time[imageCnt] = (end_time - start_time);
				}
				

                if (classify_scores)
                {
                    classifier_sde->boost_classify(sde_scr.score,
                        feat_side->hog_out_avg, feat_side->hof_out_avg,
                        &feat_side->hog_shape, &feat_side->hof_shape,
                        classifier_sde->model,
                        imageCnt, "viewA");

                    classifier_frt->boost_classify(frt_scr.score,
                        feat_front->hog_out_avg, feat_front->hof_out_avg,
                        &feat_front->hog_shape, &feat_front->hof_shape,
                        classifier_sde->model,
                        imageCnt, "viewB");
                    classifier_sde->addScores(sde_scr.score, frt_scr.score);
                    scores[imageCnt].score = classifier_sde->finalscore.score;
                    scores[imageCnt].frameCount = imageCnt;
                    if (imageCnt == 0)
                        std::cout << "Lift Score first frame " << classifier_sde->finalscore.score[0] << std::endl;
                }
				//std::cout << imageCnt << std::endl;
                imageCnt++;
            }
        } // finished reading video from movies 
            
        if (debug) {
            gettime_->write_time_1d<int64_t>(jaaba_process_time_file, numFrames, jaaba_process_time);
            std::cout << "Time elapsed to process video frames " << (end_time - start_time)*0.001 << std::endl;
        }

		if (classify_scores) {
			string classifier_scr_file;
	        if (!movie_name_suffix.empty())
			    classifier_scr_file = output_dir_path + "/scores_offline_trial" + movie_name_suffix.back() + ".csv";
			else
				classifier_scr_file = output_dir_path + "/scores_offline.csv";
			write_score_final(classifier_scr_file, numFrames, scores);
		}
        cur_run++;
    }
   
    //fopen(output_dir_path + '');
	//wait for 5 secs before exiting to make sure finsihed writing
    Sleep(5000);

    //destroy hog hof ctx on the gpu
    HOFTeardown(feat_front->hof_ctx);
    HOGTeardown(feat_front->hog_ctx);

    HOFTeardown(feat_side->hof_ctx);
    HOGTeardown(feat_side->hog_ctx);

    //cudaDeviceReset();
   
    // delete hog/hof objects on the cpu
    delete feat_front;
    delete feat_side;

    delete[] feats_hog_out;
    delete[] featf_hog_out;
    delete[] feats_hof_out;
    delete[] featf_hof_out;

}
