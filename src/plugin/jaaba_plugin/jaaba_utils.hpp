#ifndef JAABA_UTILS_HPP
#define JAABA_UTILS_HPP

#include "rtn_status.hpp"

#include<vector>
#include <unordered_map>
#include <map>
#include <fstream>
#include <sstream>

#include <QPointer>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDialog>
#include <QMessageBox>
#include <iostream>

using namespace std;

namespace bias {

    struct PredData {

        vector<float>score = vector<float>(6,0.0);
        uint64_t score_ts = 0;
        uint64_t score_viewA_ts = 0;
        uint64_t score_viewB_ts = 0;
        unsigned int frameCount=0;
        int view = -1;
        uint64_t fstfrmtStampRef_ = 0;
    };

    class JaabaConfig {

        public:
            
            //Default Jaaba parameters
            static const string DEFAULT_VIEW;
            static const string DEFAULT_HOG_FILE;
            static const string DEFAULT_HOF_FILE;
            static const string DEFAULT_CROP_FILE;
            static const string DEFAULT_CLASSIFIER_FILE;
            static const string DEFAULT_JAABA_CONFIG_FILE;
            static const string DEFAULT_CONFIG_FILE_DIR;
            static const string DEFAULT_BEH_NAMES;
            static const string DEFAULT_CLASSIFIER_CONCATENATE_ORDER;
            static const int DEFAULT_WINDOW_SIZE;
            static const int DEFAULT_CUDA_DEVICE;
            static const int DEFAULT_NUM_BEHS;
            static const float DEFAULT_CLASSIFIER_THRES;
            static const bool DEFAULT_SETOUTPUT_TRIGGER;
            static const int DEFAULT_BAUDRATE;
            static const int DEFAULT_PERFRAME_LATENCY;

            //Jaaba parameters
            string jaaba_config_file = "";
            string config_file_dir = "";
            string view = "";
            string hog_file = "";
            string hof_file = "";
            string crop_file = "";
            string classifier_filename = "";
            string beh_names = "";
            string classifier_concatenation_order = "";
            int window_size;
            int cuda_device;
            int num_behs;
            float classifier_thresh;
            bool output_trigger;
            int baudRate;
            int perFrameLat;
            unordered_map<string, unsigned int> camera_serial_id;
            unordered_map<unsigned int, string> crop_file_list;
            
            JaabaConfig();

            //void readPluginConfig(JaabaConfig& jaaba_config, string param_file);
            //void readPluginConfig(string param_file);
            void print(std::ostream& out = std::cout);
            RtnStatus loadJAABAConfigFile();
            QVariantMap toMap();
            RtnStatus fromMap(QVariantMap configMap);
            RtnStatus setViewFromMap(QVariantMap configMap);
            RtnStatus setCropListFromMap(QVariantMap configMap);
            void convertStringtoVector(string& convertString, vector<string>& vectorString);
    };

    QJsonObject loadParams(const string& param_file);
    void saveFeatures(string filename,vector<float>& hog_feat, vector<float>& hof_feat,
                     int hog_num_elements, int hof_num_elements);
    uint64_t calculateExpectedlatency(const uint64_t fstframets, const int lat_thres,
                                  const unsigned int frameCount, int conversion_factor,
                                  unsigned long framerate);
    
}

#endif
