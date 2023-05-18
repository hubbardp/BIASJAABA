#ifndef JAABA_UTILS_HPP
#define JAABA_UTILS_HPP

#include "rtn_status.hpp"

#include<vector>
#include <unordered_map>
#include <map>
#include <fstream>

#include <QPointer>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDialog>
#include <QMessageBox>


using namespace std;

namespace bias {

    struct PredData {

        vector<float>score = vector<float>(6,0.0);
        uint64_t score_ts = 0;
        uint64_t score_side_ts = 0;
        uint64_t score_front_ts = 0;
        unsigned int frameCount=0;
        int view = -1;
    };

    class JaabaConfig {

        public:
            
            //Default Jaaba parameters
            static const string DEFAULT_VIEW;
            static const string DEFAULT_HOG_FILE;
            static const string DEFAULT_HOF_FILE;
            static const string DEFAULT_CROP_FILE;
            static const string DEFAULT_CLASSIFIER_FILE;
            static const string DEFAULT_CONFIG_FILE_DIR;
            static const int DEFAULT_WINDOW_SIZE;


            //Jaaba parameters
            string config_file_dir = "";
            string view = "";
            string hog_file = "";
            string hof_file = "";
            string crop_file = "";
            string classifier_filename = "";
            int window_size;
            unordered_map<string, unsigned int> camera_serial_id;
            unordered_map<unsigned int, string> crop_file_list;
            
            JaabaConfig();

            //void readPluginConfig(JaabaConfig& jaaba_config, string param_file);
            void readPluginConfig(string param_file);
            QVariantMap toMap();
            RtnStatus fromMap(QVariantMap configMap);
            RtnStatus setViewFromMap(QVariantMap configMap);
            RtnStatus setCropListFromMap(QVariantMap configMap);
    };

    QJsonObject loadParams(const string& param_file);
    void saveFeatures(string filename,vector<float>& hog_feat, vector<float>& hof_feat,
                     int hog_num_elements, int hof_num_elements);
    
}

#endif
