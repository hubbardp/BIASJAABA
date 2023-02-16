#ifndef JAABA_UTILS_HPP
#define JAABA_UTILS_HPP

#include<vector>
#include <unordered_map>
#include <map>

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

    struct JaabaConfig {

        string plugin_file_dir = "";
        string hog_file = "";
        string hof_file = "";
        unordered_map<string, unsigned int> camera_serial_id;
        unordered_map<unsigned int, string> crop_file_list;
        string classifier_filename = "";

    };

    void readPluginConfig(JaabaConfig& jaaba_config, string param_file);
    QJsonObject loadParams(const string& param_file);
}

#endif
