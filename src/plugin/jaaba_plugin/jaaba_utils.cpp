#include "jaaba_utils.hpp"


#include <iostream>
#include <iomanip>
#include <unordered_map>

#include <QVariantMap>
#include <QMessageBox>

namespace bias {

    // Default Jaaba parameters
    const string JaabaConfig::DEFAULT_VIEW = "";
    const string JaabaConfig::DEFAULT_HOG_FILE = "json_files/HOGParam.json";
    const string JaabaConfig::DEFAULT_HOF_FILE = "json_files/HOFParam.json";
    const string JaabaConfig::DEFAULT_CROP_FILE = "";
    const string JaabaConfig::DEFAULT_CLASSIFIER_FILE = "json_files/multiclassifier.mat";
    const string JaabaConfig::DEFAULT_BEH_NAMES = "Lift,Handopen,Grab,Supinate,Chew,Atmouth";
    const string JaabaConfig::DEFAULT_JAABA_CONFIG_FILE = "json_files/jaaba_config.json";
    const string JaabaConfig::DEFAULT_CONFIG_FILE_DIR = "";
    const string JaabaConfig::DEFAULT_CLASSIFIER_CONCATENATE_ORDER = "";
    const int JaabaConfig::DEFAULT_WINDOW_SIZE = 1;
    const int JaabaConfig::DEFAULT_CUDA_DEVICE = 0;
    const int JaabaConfig::DEFAULT_NUM_BEHS = 6;
    const int JaabaConfig::DEFAULT_BAUDRATE = 9600;
    const bool JaabaConfig::DEFAULT_SETOUTPUT_TRIGGER = 1;
    const float JaabaConfig::DEFAULT_CLASSIFIER_THRES = 0.0;
    const int JaabaConfig::DEFAULT_PERFRAME_LATENCY = 6000; // time in usec

    JaabaConfig::JaabaConfig()
    {
        view = DEFAULT_VIEW;
        hog_file = DEFAULT_HOG_FILE;
        hof_file = DEFAULT_HOF_FILE;
        crop_file = DEFAULT_CROP_FILE;
        classifier_filename = DEFAULT_CLASSIFIER_FILE;
        window_size = DEFAULT_WINDOW_SIZE;
        cuda_device = DEFAULT_CUDA_DEVICE;
        num_behs = DEFAULT_NUM_BEHS;
        beh_names = DEFAULT_BEH_NAMES;
        classifier_concatenation_order = DEFAULT_CLASSIFIER_CONCATENATE_ORDER;
    }

    QVariantMap JaabaConfig::toMap()
    {

        //camera view map
        QVariantMap viewMap;
        viewMap.insert("viewA", "");
        viewMap.insert("viewB", "");

        //crop file list
        QVariantMap cropListMap;
        cropListMap.insert("viewA_cropList", "");
        cropListMap.insert("viewB_cropList", "");

        // Create map for whole configuration
        QVariantMap configMap;
        configMap.insert("view", viewMap);
        configMap.insert("hog_file", "");
        configMap.insert("hof_file", "");
        configMap.insert("crop_file_list", cropListMap);
        configMap.insert("classifier_filename", "");
        configMap.insert("jaaba_config_file", "");
        configMap.insert("config_file_dir", "");
        configMap.insert("classifier_concatenation_order", "");
        configMap.insert("window_size", 1);
        configMap.insert("cuda_device", 0);
        configMap.insert("num_behs", 6);
        configMap.insert("beh_names","");
        configMap.insert("classifier_threshold", 0.0);
        configMap.insert("setoutputTrigger", true);
        configMap.insert("triggerDevcieBaudRate", 9600);
        configMap.insert("latency_threshold_perframe", 6000);
        return configMap;
    }

    void JaabaConfig::print(std::ostream& out) {
        out << "jaaba_config_file: " << jaaba_config_file << std::endl;
        out << "config_file_dir: " << config_file_dir << std::endl;
        out << "hog_file: " << hog_file << std::endl;
        out << "hof_file: " << hof_file << std::endl;
        out << "classifier_filename: " << classifier_filename << std::endl;
        out << "window_size: " << window_size << std::endl;
        out << "camera_serial_id:\n";
        for (const auto& pair : camera_serial_id) {
            std::cout << pair.first << ": " << pair.second << std::endl;
        }
        out << "crop_file_list:\n";
        for (const auto& pair : crop_file_list) {
            std::cout << pair.first << ": " << pair.second << std::endl;
        }
        out << "num_behs: " << num_behs << std::endl;
        out << "beh_names: " << beh_names << std::endl;
        out << "classifier_thresh: " << classifier_thresh << std::endl;
        out << "output_trigger: " << output_trigger << std::endl;
        out << "baudRate: " << baudRate << std::endl;
        out << "perFrameLat: " << perFrameLat << std::endl;
        out << "classifier_concatenation_order: " << classifier_concatenation_order << std::endl;
        out << "cuda_device: " << cuda_device << std::endl;
    }

    RtnStatus JaabaConfig::loadJAABAConfigFile() {

        RtnStatus rtnStatus;
        QString errMsgTitle("Load Parameter Error");
        std::cout << "Loading JAABA config file " << jaaba_config_file << std::endl;

        QJsonObject obj = loadParams(jaaba_config_file);
        QJsonObject jsonobj;
        QJsonValue value;
        // read config file dir
        if (obj.contains("config_file_dir"))
        {
            value = obj.value("config_file_dir");
            if (value.isString()){
                config_file_dir = value.toString().toStdString();
                std::cout << "JAABA config -> config_file_dir " << config_file_dir << std::endl;
            }
            else
            {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert config file dir to string");
            }
        }

        //read hog file from config
        if (obj.contains("hog_file"))
        {
            value = obj.value("hog_file");
            if (value.isString())
            {
                hog_file = value.toString().toStdString();
            }
            else
            {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert hog_file to string");
            }
        }

        //read hof file from config
        if (obj.contains("hof_file"))
        {
            value = obj.value("hof_file");
            if (value.isString())
            {
                hof_file = obj["hof_file"].toString().toStdString();
            }
            else
            {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert hof file to string");
            }
        }

        //read classifier name from config
        if (obj.contains("classifier_filename"))
        {
            value = obj.value("classifier_filename");
            if (value.isString())
            {
                classifier_filename = obj["classifier_filename"].toString().toStdString();
            }
            else
            {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert classifier file to string");
            }
        }

        //read window size from config
        if (obj.contains("window_size"))
        {
            value = obj.value("window_size");
            if (value.isString())
            {
                window_size = value.toString().toInt();
            }
            else if (value.isDouble()) {
                window_size = value.toInt();
            }
            else
            {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert window size to int");
            }
        }

        //read camera views guids 
        if (obj.contains("view"))
        {
            value = obj.value("view");
            if (value.isObject()) {
                RtnStatus rtnStatusView = setViewFromMap(value.toObject().toVariantMap());
                if(!rtnStatusView.success){
                    rtnStatus.success = false;
                    rtnStatus.appendMessage(rtnStatusView.message);
                }
            }
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to parse view");
            }

        }

        //read crop list from Map
        if (obj.contains("crop_file_list"))
        {
            value = obj.value("crop_file_list");
            if (value.isObject()) {
                RtnStatus rtnStatusCropList = setCropListFromMap(value.toObject().toVariantMap());
                if (!rtnStatusCropList.success) {
                    rtnStatus.success = false;
                    rtnStatus.appendMessage(rtnStatusCropList.message);
                }
            }
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to parse crop_file_list");
            }
        }

        //read number of behaviors 
        //made this obsolete -- number of behaviors in jaaba plugin is set by size of behavior names
        if (obj.contains("num_behaviors"))
        {

            value = obj.value("num_behaviors");
            if (value.isString())
            {
                num_behs = value.toString().toInt();
            }
            else if (value.isDouble()) {
                num_behs = value.toInt();
            }
            else
            {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert num_behaviors to int");
            }
        }

        //read behavior names 
        if (obj.contains("behavior_names"))
        {
            value = obj.value("behavior_names");
            if (value.isString())
            {
                beh_names = obj["behavior_names"].toString().toStdString();

            }
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert behavior names to string");
            }
        }

        //read output Trigger
        if (obj.contains("classifier_threshold"))
        {
            value = obj.value("classifier_threshold");
            if (value.isString())
            {
                classifier_thresh = value.toString().toFloat();
            }
            else if (value.isDouble()) {
                classifier_thresh = (float)value.toDouble();
            }
            else
            {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert classifier_thresh to float");
            }
        }

        //read outputTrigger boolean value
        if (obj.contains("setOutputTrigger"))
        {
            value = obj.value("setOutputTrigger");
            if (value.isBool())
            {
                output_trigger = value.toBool();
            }
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert output trigger to bool");
            }
        }

        // read baudrate int value
        if (obj.contains("triggerDeviceBaudRate"))
        {
            value = obj.value("triggerDeviceBaudRate");
            if (value.isString())
            {
                baudRate = value.toString().toInt();
            }
            else if (value.isDouble()) {
                baudRate = value.toInt();
            }
            else
            {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert triggerDeviceBaudRate to int");
            }
        }

        // read latency threshold per frame
        if (obj.contains("latency_threshold_perframe"))
        {
            value = obj.value("latency_threshold_perframe");
            if (value.isString())
            {
                perFrameLat = value.toString().toInt();
            }
            else if (value.isDouble()) {
                perFrameLat = value.toInt();
            }
            else
            {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert latency_threshold_perframe to int");
            }
        }

        if (obj.contains("classifier_concatenation_order"))
        {
            value = obj.value("classifier_concatenation_order");
            if (value.isString())
            {
                classifier_concatenation_order = value.toString().toStdString();

            }
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert classifier_concatenation_order to string");
            }
        }

        return rtnStatus;

    }

    RtnStatus JaabaConfig::fromMap(QVariantMap configMap)
    {
        RtnStatus rtnStatus;
        QVariantMap oldConfigMap = toMap();

        // read JAABA config file
        if (configMap.contains("jaaba_config_file"))
        {
            if (configMap["jaaba_config_file"].canConvert<QString>())
            {
                jaaba_config_file = configMap["jaaba_config_file"].toString().toStdString();
                std::cout << "JAABA Config File "  << jaaba_config_file << std::endl;
                rtnStatus = loadJAABAConfigFile();
            }
            else
            {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert JAABA config file to string");
            }
        }

        //read cuda device 
        if(configMap.contains("cuda_device"))
        {
            if (configMap["cuda_device"].canConvert<QString>())
            {
                cuda_device = configMap["cuda_device"].toInt();
            }
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert cuda device to int");
            }
        }

      
        return rtnStatus;
    }

    RtnStatus JaabaConfig::setViewFromMap(QVariantMap configMap)
    {
        RtnStatus rtnStatus;

        if (configMap.isEmpty())
        {
            rtnStatus.success = true;
            rtnStatus.message = QString("jaaba config view config empty");
            return rtnStatus;
        }

        rtnStatus.success = true;
        rtnStatus.message = QString("");
        QVariantMap::iterator it_start = configMap.begin();
        QVariantMap::iterator it_end = configMap.end();
        QVariant value, key;

        while (it_start != it_end) {

            value = it_start.value();
            key = it_start.key();
            //std::cout << "Key " << key.toString().toStdString() << ", Value "
            //    << value.toString().toStdString() << std::endl;

            camera_serial_id.insert(std::make_pair<const string, unsigned int>
                (key.toString().toStdString(), value.toString().toInt()));
            crop_file_list.insert(std::make_pair <unsigned int, string>
                (value.toString().toInt(), " "));

            it_start++;
        }

        return rtnStatus;
    }

    RtnStatus JaabaConfig::setCropListFromMap(QVariantMap configMap)
    {
        RtnStatus rtnStatus;

        if (configMap.isEmpty())
        {
            rtnStatus.success = true;
            rtnStatus.message = QString("jaaba config crop list empty");
            return rtnStatus;
        }

        rtnStatus.success = true;
        rtnStatus.message = QString("");
        QVariantMap::iterator it_start = configMap.begin();
        QVariantMap::iterator it_end = configMap.end();
        QVariant value, key;
        unsigned int cam_val;

        while (it_start != it_end) {

            value = it_start.value();
            key = it_start.key();

            const string cam_view = key.toString().toStdString().substr(0, 5);
            cam_val = camera_serial_id[cam_view];

            //std::cout << "Key " << key.toString().toStdString() << "Value "
            //    << value.toString().toStdString() << std::endl;

            crop_file_list[cam_val] = value.toString().toStdString();

            it_start++;
        }

        return rtnStatus;
    }

    //void readPluginConfig(JaabaConfig& jaaba_config, string param_file)
    // I don't see that this is used anywhere anymore
    //void JaabaConfig::readPluginConfig(string param_file)
    //{
    //    //string param_file = "C:/Users/27rut/BIAS/BIASJAABA/src/plugin/jaaba_plugin/plugin_config_files/jaaba_config_load.json";
    //    QJsonObject obj = loadParams(param_file);
    //    QJsonValue value;
    //    QString tmp_file;
    //    string filename ;

    //    foreach(const QString& key, obj.keys())
    //    {

    //        value = obj.value(key);
    //        if (value.isString() && key == "path_dir") {

    //            tmp_file = value.toString();
    //            filename = tmp_file.toStdString();
    //            const char old_val = '/';
    //            const char new_val = '\\';
    //            std::replace(filename.begin(), filename.end(), old_val, new_val);
    //            config_file_dir = filename;
    //        }
    //        else if (value.isString() && key == "jaaba_config_file") {

    //            tmp_file = value.toString();
    //            filename = tmp_file.toStdString();
    //            const char old_val = '/';
    //            const char new_val = '\\';
    //            std::replace(filename.begin(), filename.end(), old_val, new_val);
    //            jaaba_config_file = filename;
    //        }
    //        else if (value.isString() && (key == "viewA" || key == "viewB"))
    //        {

    //            camera_serial_id.insert(std::make_pair<const string,unsigned int>
    //                     ( key.toLocal8Bit().constData(),value.toString().toInt() ));
    //            crop_file_list.insert(std::make_pair <unsigned int, string>
    //                     (value.toString().toInt(), " "));
    //        }
    //        else if (value.isString() && (key == "viewA_Crop_file" || key == "viewB_Crop_file")) {
    //            
    //            unsigned int cam_val;
    //            const string cam_view = key.toStdString().substr(0, 5);
    //            cam_val = camera_serial_id[cam_view];

    //            tmp_file = value.toString();
    //            filename = tmp_file.toStdString();
    //            const char old_val = '/';
    //            const char new_val = '\\';
    //            std::replace(filename.begin(), filename.end(), old_val, new_val);

    //            crop_file_list[cam_val] = filename;
    //         
    //        }
    //        else if (value.isString() && (key == "HOG_file")) {
    //            
    //            tmp_file = value.toString();
    //            filename = tmp_file.toStdString();
    //            const char old_val = '/';
    //            const char new_val = '\\';
    //            std::replace(filename.begin(), filename.end(), old_val, new_val);
    //            hog_file = filename;

    //        }
    //        else if (value.isString() && (key == "HOF_file")) {

    //            tmp_file = value.toString();
    //            filename = tmp_file.toStdString();
    //            const char old_val = '/';
    //            const char new_val = '\\';
    //            std::replace(filename.begin(), filename.end(), old_val, new_val);
    //            hof_file = filename;

    //        }
    //        else if (value.isString() && (key == "classifier_filename")) {
    //            
    //            tmp_file = value.toString();
    //            filename = tmp_file.toStdString();
    //            const char old_val = '/';
    //            const char new_val = '\\';
    //            std::replace(filename.begin(), filename.end(), old_val, new_val);
    //            classifier_filename = filename;
    //        }
    //        else if (value.isString() && (key == "window_size")) {
    //            window_size = value.toInt();
    //        }

    //    }

    //    std::cout << "Window Size" << window_size << std::endl;
    //    std::cout << filename << std::endl;
    //}

    void JaabaConfig::convertStringtoVector(string& convertString, vector<string>& vectorString)
    {
        stringstream convertstringstream(convertString);
        string cur_str;
        while (!convertstringstream.eof())
        {
            getline(convertstringstream, cur_str, ',');
            vectorString.push_back(cur_str);
        }
    }

    QJsonObject loadParams(const string& param_file)
    {

        QString pfile;
        pfile = QString::fromStdString(param_file);

        //RtnStatus rtnStatus;
        QJsonObject obj;
        QString errMsgTitle("Load Parameter Error");

        QFile parameterFile(pfile);
        if (!parameterFile.exists())
        {

            QString errMsgTitle = QString("Params");
            QString errMsgText = QString("Parameter file, %1").arg(pfile);
            errMsgText += QString(", does not exist - using default values");
            QMessageBox::critical(nullptr, errMsgTitle, errMsgText);
            return obj;
        }

        bool ok = parameterFile.open(QIODevice::ReadOnly);
        if (!ok)
        {
            QString errMsgTitle = QString("Params");
            QString errMsgText = QString("Unable to open parameter file %1").arg(pfile);
            errMsgText += QString(" - using default values");
            QMessageBox::critical(nullptr, errMsgTitle, errMsgText);
            return obj;
        }

        QByteArray paramJson = parameterFile.readAll();
        parameterFile.close();

        QJsonDocument doc = QJsonDocument::fromJson(paramJson);
        obj = doc.object();

        /*QJsonValue value;
        printf("param file %s\n", param_file.toLocal8Bit().constData());
        foreach(const QString& key, obj.keys())
        {

            value = obj.value(key);
            std::cout << "inside****" << std::endl;
            if (value.isString() && key == "path_dir")
                plugin_file = value.toString();
        }
        printf("plugin file %s", plugin_file.toLocal8Bit().constData());*/
        return obj;
    }

    void saveFeatures(string filename, vector<float>& hog_feat, vector<float>& hof_feat,
                      int hog_num_elements, int hof_num_elements) 
    {
        std::ofstream x_out;
        x_out.open(filename.c_str(), std::ios_base::app);


        for (int j = 0; j < hog_num_elements; j++)
        {
            x_out << setprecision(6) << hog_feat[j] << ",";
        }

        for (int k = 0; k < hof_num_elements; k++)
        {
            if (k == (hof_num_elements - 1))
                x_out << setprecision(6) << hof_feat[k] << "\n";
            else
                x_out << setprecision(6) << hof_feat[k] << ",";
        }

        x_out.close();

    }

    uint64_t calculateExpectedlatency(const uint64_t fstframets, const int lat_thres,
                                      const unsigned int frameCount, int conversion_factor,
                                      unsigned long framerate)
    {

        uint64_t datarate = static_cast<uint64_t>((1.0 / (float)framerate) * 1000000); // unit usecs
        //std::cout << "datarate " << datarate << "\n"
        //    << "fstframets " << fstframets  << "\n"
        //    << "conversion factor " << conversion_factor << std::endl;
        uint64_t expLat = (fstframets*conversion_factor) + static_cast<uint64_t>(lat_thres)
                          + static_cast<uint64_t>(datarate*(frameCount));
        return expLat;
    }

}
