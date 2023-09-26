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
    const string JaabaConfig::DEFAULT_CONFIG_FILE_DIR = "";
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
        configMap.insert("config_file_dir", "");
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

    RtnStatus JaabaConfig::fromMap(QVariantMap configMap)
    {
        RtnStatus rtnStatus;
        QVariantMap oldConfigMap = toMap();

        // read jaaba config file dir
        //read hog file from config
        if (configMap.contains("config_file_dir"))
        {
            if (configMap["config_file_dir"].canConvert<QString>())
            {
                config_file_dir = configMap["config_file_dir"].toString().toStdString();
                std::cout << "Config file Dir " << config_file_dir << std::endl;
            }
            else
            {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert config file dir to string");
            }
        }

        //read hog file from config
        if (configMap.contains("hog_file"))
        {
            if (configMap["hog_file"].canConvert<QString>())
            {
                hog_file = configMap["hog_file"].toString().toStdString();
            }
            else
            {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert hog_file to string");
            }
        }

        //read hof file from config
        if (configMap.contains("hog_file"))
        {
            if (configMap["hof_file"].canConvert<QString>())
            {
                hof_file = configMap["hof_file"].toString().toStdString();
            }
            else
            {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert hof file to string");
            }
        }

        //read classifier name from config
        if (configMap.contains("classifier_filename"))
        {
            if (configMap["hog_file"].canConvert<QString>())
            {
                classifier_filename = configMap["classifier_filename"].toString().toStdString();
            }
            else
            {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert classifier file to string");
            }
        }


        //read window size from config
        if (configMap.contains("window_size"))
        {
            if (configMap["window_size"].canConvert<QString>())
            {
                window_size = configMap["window_size"].toInt();
     
            }
            else
            {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert window size to int");
            }
        }
        
        //read camera views guids 
        if (configMap.contains("view"))
        {
            QVariantMap viewMap;
            viewMap = configMap["view"].toMap();

            RtnStatus rtnStatusView = setViewFromMap(viewMap);
            if (!rtnStatusView.success)
            {
                setViewFromMap(oldConfigMap["view"].toMap());
            }
        }

        //read crop list from Map
        if (configMap.contains("crop_file_list"))
        {
            QVariantMap cropListMap;
            cropListMap = configMap["crop_file_list"].toMap();

            RtnStatus rtnStatusView = setCropListFromMap(cropListMap);
            if (!rtnStatusView.success)
            {
                setViewFromMap(oldConfigMap["crop_file_list"].toMap());
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

        //read number of behaviors 
        if (configMap.contains("num_behaviors"))
        {
            if (configMap["num_behaviors"].canConvert<QString>())
            {
                num_behs = configMap["num_behaviors"].toInt();
            }
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert cuda device to int");
            }
        }

        //read behavior names 
        if (configMap.contains("behavior_names"))
        {
            if (configMap["behavior_names"].canConvert<QString>())
            {
                beh_names = configMap["behavior_names"].toString().toStdString();
          
            }
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert cuda device to int");
            }
        }

        //read output Trigger
        if (configMap.contains("classifier_threshold"))
        {
            if (configMap["classifier_threshold"].canConvert<QString>())
            {
                classsifer_thres = configMap["classifier_threshold"].toFloat();
            }
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert classifier thres to float");
            }
        }

        //read outputTrigger boolean value
        if (configMap.contains("setOutputTrigger"))
        {
            if (configMap["setOutputTrigger"].canConvert<QString>())
            {
                output_trigger = configMap["setOutputTrigger"].toBool();
                std::cout << "set output trigger ********** " << output_trigger << std::endl;
            }
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert output trigger to bool");
            }
        }

        // read baudrate int value
        if (configMap.contains("triggerDeviceBaudRate"))
        {
            if (configMap["triggerDeviceBaudRate"].canConvert<QString>())
            {
                baudRate = configMap["triggerDeviceBaudRate"].toInt();
            }
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert baudRate to Int");
            }
        }

        // read latency threshold per frame
        if (configMap.contains("latency_threshold_perframe"))
        {
            if (configMap["latency_threshold_perframe"].canConvert<QString>())
            {
                perFrameLat = configMap["latency_threshold_perframe"].toInt();
            }
            else {
                rtnStatus.success = false;
                rtnStatus.appendMessage("unable to convert preframe latency to int");
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
            //std::cout << "Key " << key.toString().toStdString() << "Value "
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
    void JaabaConfig::readPluginConfig(string param_file)
    {
        //string param_file = "C:/Users/27rut/BIAS/BIASJAABA/src/plugin/jaaba_plugin/plugin_config_files/jaaba_config_load.json";
        QJsonObject obj = loadParams(param_file);
        QJsonValue value;
        QString tmp_file;
        string filename ;

        foreach(const QString& key, obj.keys())
        {

            value = obj.value(key);
            if (value.isString() && key == "path_dir") {

                tmp_file = value.toString();
                filename = tmp_file.toStdString();
                const char old_val = '/';
                const char new_val = '\\';
                std::replace(filename.begin(), filename.end(), old_val, new_val);
                config_file_dir = filename;
            }
            else if (value.isString() && (key == "viewA" || key == "viewB"))
            {

                camera_serial_id.insert(std::make_pair<const string,unsigned int>
                         ( key.toLocal8Bit().constData(),value.toString().toInt() ));
                crop_file_list.insert(std::make_pair <unsigned int, string>
                         (value.toString().toInt(), " "));
            }
            else if (value.isString() && (key == "viewA_Crop_file" || key == "viewB_Crop_file")) {
                
                unsigned int cam_val;
                const string cam_view = key.toStdString().substr(0, 5);
                cam_val = camera_serial_id[cam_view];

                tmp_file = value.toString();
                filename = tmp_file.toStdString();
                const char old_val = '/';
                const char new_val = '\\';
                std::replace(filename.begin(), filename.end(), old_val, new_val);

                crop_file_list[cam_val] = filename;
             
            }
            else if (value.isString() && (key == "HOG_file")) {
                
                tmp_file = value.toString();
                filename = tmp_file.toStdString();
                const char old_val = '/';
                const char new_val = '\\';
                std::replace(filename.begin(), filename.end(), old_val, new_val);
                hog_file = filename;

            }
            else if (value.isString() && (key == "HOF_file")) {

                tmp_file = value.toString();
                filename = tmp_file.toStdString();
                const char old_val = '/';
                const char new_val = '\\';
                std::replace(filename.begin(), filename.end(), old_val, new_val);
                hof_file = filename;

            }
            else if (value.isString() && (key == "classifier_filename")) {
                
                tmp_file = value.toString();
                filename = tmp_file.toStdString();
                const char old_val = '/';
                const char new_val = '\\';
                std::replace(filename.begin(), filename.end(), old_val, new_val);
                classifier_filename = filename;
            }
            else if (value.isString() && (key == "window_size")) {
                window_size = value.toInt();
            }

        }

        std::cout << "Window Size" << window_size << std::endl;
        std::cout << filename << std::endl;
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