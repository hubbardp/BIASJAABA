#include <iostream>
#include "jaaba_utils.hpp"
#include <unordered_map>

namespace bias {

    void readPluginConfig(JaabaConfig& jaaba_config, string param_file)
    {
        //string param_file = "C:/Users/27rut/BIAS/BIASJAABA/src/plugin/jaaba_plugin/plugin_config_files/jaaba_config_load.json";
        std::cout << "read" << param_file << std::endl;
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
                jaaba_config.plugin_file_dir = filename;
            }
            else if (value.isString() && (key == "viewA" || key == "viewB"))
            {

                jaaba_config.camera_serial_id.insert(std::make_pair<const string,unsigned int>
                         ( key.toLocal8Bit().constData(),value.toString().toInt() ));
                jaaba_config.crop_file_list.insert(std::make_pair <unsigned int, string>
                         (value.toString().toInt(), " "));
            }
            else if (value.isString() && (key == "viewA_Crop_file" || key == "viewB_Crop_file")) {
                
                unsigned int cam_val;
                const string cam_view = key.toStdString().substr(0, 5);
                cam_val = jaaba_config.camera_serial_id[cam_view];

                tmp_file = value.toString();
                filename = tmp_file.toStdString();
                const char old_val = '/';
                const char new_val = '\\';
                std::replace(filename.begin(), filename.end(), old_val, new_val);

                jaaba_config.crop_file_list[cam_val] = filename;
             
            }
            else if (value.isString() && (key == "HOG_file")) {
                
                tmp_file = value.toString();
                filename = tmp_file.toStdString();
                const char old_val = '/';
                const char new_val = '\\';
                std::replace(filename.begin(), filename.end(), old_val, new_val);
                jaaba_config.hog_file = filename;

            }
            else if (value.isString() && (key == "HOF_file")) {

                tmp_file = value.toString();
                filename = tmp_file.toStdString();
                const char old_val = '/';
                const char new_val = '\\';
                std::replace(filename.begin(), filename.end(), old_val, new_val);
                jaaba_config.hof_file = filename;

            }
            else if (value.isString() && (key == "classifier_filename")) {
                
                tmp_file = value.toString();
                filename = tmp_file.toStdString();
                const char old_val = '/';
                const char new_val = '\\';
                std::replace(filename.begin(), filename.end(), old_val, new_val);
                jaaba_config.classifier_filename = filename;
            }

        }

        
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

}