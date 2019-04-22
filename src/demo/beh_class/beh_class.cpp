#include "beh_class.hpp"
#include "json.hpp"
#include "json_utils.hpp"
#include <iostream> 
#include <QDebug>
#include <QJsonDocument>
#include <QJsonValue>
#include <QJsonArray>
#include <QJsonObject>


//CONSTANTS

QString HOGParam_file = "/groups/branson/home/patilr/BIAS/BIASJAABA/src/demo/beh_class/json_files/HOGparam.json";
QString HOFParam_file = "/groups/branson/home/patilr/BIAS/BIASJAABA/src/demo/beh_class/json_files/HOFparam.json";


int main(int argc, char* argv[]) {

    beh_class classifier;
    classifier.loadHOGParams();
    classifier.loadHOFParams();
    // Create video context 
    /*QString vidFile = "/nrs/branson/jab_experiments/M274Vglue2_Gtacr2_TH/20180814/M274_20180814_v002/cuda_dir/movie_sde.avi" ;
    bias::videoBackend vid(vidFile) ;
    cv::VideoCapture capture = vid.videoCapObject(vid);

    //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    //cv::imshow( "Display window", img );                   // Sh
    //cv::waitKey(0);
    //qDebug() << "sdfs " << vid.filename;

    // HOG/HOF Context 
    HOFParameters HOF_params;
    HOGParameters HOG_params;
    CropParams crp_params;
    HOGImage img;

    std::string fname = vidFile.toStdString();
    std::string view = "";
    int verbose = 0;
    int num_frames = vid.getNumFrames(capture); 
    int height = vid.getImageHeight(capture);
    int width =  vid.getImageWidth(capture);
    std::cout << height << " " << width << " " << "" << num_frames << std::endl;
    parse_input_HOG(argc, argv, HOG_params, img, num_frames, verbose, view, fname, crp_params);
    parse_input_HOF(argc, argv, HOF_params, num_frames, verbose, view, fname, crp_params);  
 
    cv::Mat tmp_img = vid.getImage(capture);
    vid.convertImagetoFloat(tmp_img);
    img.buf = tmp_img.ptr<float>(0);
    vid.releaseCapObject(capture);*/

}

using namespace bias; 

void beh_class::loadHOGParams() { 

    RtnStatus rtnStatus;
    QString errMsgTitle("Load Parameter Error");

    QFile parameterFile(HOGParam_file);
    if (!parameterFile.exists())
    {
        QString errMsgText = QString("Parameter file, %1").arg(HOGParam_file);
        errMsgText += QString(", does not exist - using default values");
        qDebug() << errMsgText;
        return;
    }

    bool ok = parameterFile.open(QIODevice::ReadOnly);
    if (!ok)
    {
        QString errMsgText = QString("Unable to open parameter file %1").arg(HOGParam_file);
        errMsgText += QString(" - using default values");
        qDebug() << errMsgText;
        return;
    }

    QByteArray paramJson = parameterFile.readAll();
    parameterFile.close();

    QJsonDocument doc = QJsonDocument::fromJson(paramJson);
    QJsonObject obj = doc.object();
    copytoHOGParams(obj);
   
}

void beh_class::loadHOFParams() {

    RtnStatus rtnStatus;
    QString errMsgTitle("Load Parameter Error");

    QFile parameterFile(HOFParam_file);
    if (!parameterFile.exists())
    {
        QString errMsgText = QString("Parameter file, %1").arg(HOFParam_file);
        errMsgText += QString(", does not exist - using default values");
        qDebug() << errMsgText;
        return;
    }

    bool ok = parameterFile.open(QIODevice::ReadOnly);
    if (!ok)
    {
        QString errMsgText = QString("Unable to open parameter file %1").arg(HOFParam_file);
        errMsgText += QString(" - using default values");
        qDebug() << errMsgText;
        return;
    }

    QByteArray paramJson = parameterFile.readAll();
    parameterFile.close();

    QJsonDocument doc = QJsonDocument::fromJson(paramJson);  
    QJsonObject obj = doc.object();
    copytoHOFParams(obj);

}


void beh_class::copytoHOGParams(QJsonObject& obj) {

    QJsonValue value;
    foreach(const QString& key, obj.keys()) {

        value = obj.value(key);
        if(value.isString() && key == "nbins")
            HOGParams.nbins = value.toString().toInt();

        if(value.isObject() && key == "cell") {
          
            QJsonObject ob = value.toObject();
            HOGParams.cell.h = copyValueInt(ob, "h");
            HOGParams.cell.w = copyValueInt(ob, "w");

        }
    }
}


void beh_class::copytoHOFParams(QJsonObject& obj) {

    QJsonValue value;  
    foreach(const QString& key, obj.keys()) {
    
        value = obj.value(key);
        if(value.isString() && key == "nbins")
            HOFParams.nbins = value.toString().toInt();
     
        if(value.isObject() && key == "cell") {
   
            QJsonObject ob = value.toObject();
            HOFParams.cell.h = copyValueInt(ob, "h");
            HOFParams.cell.w = copyValueInt(ob, "w");           

        }

        if(value.isObject() && key == "lk") {

            QJsonObject ob = value.toObject();
            foreach(const QString& key, ob.keys()) {

                value = ob.value(key);
                if(value.isString() && key == "threshold") 
                    HOFParams.lk.threshold = value.toString().toFloat();

                if(value.isObject() && key == "sigma") {

                    QJsonObject ob = value.toObject();
                    HOFParams.lk.sigma.smoothing = copyValueFloat(ob, "smoothing");
                    HOFParams.lk.sigma.derivative = copyValueFloat(ob, "derivative");

                }
            }
        }
    }   
}

int beh_class::copyValueInt(QJsonObject& ob, 
                            QString subobj_key) {

    QJsonValue value = ob.value(subobj_key);
    if(value.isString())
        return (value.toString().toInt());
                        
}


float beh_class::copyValueFloat(QJsonObject& ob, 
                            QString subobj_key) {

    QJsonValue value = ob.value(subobj_key);
    if(value.isString())
        return (value.toString().toFloat());

}


