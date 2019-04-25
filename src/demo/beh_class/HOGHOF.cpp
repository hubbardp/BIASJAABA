#include "HOGHOF.hpp"
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <QDebug>

using namespace bias; 

void HOGHOF::loadHOGParams() { 

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

void HOGHOF::loadHOFParams() {

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

void HOGHOF::loadCropParams() {

    RtnStatus rtnStatus;
    QString errMsgTitle("Load Parameter Error");

    QFile parameterFile(CropParam_file);
    if (!parameterFile.exists())
    {
        QString errMsgText = QString("Parameter file, %1").arg(CropParam_file);
        errMsgText += QString(", does not exist - using default values");
        qDebug() << errMsgText;
        return;
    }

    bool ok = parameterFile.open(QIODevice::ReadOnly);
    if (!ok)
    {
        QString errMsgText = QString("Unable to open parameter file %1").arg(CropParam_file);
        errMsgText += QString(" - using default values");
        qDebug() << errMsgText;
        return;
    }

    QByteArray paramJson = parameterFile.readAll();
    parameterFile.close();

    QJsonDocument doc = QJsonDocument::fromJson(paramJson);
    QJsonObject obj = doc.object();
    copytoCropParams(obj);

}

void HOGHOF::loadImageParams(int img_width, int img_height) {

    img.w = img_width;
    img.h = img_height;
    img.pitch = img_width;
    img.type = hog_f32;
    img.buf = nullptr;
 
}

/*void HOGHOF::loadclassifiermodel(boost_classifier& model, QString& file) {

    std::vector<std::string>dataset_name{"alpha","dim","dir","error","tr"};
    allocate_model(file,dataset_name[0],&model);
    readh5(file,dataset_name[0],&model.cls_alpha.data()[0]);
    readh5(file,dataset_name[1],&model.cls_dim.data()[0]);
    readh5(file,dataset_name[2],&model.cls_dir.data()[0]);
    readh5(file,dataset_name[3],&model.cls_error.data()[0]);
    readh5(file,dataset_name[4],&model.cls_tr.data()[0]);

}*/

void HOGHOF::copytoHOGParams(QJsonObject& obj) {

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


void HOGHOF::copytoHOFParams(QJsonObject& obj) {

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


void HOGHOF::copytoCropParams(QJsonObject& obj) {

    QJsonValue value;
    foreach(const QString& key, obj.keys()) {

        value = obj.value(key);
        if(value.isString() && key == "crop_flag")
            Cropparams.crop_flag = value.toString().toInt();
    
        if(value.isString() && key == "ncells")
            Cropparams.ncells = value.toString().toInt();
 
        if(value.isString() && key == "npatches")
            Cropparams.npatches = value.toString().toInt();

        if(value.isArray() && key == "interest_pnts") {

            QJsonArray arr = value.toArray();
            allocateCrop(arr.size());
            int idx = 0;
            foreach(const QJsonValue& id, arr) {       
                
                QJsonArray ips = id.toArray();
                Cropparams.interest_pnts[idx*2+ 0] = ips[0].toInt();
                Cropparams.interest_pnts[idx*2 + 1] = ips[1].toInt();
                idx = idx + 1;
            }
        }
    }
    if(!Cropparams.crop_flag) {  // if croping is not enabled
      Cropparams.interest_pnts = nullptr;
      Cropparams.ncells = 0; 
      Cropparams.npatches = 0;      
    }
}

void HOGHOF::allocateCrop(int sz) {

    Cropparams.interest_pnts = (int*)malloc(2*sz*sizeof(int));

}


int HOGHOF::copyValueInt(QJsonObject& ob, 
                            QString subobj_key) {

    QJsonValue value = ob.value(subobj_key);
    if(value.isString())
        return (value.toString().toInt());
                        
}


float HOGHOF::copyValueFloat(QJsonObject& ob, 
                            QString subobj_key) {

    QJsonValue value = ob.value(subobj_key);
    if(value.isString())
        return (value.toString().toFloat());

}

void HOGHOF::allocateHOGoutput(float* out, HOGContext* hog_init) {

    size_t hog_nbytes = HOGOutputByteCount(hog_init);
    out = (float*)malloc(hog_nbytes);

}


void HOGHOF::allocateHOFoutput(float* out, const HOFContext* hof_init) {

    size_t hof_nbytes = HOFOutputByteCount(hof_init);
    out = (float*)malloc(hof_nbytes);

}
