#ifndef HOGHOF_CLASS_HPP
#define HOGHOF_CLASS_HPP 

#include "hog.h"
#include "hof.h"
#include "rtn_status.hpp"

#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

class HOGHOF {

  public:
 
    QString HOGParam_file; 
    QString HOFParam_file;
    QString CropParam_file;
    
    HOGParameters HOGParams;
    HOFParameters HOFParams;
    CropParams Cropparams;
    HOGImage img;
    float* hog_out;
    float* hof_out;

    void loadHOGParams();
    void loadHOFParams();
    void loadCropParams();
    void loadImageParams(int img_width, int img_height);

    void copytoHOGParams(QJsonObject& obj);
    void copytoHOFParams(QJsonObject& obj);
    void copytoCropParams(QJsonObject& obj);
    void allocateHOGoutput(float* out, HOGContext* hog_init);
    void allocateHOFoutput(float* out, const HOFContext* hof_init);

    float copyValueFloat(QJsonObject& ob, QString subobj_key);
    int copyValueInt(QJsonObject& ob, QString subobj_key);
    void allocateCrop(int sz);

};
































































#endif()
