#ifndef BEH_CLASS_HPP
#define BEH_CLASS_HPP 


#include "video_utils.hpp"
#include "image_fcns.h"
#include "rtn_status.hpp"

//HOGHOF includes 
#include "hog.h"
#include "hof.h"

//#include <QString>
//#include <QList>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>

class beh_class {

  public:

    HOGParameters HOGParams;
    HOFParameters HOFParams;
    CropParams Cropparams;
    
    void loadHOGParams();
    void loadHOFParams();
    void loadCropParams();
    void copytoHOGParams(QJsonObject& obj);
    void copytoHOFParams(QJsonObject& obj);
    void copytoCropParams(QJsonObject& obj);
    float copyValueFloat(QJsonObject& ob, QString subobj_key);
    int copyValueInt(QJsonObject& ob, QString subobj_key);
    void allocateCrop(int sz);

};
#endif  
