#ifndef HOGHOF_CLASS_HPP
#define HOGHOF_CLASS_HPP 

#include "hog.h"
#include "hof.h"
//#include "video_utils.hpp"
#include "logger.h"
//#include "utils.hpp"
#include "rtn_status.hpp"

#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDialog>
#include <QMessageBox>

struct HOGShape {

    int x;
    int y;
    int bin;

    void operator=(const HOGFeatureDims& other) {

        x = other.x;
        y = other.y;
        bin = other.bin;
    }

};

struct HOFShape {

    int x;
    int y;
    int bin;

    void operator=(const HOGFeatureDims& other) {

        x = other.x;
        y = other.y;
        bin = other.bin;
    }

};


namespace bias {

    class HOGHOF: public QDialog 
    {

        public:

            QString HOGParam_file; 
            QString HOFParam_file;
            QString CropParam_file;
            bool isHOGPathSet=false;
            bool isHOFPathSet=false;
            int startFrameSet=true;
            size_t hog_outputbytes;
            size_t hof_outputbytes;
            HOGHOF(QWidget *parent);
    
         
            HOGParameters HOGParams;
            HOFParameters HOFParams;
            CropParams Cropparams;
            HOGContext *hog_ctx;
            HOFContext *hof_ctx;
            HOGImage img;
            HOGShape hog_shape;
            HOFShape hof_shape;
            
            std::vector<float> hog_out;
            std::vector<float> hof_out;

            void loadHOGParams();
            void loadHOFParams();
            void loadCropParams();
            void loadImageParams(int img_width, int img_height);
            //void genFeatures();

        private: 

            void initialize();

            void copytoHOGParams(QJsonObject& obj);
            void copytoHOFParams(QJsonObject& obj);
            void copytoCropParams(QJsonObject& obj);
            //void allocateHOGoutput(float* out, HOGContext* hog_init);
            //void allocateHOFoutput(float* out, const HOFContext* hof_init);

            float copyValueFloat(QJsonObject& ob, QString subobj_key);
            int copyValueInt(QJsonObject& ob, QString subobj_key);
            void allocateCrop(int sz);
    };

}
#endif
