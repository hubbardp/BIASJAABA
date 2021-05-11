#ifndef HOGHOF_CLASS_HPP
#define HOGHOF_CLASS_HPP 

#include "hog.h"
#include "hof.h"
#include "video_utils.hpp"
#include "logger.h"
#include "utils.hpp"
#include "rtn_status.hpp"

#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
 
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

struct Params {

    QString HOGParam_file;
    QString HOFParam_file;
    QString CropParam_file;

};

class HOGHOF {

  public:

    HOGHOF();
    ~HOGHOF();
    QString HOGParam_file;
    QString HOFParam_file;
    QString CropParam_file;
    Params param_files;
    cv::VideoCapture capture;

    HOGParameters HOGParams;
    HOFParameters HOFParams;
    CropParams Cropparams;
    HOGImage img;
    HOGShape hog_shape;
    HOFShape hof_shape;
    struct HOGContext* hog_ctx;
    struct HOFContext* hof_ctx;
    
    float* tmp_hog;
    float* tmp_hof;
    std::vector<float> hog_out;
    std::vector<float> hof_out;

    void initialize_params(Params& param_file);
    void initialize_vidparams(bias::videoBackend& vid, Params& param_file);
    void initializeHOGHOF(int& width, int& height, int& num_frames);
    void loadHOGParams();
    void loadHOFParams();
    void loadCropParams();
    void loadImageParams(int img_width, int img_height);
    void getvid_frame(bias::videoBackend & vid);
    void process_vidFrame(int frame);
    void process_camFrame();
    void genFeatures(QString vidname, QString& CropFile);

    void copytoHOGParams(QJsonObject& obj);
    void copytoHOFParams(QJsonObject& obj);
    void copytoCropParams(QJsonObject& obj);
    //void allocateHOGoutput(float* out, HOGContext* hog_init);
    //void allocateHOFoutput(float* out, const HOFContext* hof_init);

    float copyValueFloat(QJsonObject& ob, QString subobj_key);
    int copyValueInt(QJsonObject& ob, QString subobj_key);
    void allocateCrop(int sz);

    void deInitializeHOGHOF();
    //TEST
    //void write_time(std::string file, int framenum, std::vector<float> timeVec);

};

#endif
