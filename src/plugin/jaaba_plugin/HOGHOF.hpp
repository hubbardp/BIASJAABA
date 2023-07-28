#ifndef HOGHOF_CLASS_HPP
#define HOGHOF_CLASS_HPP 

#include "hog.h"
#include "hof.h"
//#include "video_utils.hpp"
#include "logger.h"
//#include "utils.hpp"
#include "rtn_status.hpp"
#include "jaaba_utils.hpp"
#include "lockable.hpp"

#include "string.h"
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDialog>
#include <QMessageBox>

#include <queue>

using namespace std;

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

    class HOGHOF : public QDialog 
    {

        Q_OBJECT

        public:

            QString plugin_file;
            string HOGParam_file;
            string HOFParam_file;
            string CropParam_file;
            bool isHOGPathSet=false;
            bool isHOFPathSet=false;
            int startFrameSet=true;
            size_t hog_outputbytes;
            size_t hof_outputbytes;
            bool isInitialized=false;
            HOGHOF(); //(QWidget *parent);
    
         
            HOGParameters HOGParams;
            HOFParameters HOFParams;
            CropParams Cropparams;
            HOGContext *hog_ctx;
            HOFContext *hof_ctx;
            HOGImage img;
            HOGShape hog_shape;
            HOFShape hof_shape;
            
            std::vector<float> hog_out;
            std::vector<float> hof_out; // jaaba output
            LockableQueue<vector<float>> hog_out_past;
            LockableQueue<vector<float>> hof_out_past;// window storing windowed past features
            std::vector<float> hog_out_avg;
            std::vector<float> hof_out_avg; // moving average features
            std::vector<float>hog_out_skip;
            std::vector<float>hof_out_skip;//features for skip frames


            //QJsonObject loadParams(const string& param_file);
            void loadHOGParams();
            void loadHOFParams();
            void loadCropParams();
            void loadImageParams(int img_width, int img_height);
            void setLastInput();
            void initHOGHOF(int img_height, int img_width);
            void genFeatures(int frame);
            void initialize_HOGHOFParams();
            //void genFeatures();
            /*void averageWindowFeatures(vector<float>& hog_feat, vector<float>& hof_feat,
                vector<float>& hog_feat_avg, vector<float>& hof_feat_avg,
                std::queue<vector<float>>& hog_feat_past, std::queue<vector<float>>& hof_feat_past,
                int frameCount, int window_size);*/
            void averageWindowFeatures(int window_size, int frameCount, int isSkip);
            void resetHOGHOFVec();

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
