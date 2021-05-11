
#include "HOGHOF.hpp"
#include <string>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/opencv.h>
#include <QDebug>
#include <fstream>
#include <iostream>
#include "timer.h"
#include "exception.hpp"

using namespace bias; 

HOGHOF::HOGHOF() {}


void HOGHOF::loadHOGParams() { 

    RtnStatus rtnStatus;
    QString errMsgTitle("Load Parameter Error");

    QFile parameterFile(param_files.HOGParam_file);
    if (!parameterFile.exists())
    {
        QString errMsgText = QString("Parameter file, %1").arg(param_files.HOGParam_file);
        errMsgText += QString(", does not exist - using default values");
        qDebug() << errMsgText;
        qDebug() << HOGParam_file;
        return;
    }

    bool ok = parameterFile.open(QIODevice::ReadOnly);
    if (!ok)
    {
        QString errMsgText = QString("Unable to open parameter file %1").arg(param_files.HOGParam_file);
        errMsgText += QString(" - using default values");
        qDebug() << errMsgText;
        qDebug() << HOGParam_file;
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

    QFile parameterFile(param_files.HOFParam_file);
    if (!parameterFile.exists())
    {
        QString errMsgText = QString("Parameter file, %1").arg(param_files.HOFParam_file);
        errMsgText += QString(", does not exist - using default values");
        qDebug() << errMsgText;
        qDebug() << HOFParam_file;
        return;
    }

    bool ok = parameterFile.open(QIODevice::ReadOnly);
    if (!ok)
    {
        QString errMsgText = QString("Unable to open parameter file %1").arg(param_files.HOFParam_file);
        errMsgText += QString(" - using default values");
        qDebug() << errMsgText;
        qDebug() << HOFParam_file;
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

    QFile parameterFile(param_files.CropParam_file);
    if (!parameterFile.exists())
    {
        QString errMsgText = QString("Parameter file, %1").arg(param_files.CropParam_file);
        errMsgText += QString(", does not exist - using default values");
        qDebug() << errMsgText;
        return;
    }

    bool ok = parameterFile.open(QIODevice::ReadOnly);
    if (!ok)
    {
        QString errMsgText = QString("Unable to open parameter file %1").arg(param_files.CropParam_file);
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

void HOGHOF::initialize_params(Params& param_file) {

    param_files = param_file;
    loadHOGParams();
    loadHOFParams();
    loadCropParams();
    HOFParams.input.w = 300;
    HOFParams.input.h = 300;
    HOFParams.input.pitch = 300;
    loadImageParams(HOFParams.input.w, HOFParams.input.h);
}

void HOGHOF::initialize_vidparams(bias::videoBackend& vid, Params& param_file) {
  
    capture = vid.videoCapObject();

    //getvideo params
    int num_frames = vid.getNumFrames(capture);
    int height = vid.getImageHeight(capture);
    int width = vid.getImageWidth(capture);
    float fps = capture.get(cv::CAP_PROP_FPS);
    param_files = param_file;
    
    loadHOGParams();
    loadHOFParams();
    HOFParams.input.w = width;
    HOFParams.input.h = height;
    HOFParams.input.pitch = width;
    loadImageParams(width, height);
    loadCropParams();

}


void HOGHOF::getvid_frame(bias::videoBackend& vid) {

    cv::Mat cur_frame;
    cur_frame = vid.getImage(capture);
    //convert to Float and normalize
    vid.convertImagetoFloat(cur_frame);
    img.buf = cur_frame.ptr<float>(0);

}

void HOGHOF::initializeHOGHOF(int& width, int& height, 
                              int& num_frames) {

    // create input HOGContext / HOFConntext
    struct HOGContext hog_ctx_ = HOGInitialize(logger, HOGParams, width, height, Cropparams);
    struct HOFContext hof_ctx_ = HOFInitialize(logger, HOFParams, Cropparams);

    hog_ctx = (HOGContext*)malloc(sizeof(hog_ctx_));
    hof_ctx = (HOFContext*)malloc(sizeof(hof_ctx_));
    memcpy(hog_ctx, &hog_ctx_, sizeof(hog_ctx_));
    memcpy(hof_ctx, &hof_ctx_, sizeof(hof_ctx_));

    struct HOGFeatureDims hogshape;
    HOGOutputShape(hog_ctx, &hogshape);
    struct HOGFeatureDims hofshape;
    HOFOutputShape(hof_ctx, &hofshape);
    hog_shape = hogshape;
    hof_shape = hofshape;

    int hof_num_elements = hof_shape.x * hof_shape.y * hof_shape.bin;
    int hog_num_elements = hog_shape.x * hog_shape.y * hog_shape.bin;
    hof_out.resize(hof_num_elements);
    hog_out.resize(hog_num_elements);

}

void  HOGHOF::process_vidFrame(int frame) {

    
    size_t hog_outputbytes = HOGOutputByteCount(hog_ctx);
    size_t hof_outputbytes = HOFOutputByteCount(hof_ctx);

    int hof_num_elements = hof_shape.x * hof_shape.y * hof_shape.bin;
    int hog_num_elements = hog_shape.x * hog_shape.y * hog_shape.bin;

    HOFCompute(hof_ctx, img.buf, hof_f32); // call to compute and copy is asynchronous
    HOFOutputCopy(hof_ctx, hof_out.data() , hof_outputbytes); // should be called one after 
                                                       // the other to get correct answer                                                             
    HOGCompute(hog_ctx, img);
    HOGOutputCopy(hog_ctx, hog_out.data() , hog_outputbytes);

    /*if (frame > 0) {
        copy_features1d(frame - 1, hog_num_elements, hog_out, tmp_hog);
        copy_features1d(frame - 1, hof_num_elements, hof_out, tmp_hof);
    }*/

}


void HOGHOF::process_camFrame() {

    size_t hog_outputbytes = HOGOutputByteCount(hog_ctx);
    size_t hof_outputbytes = HOFOutputByteCount(hof_ctx);

    int hof_num_elements = hof_shape.x * hof_shape.y * hof_shape.bin;
    int hog_num_elements = hog_shape.x * hog_shape.y * hog_shape.bin;

    HOFCompute(hof_ctx, img.buf, hof_f32); // call to compute and copy is asynchronous
    HOFOutputCopy(hof_ctx, hof_out.data(), hof_outputbytes); // should be called one after 
                                                       // the other to get correct answer                                                             
    HOGCompute(hog_ctx, img);
    HOGOutputCopy(hog_ctx, hog_out.data(), hog_outputbytes);

}

void HOGHOF::deInitializeHOGHOF() {

    HOFTeardown(hof_ctx);
    HOGTeardown(hog_ctx);

   
}

void HOGHOF::genFeatures(QString vidname, QString& CropFile) {


    bias::videoBackend vid(vidname);
    cv::VideoCapture capture = vid.videoCapObject();

    std::string fname = vidname.toStdString();
    int num_frames = vid.getNumFrames(capture);
    int height = vid.getImageHeight(capture);
    int width =  vid.getImageWidth(capture);
    float fps = capture.get(cv::CAP_PROP_FPS);
    std::cout << num_frames << " " << fps <<  std::endl;

    // Parse HOG/HOF/Crop Params
    loadHOGParams();
    loadHOFParams();
    HOFParams.input.w = width;
    HOFParams.input.h = height;
    HOFParams.input.pitch = width;
    loadImageParams(width, height);
    loadCropParams();

    // create input HOGContext / HOFConntext
    struct HOGContext hog_ctx = HOGInitialize(logger, HOGParams, width, height, Cropparams);
    struct HOFContext hof_ctx = HOFInitialize(logger, HOFParams, Cropparams);

    //allocate output HOG/HOF per frame 
    size_t hog_outputbytes = HOGOutputByteCount(&hog_ctx);
    size_t hof_outputbytes = HOFOutputByteCount(&hof_ctx);
    float* tmp_hog = (float*)malloc(hog_outputbytes);
    float* tmp_hof = (float*)malloc(hof_outputbytes);

    struct HOGFeatureDims hogshape;
    HOGOutputShape(&hog_ctx, &hogshape);
    struct HOGFeatureDims hofshape;
    HOFOutputShape(&hof_ctx, &hofshape);

    hog_shape = hogshape;
    hof_shape = hofshape;
    int hof_num_elements = hof_shape.x * hof_shape.y * hof_shape.bin;
    int hog_num_elements = hog_shape.x * hog_shape.y * hog_shape.bin;
    hof_out.resize(num_frames*hof_num_elements,0.0);
    hog_out.resize(num_frames*hog_num_elements,0.0);

    cv::Mat cur_frame;
    int frame = 0;
    GpuTimer timer1;
    std::vector<float>time_result;

    while(frame < num_frames) {

        //capture frame and convert to grey
        //timer1.Start();
        cur_frame = vid.getImage(capture);
        //convert to Float and normalize
        vid.convertImagetoFloat(cur_frame);
        img.buf = cur_frame.ptr<float>(0);
        //write_output("./beh_feat/beh_" + std::to_string(frame) + ".csv", cur_frame.ptr<float>(0), width, height);


        //Compute and copy HOG/HOF     

        timer1.Start();
        HOFCompute(&hof_ctx, img.buf, hof_f32); // call to compute and copy is asynchronous
        HOFOutputCopy(&hof_ctx, tmp_hof, hof_outputbytes); // should be called one after 
                                                           // the other to get correct answer                                                             
        HOGCompute(&hog_ctx, img);
        HOGOutputCopy(&hog_ctx, tmp_hog, hog_outputbytes);

      
        timer1.Stop();
        time_result.push_back(timer1.Elapsed()/1000);
        //std::cout << "time elapsed: " << timer1.Elapsed()/1000 << std::endl;       

        //copy_features1d(frame, hog_num_elements, hog_out, tmp_hog);
        if(frame > 0) {
            copy_features1d(frame-1, hog_num_elements, hog_out, tmp_hog);
            copy_features1d(frame-1, hof_num_elements, hof_out, tmp_hof);
        }

        frame++;

    }
    //write_time("offline_time.csv",num_frames,time_result);

    vid.releaseCapObject(capture) ;
    HOFTeardown(&hof_ctx);
    HOGTeardown(&hog_ctx);
    free(tmp_hog);
    free(tmp_hof);

}

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
    try{
		if(value.isString())
            return (value.toString().toInt());

    }catch (RuntimeError &runtimeError) {

		runtimeError.what();
	    printf("Not a String Value");
    }

	return 0;
                        
}


float HOGHOF::copyValueFloat(QJsonObject& ob, 
                            QString subobj_key) {

    QJsonValue value = ob.value(subobj_key);
	try{
		if(value.isString())
		    return (value.toString().toFloat());

	}catch(RuntimeError &runtimeError) {
	
		runtimeError.what();
		printf( "Not a String Value");
	}
	 
	return 0.0;
}


//Test

/*void HOGHOF::write_time(std::string file, int framenum, std::vector<float> timeVec)
{

    std::ofstream x_out;
    x_out.open(file.c_str(), std::ios_base::app);

    for(int frame_id= 0; frame_id < framenum; frame_id++)
    {

        x_out << frame_id << "," << timeVec[frame_id] << "\n";

    }

}*/


/*void HOGHOF::allocateHOGoutput(float* out, HOGContext* hog_init) {

    size_t hog_nbytes = HOGOutputByteCount(hog_init);
    out = (float*)malloc(hog_nbytes);

}


void HOGHOF::allocateHOFoutput(float* out, const HOFContext* hof_init) {

    size_t hof_nbytes = HOFOutputByteCount(hof_init);
    out = (float*)malloc(hof_nbytes);

}*/

HOGHOF::~HOGHOF() {


}
