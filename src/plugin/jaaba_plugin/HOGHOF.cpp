#include "HOGHOF.hpp"
#include <algorithm>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <QDebug>

#include <iostream>

namespace bias 
{

    HOGHOF::HOGHOF(QWidget *parent): QDialog(parent)
    { 

        initialize();      
    
    }



    void HOGHOF::initialize() 
    {

        isHOGPathSet=false;
        isHOFPathSet=false;
        startFrameSet=true;
        isInitialized=false;

    }


    void HOGHOF::loadHOGParams() 
    { 

        QJsonObject obj = loadParams(HOGParam_file);
        copytoHOGParams(obj);
        isHOGPathSet = true;

    }


    void HOGHOF::loadHOFParams() 
    {

        QJsonObject obj = loadParams(HOFParam_file);
        copytoHOFParams(obj);
        isHOFPathSet = true;

    }


    void HOGHOF::loadCropParams() 
    {

        QJsonObject obj = loadParams(CropParam_file);
        copytoCropParams(obj);

    }


    void HOGHOF::loadImageParams(int img_width, int img_height) 
    {

        img.w = img_width;
        img.h = img_height;
        img.pitch = img_width;
        img.type = hog_f32;
        img.buf = nullptr;

        HOFParams.input.w = img_width;
        HOFParams.input.h = img_height;
        HOFParams.input.pitch = img_width;
	 
    }


    //void HOGHOF::genFeatures() 
    //{

        //std::cout << this->hog_ctx << std::endl;
        //if(this->hog_ctx==nullptr) 
        //size_t hog_outputbytes = HOGOutputByteCount(this->hog_ctx);
        //size_t hof_outputbytes = HOFOutputByteCount(this->hof_ctx);
        
        //float* tmp_hog = (float*)malloc(hog_outputbytes);
        //float* tmp_hof = (float*)malloc(hof_outputbytes);

        //std::cout << " frame 1" << std::endl;
        //HOGCompute(this->hog_ctx,this->img);
        //HOGOutputCopy(this->hog_ctx, tmp_hog, hog_outputbytes);


    //}


    /*void HOGHOF::genFeatures(QString vidname, QString CropFile) 
    {


	    bias::videoBackend vid(vidname) ;
	    cv::VideoCapture capture = vid.videoCapObject(vid);

	    std::string fname = vidname.toStdString();
	    int num_frames = vid.getNumFrames(capture);
	    int height = vid.getImageHeight(capture);
	    int width =  vid.getImageWidth(capture);
	    float fps = capture.get(cv::CAP_PROP_FPS);

	    // Parse HOG/HOF/Crop Params
	    loadHOGParams();
	    loadHOFParams();
	    HOFParams.input.w = width;
	    HOFParams.input.h = height;
	    HOFParams.input.pitch = width;
	    loadImageParams(width, height);
	    loadCropParams(CropFile);

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
	    while(frame < num_frames) {

		//capture frame and convert to grey
		cur_frame = vid.getImage(capture);

		//convert to Float and normalize
		vid.convertImagetoFloat(cur_frame);
		img.buf = cur_frame.ptr<float>(0);

		//Compute and copy HOG/HOF      
		HOFCompute(&hof_ctx, img.buf, hof_f32); // call to compute and copy is asynchronous
		HOFOutputCopy(&hof_ctx, tmp_hof, hof_outputbytes); // should be called one after 
								   // the other to get correct answer
								     

		HOGCompute(&hog_ctx, img);
		HOGOutputCopy(&hog_ctx, tmp_hog, hog_outputbytes);

		copy_features1d(frame, hog_num_elements, hog_out, tmp_hog);
		if(frame > 0)
		    copy_features1d(frame-1, hof_num_elements, hof_out, tmp_hof);

		frame++;

	    }

	    vid.releaseCapObject(capture) ;
	    HOFTeardown(&hof_ctx);
	    HOGTeardown(&hog_ctx);
	    free(tmp_hog);
	    free(tmp_hof);

    }*/


    void HOGHOF::copytoHOGParams(QJsonObject& obj) 
    {

        QJsonValue value;
      	foreach(const QString& key, obj.keys()) 
        {

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


    void HOGHOF::copytoHOFParams(QJsonObject& obj) 
    {

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
		if(!Cropparams.crop_flag) 
		{  
			// if croping is not enabled
			Cropparams.interest_pnts = nullptr;
			Cropparams.ncells = 0; 
			Cropparams.npatches = 0;      
		}
    }


    void HOGHOF::allocateCrop(int sz) 
    {

		Cropparams.interest_pnts = (int*)malloc(2*sz*sizeof(int));

    }


    int HOGHOF::copyValueInt(QJsonObject& ob, 
				QString subobj_key) 
    {

		QJsonValue value = ob.value(subobj_key);
		if (value.isString())
			return (value.toString().toInt());
		else
			return 0;
    }


    float HOGHOF::copyValueFloat(QJsonObject& ob, 
				QString subobj_key) 
    {

		QJsonValue value = ob.value(subobj_key);
		if(value.isString())
			return (value.toString().toFloat());
		else
			return 0.0;
    }


    void HOGHOF::setLastInput() {

        std::cout << "vid HOGHOF" << std::endl;
        HOFSetLastInput(hof_ctx);

    }


    /*void HOGHOF::allocateHOGoutput(float* out, HOGContext* hog_init) {

	    size_t hog_nbytes = HOGOutputByteCount(hog_init);
	    out = (float*)malloc(hog_nbytes);

    }


    void HOGHOF::allocateHOFoutput(float* out, const HOFContext* hof_init) {

	    size_t hof_nbytes = HOFOutputByteCount(hof_init);
            out = (float*)malloc(hof_nbytes);

    }*/

}
