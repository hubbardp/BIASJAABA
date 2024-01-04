#include "HOGHOF.hpp"
#include <algorithm>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <QDebug>
#include <cuda_runtime_api.h>
#include <iostream>

namespace bias 
{

    HOGHOF::HOGHOF()//(QWidget *parent): QDialog(parent)
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

        std::cout << "Trying to load HOG parameters from " << HOGParam_file << std::endl;

        QJsonObject obj = loadParams(HOGParam_file);

        if (obj.contains("hog"))
        {
            std::cout << "Found hog within parameters" << std::endl;
            QJsonValue value = obj.value("hog");
            if (value.isObject()) {
                copytoHOGParams(value.toObject());
                isHOGPathSet = true;
            }
        }
        else {
            copytoHOGParams(obj);
            isHOGPathSet = true;
        }

    }


    void HOGHOF::loadHOFParams() 
    {

        QJsonObject obj = loadParams(HOFParam_file);
        if (obj.contains("hof"))
        {
            std::cout << "Found hof within parameters" << std::endl;
            QJsonValue value = obj.value("hof");
            if (value.isObject()) {
                copytoHOFParams(value.toObject());
                isHOFPathSet = true;
            }
        }
        else {
            copytoHOFParams(obj);
            isHOFPathSet = true;
        }
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

    void HOGHOF::initHOGHOF(int img_height, int img_width)
    {

        std::cout << "Gpu Initialized - - - " << std::endl;
        int nDevices;
        cudaError_t err = cudaGetDeviceCount(&nDevices);
        if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
        loadImageParams(img_width, img_height);
        hog_ctx = HOGInitialize(logger, HOGParams, img_width, img_height, Cropparams);
        hof_ctx = HOFInitialize(logger, HOFParams, Cropparams);       

        //hog_ctx = (HOGContext*)malloc(sizeof(hogctx));
        //hof_ctx = (HOFContext*)malloc(sizeof(hofctx));
        
        //memcpy(hog_ctx, &hogctx, sizeof(hogctx));
        //memcpy(hof_ctx, &hofctx, sizeof(hofctx));
        //hoghof->startFrameSet = false;
        
        //allocate output bytes HOG/HOF per frame
        hog_outputbytes = HOGOutputByteCount(hog_ctx);
        hof_outputbytes = HOFOutputByteCount(hof_ctx);
        
        //output shape 
        struct HOGFeatureDims hogshape;
        HOGOutputShape(hog_ctx, &hogshape);
        struct HOGFeatureDims hofshape;
        HOFOutputShape(hof_ctx, &hofshape);
        hog_shape = hogshape;
        hof_shape = hofshape;

        size_t hog_num_elements = hog_shape.x * hog_shape.y * hog_shape.bin;
        size_t hof_num_elements = hof_shape.x * hof_shape.y * hof_shape.bin;
#if 0        
           std::cout << "hog_num_elements" << hog_num_elements << std::endl;
           std::cout << "hof_num_elements" << hof_num_elements << std::endl;
#endif
        hog_out.resize(hog_num_elements);
        hof_out.resize(hof_num_elements);
        hog_out_avg.resize(hog_num_elements, 0.0);
        hof_out_avg.resize(hof_num_elements, 0.0);
        hog_out_skip.resize(hog_num_elements, 0.0);
        hof_out_skip.resize(hof_num_elements, 0.0);

    }

    void HOGHOF::genFeatures(int frame)
    {
        
        // Test
        //HOGImage fake_img;
        //fake_img.w = 448;
        //fake_img.h = 290;
        //fake_img.pitch = 448;
        //fake_img.type = hog_f32;
        //fake_img.buf = nullptr;
        //cv::Mat fake_img_test = cv::Mat::zeros(cv::Size(448, 290), CV_32FC1);
        //fake_img.buf = fake_img_test.ptr<float>(0);

        //Compute and copy HOG/HOF

        HOFCompute(hof_ctx, img.buf, hof_f32); // call to compute and copy is asynchronous
        HOFOutputCopy(hof_ctx, hof_out.data(), hof_outputbytes); // should be called one after 
                                                           // the other to get correct answer
        HOGCompute(hog_ctx, img);
        HOGOutputCopy(hog_ctx, hog_out.data(), hog_outputbytes);

    }

    void HOGHOF::initialize_HOGHOFParams()
    {

        loadHOGParams();
        if (HOGParams.nbins == 0)
            printf("HOG NOT Initialzied");

        loadHOFParams();
        if (HOFParams.nbins == 0)
            printf("HOF NOT Initialzied");

        loadCropParams();
        if (Cropparams.ncells == 0)
            printf("CROP NOT Initialzied");

    }

    void HOGHOF::resetHOGHOFVec()
    {

        size_t hog_num_elements = hog_shape.x * hog_shape.y * hog_shape.bin;
        size_t hof_num_elements = hof_shape.x * hof_shape.y * hof_shape.bin;
        if (hog_out.size() != 0)
            fill(hog_out.begin(), hog_out.end(), 0.0);
        else
            std::cout << " jaaba hog out vector empty" << std::endl;

        if (hof_out.size() != 0)
            fill(hof_out.begin(), hof_out.end(), 0.0);
        else
            std::cout << " jaaba hof out vector empty" << std::endl;

        if (hog_out_avg.size() != 0)
            fill(hog_out_avg.begin(), hog_out_avg.end(), 0.0);
        else
            std::cout << " jaaba hog out avg vector empty" << std::endl;

        if (hof_out_avg.size() != 0)
            fill(hof_out_avg.begin(), hof_out_avg.end(), 0.0);
        else
            std::cout << " jaaba hof out avg vector empty" << std::endl;

        if (hog_out_skip.size() != 0)
            fill(hog_out_skip.begin(), hog_out_skip.end(), 0.0);
        else
            std::cout << "jaaba hog out skip vector empty" << std::endl;

        if (hof_out_skip.size() != 0)
            fill(hof_out_skip.begin(), hof_out_skip.end(), 0.0);
        else
            std::cout << "jaaba hof out skip vector empty" << std::endl;
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
            if (value.isString() && key == "nbins"){
                HOGParams.nbins = value.toString().toInt();

            }else if (value.isObject() && key == "cell") {
      
		        QJsonObject ob = value.toObject();
		        HOGParams.cell.h = copyValueInt(ob, "h");
		        HOGParams.cell.w = copyValueInt(ob, "w");

            }
            else {
                printf("Key %s missing in HOG params", key);
            }
        }
    }


    void HOGHOF::copytoHOFParams(QJsonObject& obj) 
    {

	    QJsonValue value;  
        foreach(const QString& key, obj.keys()) {

            value = obj.value(key);
            if (value.isString() && key == "nbins"){

                HOFParams.nbins = value.toString().toInt();

            }else if (value.isObject() && key == "cell") {
   
		        QJsonObject ob = value.toObject();
		        HOFParams.cell.h = copyValueInt(ob, "h");
		        HOFParams.cell.w = copyValueInt(ob, "w");           

            }
            else if(value.isObject() && key == "lk") {

	            QJsonObject ob = value.toObject();
	            foreach(const QString& key, ob.keys()) {

		            value = ob.value(key);
                    if (value.isString() && key == "threshold") {
                        HOFParams.lk.threshold = value.toString().toFloat();

                    }else if(value.isObject() && key == "sigma") {

                        QJsonObject ob = value.toObject();
                        HOFParams.lk.sigma.smoothing = copyValueFloat(ob, "smoothing");
                        HOFParams.lk.sigma.derivative = copyValueFloat(ob, "derivative");

                    }
                    else {
                        printf("Key %s missing in lk params", key);
                    }
		        }
            }
            else {
                printf("Key %s missing in HOF params", key);
            }
        }	       
    }


    void HOGHOF::copytoCropParams(QJsonObject& obj) {

		QJsonValue value;
        foreach(const QString& key, obj.keys()) {

            value = obj.value(key);
            if (value.isString() && key == "crop_flag") {
                Cropparams.crop_flag = value.toString().toInt();

            }else if (value.isString() && key == "ncells") {
                Cropparams.ncells = value.toString().toInt();

            }else if (value.isString() && key == "npatches") {
                
                Cropparams.npatches = value.toString().toInt();

            }else if (value.isObject() && key == "interest_pnts") {

                std::cout << " Interest points enetred" << std::endl;
                QJsonObject ob = value.toObject();
                allocateCrop(ob.size());
                int idx = 0;
                foreach(const QString& key, ob.keys()) {

                    value = ob.value(key);
                    if (value.isArray() && key == "food") {
                        QJsonArray arr = value.toArray();
                        Cropparams.interest_pnts[idx * 2 + 0] = arr[0].toInt();
                        Cropparams.interest_pnts[idx * 2 + 1] = arr[1].toInt();
                        std::cout << "idx " << idx << " ips x " << arr[0].toInt() << std::endl;
                        idx = idx + 1;
                        
                    }
                    else if (value.isArray() && key == "mouth") {
                        QJsonArray arr = value.toArray();
                        Cropparams.interest_pnts[idx * 2 + 0] = arr[0].toInt();
                        Cropparams.interest_pnts[idx * 2 + 1] = arr[1].toInt();
                        std::cout << "idx " << idx << " ips x " << arr[0].toInt() << std::endl;
                        idx = idx + 1;
                    }
                    else if (value.isArray() && key == "perch") {
                        QJsonArray arr = value.toArray();
                        Cropparams.interest_pnts[idx * 2 + 0] = arr[0].toInt();
                        Cropparams.interest_pnts[idx * 2 + 1] = arr[1].toInt();
                        std::cout << "idx " << idx << " ips x " << arr[0].toInt() << std::endl;
                        idx = idx + 1;
                    }
                    else {
                        printf("Key %s missing in Crop interest params", key);
                    }
                }
				/*QJsonArray arr = value.toArray();
				allocateCrop(arr.size());
				int idx = 0;
				foreach(const QJsonValue& id, arr) {       
		
					QJsonArray ips = id.toArray();
					Cropparams.interest_pnts[idx*2+ 0] = ips[0].toInt();
					Cropparams.interest_pnts[idx*2 + 1] = ips[1].toInt();
                    std::cout << "Read x " << ips[0].toInt() << std::endl;
					idx = idx + 1;
				}*/
                
            }
            else {
                printf("Key %s missing in Crop params", key);
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


    void HOGHOF::setLastInput() 
    {

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

    void HOGHOF::averageWindowFeatures(int window_size, int frameCount, int isSkip)
    {

        /*std::vector<float> hog_out = hoghof_obj->hog_out;
        std::vector<float> hof_out = hoghof_obj->hof_out;
        std::vector<float> hog_out_avg = hoghof_obj->hog_out_avg;
        std::vector<float> hof_out_avg = hoghof_obj->hof_out_avg;
        std::queue<vector<float>> hog_out_past = hoghof_obj->hog_out_past;
        std::queue<vector<float>> hof_out_past = hoghof_obj->hof_out_past;*/

        std::vector<float>hog_feat;
        std::vector<float>hof_feat;
        if (isSkip)
        {
            hog_feat = hog_out_skip;
            hof_feat = hof_out_skip;
        }
        else {
            hog_feat = hog_out;
            hof_feat = hof_out;
        }

        // average features over window size
        unsigned int feat_size = static_cast<unsigned int>(hog_out.size());
        transform(hog_feat.begin(), hog_feat.end(), hog_feat.begin(), [window_size](float &c) {return (c / window_size); });
        transform(hof_feat.begin(), hof_feat.end(), hof_feat.begin(), [window_size](float &c) {return (c / window_size); });
        //transform(hog_out.begin(), hog_out.end(), hog_out.begin(), [window_size](float &c) {return (c / window_size); });
        //transform(hof_out.begin(), hof_out.end(), hof_out.begin(), [window_size](float &c) {return (c / window_size); });

        for (int i = 0; i < feat_size; i++) {

            //hog_feat_avg[i] = hog_feat_avg[i] + hog_feat[i];
            //hof_feat_avg[i] = hof_feat_avg[i] + hof_feat[i];
            hog_out_avg[i] = hog_out_avg[i] + hog_feat[i];
            hof_out_avg[i] = hof_out_avg[i] + hof_feat[i];
        }

        //update moving average over the window
        if (frameCount >= window_size) {

            if ((hog_out_past.size() == window_size)
                && (hof_out_past.size() == window_size))
            {
                vector<float> hog_past = hog_out_past.front();
                vector<float> hof_past = hof_out_past.front();

                // subtract last window feature
                unsigned int feat_size = static_cast<unsigned int>(hog_past.size());
                for (unsigned int i = 0; i < feat_size; i++) {

                    //hog_feat_avg[i] -= hog_past[i];
                    //hof_feat_avg[i] -= hof_past[i];
                    hog_out_avg[i] -= hog_past[i];
                    hof_out_avg[i] -= hof_past[i];

                }

                //hog_feat_past.pop();
                //hof_feat_past.pop();
                hog_out_past.pop();
                hof_out_past.pop();

            }
        }

        // pushing the inciming feature average to moving window average queue
        //hog_feat_past.push(hog_feat);
        //hof_feat_past.push(hof_feat);
        hog_out_past.push(hog_feat);
        hof_out_past.push(hof_feat);

    }
}
