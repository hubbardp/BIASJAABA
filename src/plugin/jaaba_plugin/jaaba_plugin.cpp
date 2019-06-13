#include "jaaba_plugin.hpp"
#include "image_label.hpp"
//#include "camera_window.hpp"
//#include "image_grabber.hpp"
#include <QMessageBox>
#include <iostream>
#include <QDebug>
#include <cuda_runtime.h>
#include <opencv2/highgui/highgui.hpp>
#include "video_utils.hpp"

namespace bias {

    //Public static variables 
    const QString JaabaPlugin::PLUGIN_NAME = QString("jaabaPlugin");
    const QString JaabaPlugin::PLUGIN_DISPLAY_NAME = QString("Jaaba Plugin");


    // Public Methods
    JaabaPlugin::JaabaPlugin(int numberOfCameras, QWidget *parent) : BiasPlugin(parent)
    {

        nviews_ = numberOfCameras;
        setupUi(this);
        connectWidgets();
        initialize();
    }


    QString JaabaPlugin::getName()
    {
        return PLUGIN_NAME;
    }

    
    QString JaabaPlugin::getDisplayName()
    {
        return PLUGIN_DISPLAY_NAME;
    }

    
    cv::Mat JaabaPlugin::getCurrentImage()
    {
        acquireLock();
        cv::Mat currentImageCopy = currentImage_.clone();
        releaseLock();
        return currentImageCopy;
    }


    void JaabaPlugin::stop() 
    {

        if(sideRadioButtonPtr_->isChecked())
        {
            if(HOGHOF_side->isHOGPathSet && HOGHOF_side->isHOFPathSet)
            {

                std::cout << "inside" << std::endl;
                HOFTeardown(HOGHOF_side->hof_ctx);
                HOGTeardown(HOGHOF_side->hog_ctx);

            }
        }
        
        if(frontRadioButtonPtr_->isChecked())
        {

            if(HOGHOF_front->isHOGPathSet && HOGHOF_front->isHOFPathSet)
            {

                HOFTeardown(HOGHOF_front->hof_ctx);
                HOGTeardown(HOGHOF_front->hog_ctx);
            }
        }
    }


    void JaabaPlugin::initHOGHOF(QPointer<HOGHOF> hoghof)
    {
 
        hoghof->loadImageParams(currentImage_.cols, currentImage_.rows);
        struct HOGContext hogctx = HOGInitialize(logger, hoghof->HOGParams, currentImage_.cols, currentImage_.rows, hoghof->Cropparams);
        struct HOFContext hofctx = HOFInitialize(logger, hoghof->HOFParams, hoghof->Cropparams);
        hoghof->hog_ctx = (HOGContext*)malloc(sizeof(hogctx));
        hoghof->hof_ctx = (HOFContext*)malloc(sizeof(hofctx));
        memcpy(hoghof->hog_ctx, &hogctx, sizeof(hogctx));
        memcpy(hoghof->hof_ctx, &hofctx, sizeof(hofctx));
        hoghof->startFrameSet = false;
        
        //allocate output bytes HOG/HOF per frame 
        hoghof->hog_outputbytes = HOGOutputByteCount(hoghof->hog_ctx);
        hoghof->hof_outputbytes = HOFOutputByteCount(hoghof->hof_ctx);

        //output shape 
        struct HOGFeatureDims hogshape;
        HOGOutputShape(&hogctx, &hogshape);
        struct HOGFeatureDims hofshape;
        HOFOutputShape(&hofctx, &hofshape);
        hoghof->hog_shape = hogshape;
        hoghof->hof_shape = hofshape;
        hoghof->hog_out.resize(hoghof->hog_shape.x * hoghof->hog_shape.y * hoghof->hog_shape.bin);
        hoghof->hof_out.resize(hoghof->hof_shape.x * hoghof->hof_shape.y * hoghof->hof_shape.bin);
    }


    void JaabaPlugin::genFeatures(QPointer<HOGHOF> hoghof,int frame)
    {

        float* tmp_hof = (float*)malloc(hoghof->hof_outputbytes);
        float* tmp_hog = (float*)malloc(hoghof->hog_outputbytes);
        size_t hog_num_elements = hoghof->hog_shape.x * hoghof->hog_shape.y * hoghof->hog_shape.bin;
        size_t hof_num_elements = hoghof->hof_shape.x * hoghof->hof_shape.y * hoghof->hof_shape.bin;

        //Compute and copy HOG/HOF
        
        HOFCompute(hoghof->hof_ctx, hoghof->img.buf, hof_f32); // call to compute and copy is asynchronous
        HOFOutputCopy(hoghof->hof_ctx, hoghof->hof_out.data(), hoghof->hof_outputbytes); // should be called one after 
                                                           // the other to get correct answer
        HOGCompute(hoghof->hog_ctx, hoghof->img);
        HOGOutputCopy(hoghof->hog_ctx, hoghof->hog_out.data(), hoghof->hog_outputbytes);
        

        if(save) { 
            createh5("./out_feat/hoghof_", frame, 1, hog_num_elements, hof_num_elements, hoghof->hog_out, hoghof->hof_out);
        }

    }

              
    void JaabaPlugin::processFrames(QList<StampedImage> frameList)
    {

        StampedImage latestFrame = frameList.back();
        int frame = latestFrame.frameCount;
        frameList.clear();
 
        //get current image
        cv::Mat currentImageFloat;
        cv::Mat workingImage = latestFrame.image.clone();

        // initialize gpu HOGHOF Context
        if((workingImage.rows != 0) && (workingImage.cols != 0)) 
        {
         
            acquireLock();
            currentImage_ = workingImage;
            releaseLock();
            currentImage_.convertTo(currentImageFloat, CV_32FC1);
            currentImageFloat = currentImageFloat / 255;
            HOGHOF_side->img.buf = currentImageFloat.ptr<float>(0);
                     
            
            if(sideRadioButtonPtr_->isChecked())             
            {
             
                if(HOGHOF_side->isHOGPathSet && HOGHOF_side->isHOFPathSet) 
                {
 
                    if(HOGHOF_side->startFrameSet)
                    {
                        initHOGHOF(HOGHOF_side);
                        
                    }

                } 
               
                if(detectStarted)
                {
                    genFeatures(HOGHOF_side, frame);
     
                }
            }
             
         
            if(frontRadioButtonPtr_->isChecked())
            {
 
                if(HOGHOF_front->isHOGPathSet && HOGHOF_front->isHOFPathSet)
                {

                    if(HOGHOF_front->startFrameSet)
                    {
                        //initialize HOGHOF
                        initHOGHOF(HOGHOF_front);
                    }

                }

                if(detectStarted)
                {  
                    genFeatures(HOGHOF_front, frame);  
                }                   
            }
        }
    }

    
    void JaabaPlugin::initialize()
    {
        
        updateWidgetsOnLoad();
        setupHOGHOF();

    }


    void JaabaPlugin::connectWidgets()
    {

        connect(
            sideRadioButtonPtr_,
            SIGNAL(stateChanged(int)),
            this,
            SLOT(SideViewCheckBoxChanged(int))
           );
 
        connect(
            frontRadioButtonPtr_,
            SIGNAL(stateChanged(int)),
            this,
            SLOT(FrontViewCheckBoxChanged(int))
           );   

        connect(
            reloadPushButtonPtr_,
            SIGNAL(clicked()),
            this,
            SLOT(reloadButtonPressed())
           );

        connect(
            detectButtonPtr_,
            SIGNAL(clicked()),
            this,
            SLOT(detectClicked())
           );

        connect(
            saveButtonPtr_,
            SIGNAL(clicked()),
            this,
            SLOT(saveClicked())
           );

  
    }

    
    void JaabaPlugin::setupHOGHOF()
    {
     
        if(sideRadioButtonPtr_->isChecked())
        {

            HOGHOF *hoghofside = new HOGHOF(this);  
	    HOGHOF_side = hoghofside;
	    HOGHOF_side->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            HOGHOF_side->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            HOGHOF_side->CropParam_file = pathtodir_->placeholderText() + CropSideParamFilePtr_->placeholderText();
            HOGHOF_side->loadHOGParams();
            HOGHOF_side->loadHOFParams();
            HOGHOF_side->loadCropParams(); 

        }

        if(frontRadioButtonPtr_->isChecked()) 
        {
             
            HOGHOF *hoghoffront = new HOGHOF(this);
      	    HOGHOF_front = hoghoffront;
            HOGHOF_front->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            HOGHOF_front->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            HOGHOF_front->CropParam_file = pathtodir_->placeholderText() + CropFrontParamFilePtr_->placeholderText();
            HOGHOF_front->loadHOGParams();
            HOGHOF_front->loadHOFParams();
            HOGHOF_front->loadCropParams();
 
        }

    }

    
    int JaabaPlugin::getNumberofViews() 
    {

        return nviews_;

    }

   
    void JaabaPlugin::updateWidgetsOnLoad() 
    {

        sideRadioButtonPtr_ -> setChecked(false);
        frontRadioButtonPtr_ -> setChecked(false);
        detectButtonPtr_ ->setEnabled(false);
        saveButtonPtr_->setEnabled(false);
        detectStarted = false;
        save = false;
        checkviews();

    }
   

    void JaabaPlugin::SideViewCheckBoxChanged(int state)
    {
			
        if (state == Qt::Checked)
	{
	    sideRadioButtonPtr_ -> setChecked(true);
	}
	else
	{  
            sideRadioButtonPtr_ -> setChecked(false);
	}
        checkviews();
        setupHOGHOF();
        detectEnabled();

    }


    void JaabaPlugin::FrontViewCheckBoxChanged(int state)
    {
            
        if (state == Qt::Checked)
        {   
            frontRadioButtonPtr_ -> setChecked(true);
        }
        else
        {   
            frontRadioButtonPtr_ -> setChecked(false);
        }
        checkviews();
        setupHOGHOF();
        detectEnabled();

    }


    void JaabaPlugin::detectClicked() 
    {
        
        if(!detectStarted) 
        {
            detectButtonPtr_->setText(QString("Stop Detecting"));
            detectStarted = true;

        } else {

            detectButtonPtr_->setText(QString("Detect"));
            detectStarted = false;
            save = false;

        }

    }

    void JaabaPlugin::saveClicked()
    {

        if(!save) { 

            saveButtonPtr_->setText(QString("Stop Saving"));
            save = true;
 
        } else {

            saveButtonPtr_->setText(QString("Save"));
            save = false;

        }       

    }

    
    void JaabaPlugin::detectEnabled() 
    {

        if(sideRadioButtonPtr_->isChecked())
        {
            if(HOGHOF_side->isHOGPathSet && HOGHOF_side->isHOFPathSet)
            {
                 detectButtonPtr_->setEnabled(true);
                 saveButtonPtr_->setEnabled(true);
            }
        
        }

        if(frontRadioButtonPtr_->isChecked())
        {
            if(HOGHOF_front->isHOGPathSet && HOGHOF_front->isHOFPathSet)
            {
                 detectButtonPtr_->setEnabled(true);
                 saveButtonPtr_->setEnabled(true);
            }

        }

    }    
    

    void JaabaPlugin::reloadButtonPressed()
    {

        pathtodir_->setPlaceholderText(pathtodir_->displayText());
        // load side HOGHOFParams if side view checked
        if(HOGHOF_side == nullptr) 
        {
            setupHOGHOF();

        } else {

            HOGHOF_side->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            HOGHOF_side->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            HOGHOF_side->CropParam_file = pathtodir_->placeholderText() + CropSideParamFilePtr_->placeholderText();
            HOGHOF_side->loadHOGParams();
            HOGHOF_side->loadHOFParams();
            HOGHOF_side->loadCropParams();
            
        }

        // load front HOGHOFParams if front view checked
        if(HOGHOF_front == nullptr)
        {
            setupHOGHOF();

        } else {

            HOGHOF_front->HOGParam_file = pathtodir_->placeholderText() + HOGParamFilePtr_->placeholderText();
            HOGHOF_front->HOFParam_file = pathtodir_->placeholderText() + HOFParamFilePtr_->placeholderText();
            HOGHOF_front->CropParam_file = pathtodir_->placeholderText() + CropFrontParamFilePtr_->placeholderText();
            HOGHOF_front->loadHOGParams();
            HOGHOF_front->loadHOFParams();
            HOGHOF_front->loadCropParams();

        }
        detectEnabled();
    }


    void JaabaPlugin::checkviews() 
    {

        // if both views are checked
        if(sideRadioButtonPtr_->isChecked() && frontRadioButtonPtr_->isChecked()) 
        {
        
            if(nviews_ != 2) 
            {

                QString errMsgText = QString("Number of cameras not equal to number of views ");
                QString errMsgTitle = QString("Number of Views error");
                QMessageBox::critical(this, errMsgTitle, errMsgText);

            }
          
        }

        // Only side view checked
        if(sideRadioButtonPtr_->isChecked() && ~frontRadioButtonPtr_->isChecked()) 
        {
            if(nviews_ != 1)
            {

                QString errMsgText = QString("Number of cameras not equal to number of views ");
                QString errMsgTitle = QString("Number of Views error");
                QMessageBox::critical(this, errMsgTitle, errMsgText);

            }

        }

        // Only front View checked
        if(~sideRadioButtonPtr_->isChecked() && frontRadioButtonPtr_->isChecked()) 
        {
            if(nviews_ != 1)
            {

                QString errMsgText = QString("Number of cameras not equal to numver of views ");
                QString errMsgTitle = QString("Number of Views error");
                QMessageBox::critical(this, errMsgTitle, errMsgText);

            }
        }
    }

    // TEST DEVELOPMENT 
    
    void JaabaPlugin::copy_features1d(int frame_num, int num_elements, std::vector<float> &vec_feat, 
                                      float* array_feat) 
    {        
        // row to add is num_elements * frame_num
        int start_idx = frame_num * num_elements;
        for(int i = 0; i < num_elements; i++) {
            vec_feat[i + start_idx] = array_feat[i];
        }
    }


    int JaabaPlugin::createh5(std::string filename, int frame_num,
                              int num_frames, int hog_elements, int hof_elements,
                              std::vector<float> hog, std::vector<float> hof) 
    {
                 
        std::string out_file = filename + std::to_string(frame_num) + ".h5";
        std::cout << out_file << std::endl;
        H5::H5File file(out_file.c_str(), H5F_ACC_TRUNC);

        // Create 4 datasets
        create_dataset(file, "hog", hog, num_frames, hog_elements);
        create_dataset(file, "hof", hof, num_frames, hof_elements);
    
        file.close();

        return 0;
    }

     
    void JaabaPlugin::create_dataset(H5::H5File& file, std::string key,
        std::vector<float> features, int num_frames, int num_elements) 
    {
  
        hsize_t dims[2];
        dims[0] = num_frames;
        dims[1] = num_elements;
        H5::DataSpace dataspace(2, dims);
        H5::DataSet dataset = file.createDataSet(key, H5::PredType::IEEE_F32LE, dataspace);

        dataset.write(&features.data()[0], H5::PredType::IEEE_F32LE);

        dataset.close();
        dataspace.close();
    }

    
    void JaabaPlugin::write_output(std::string file,float* out_img, unsigned w, unsigned h) 
    {

        std::ofstream x_out;
        x_out.open(file.c_str());

        // write hist output to csv file
        for(int i = 0;i < h;i++){
            for(int j = 0; j < w;j++){
                x_out << out_img[i*w + j];
                if(j != w-1 || i != h-1)
                    x_out << ",";
            }
        }
    }


    void JaabaPlugin::read_image(std::string filename, float* img, int w, int h)
    {

	int count_row = 0;
	int count_col = 0;
	std::string lin;
	std::ifstream x_in(filename);

	// read image input from a csv file
	if(x_in) 
	{
	    while(getline(x_in, lin))
	    {
		std::stringstream iss(lin);
		std::string result;
		count_row =0;
		while(std::getline(iss, result, ','))
		{
		    img[count_col*w+count_row] = atof(result.c_str());
		    count_row=count_row+1;
		}
		count_col=count_col+1;
	    }

	} else {

	     std::cout << "File not present.Enter a valid filename." << std::endl;
    	     exit(1);
    	}

    }

}




