#include "beh_class.hpp"
#include <iostream>
#include <QDebug>

namespace bias {


    beh_class::beh_class(QWidget *parent): QDialog(parent) {}
  

    void beh_class::allocate_model() 
    {


        hsize_t dims_out[2] = {0};
        H5::Exception::dontPrint();
        try 
        {
           
            // open classfier file
            int rank, ndims;
	    H5::H5File file(this->classifier_file.toStdString(), H5F_ACC_RDONLY);

            // check the number of behaviors present
            int num_beh = beh.size();

            //initialize other arrays
            this->translated_index.resize(num_beh);
            this->flag.resize(num_beh);
            this->score.resize(num_beh,0);

            for(int nbeh =0;nbeh < num_beh;nbeh++)
            {
                if(pathExists(file.getId(), beh[nbeh]))
                {

                    try
                    {
                          
                        //allocate the model
                        H5::Group multbeh = file.openGroup(beh[nbeh]);
                        H5::DataSet dataset = multbeh.openDataSet(this->model_params[0]);
                        H5::DataSpace dataspace = dataset.getSpace();
                        ndims = dataspace.getSimpleExtentDims(dims_out,NULL);
                        this->model[nbeh].cls_alpha.resize(dims_out[0]);
                        this->model[nbeh].cls_dim.resize(dims_out[0]);
                        this->model[nbeh].cls_dir.resize(dims_out[0]);
                        this->model[nbeh].cls_error.resize(dims_out[0]);
                        this->model[nbeh].cls_tr.resize(dims_out[0]);

                        //initialize other arrays
                        this->translated_index[nbeh].resize(dims_out[0]);
                        this->flag[nbeh].resize(dims_out[0]);

                        beh_present[nbeh] = 1;
 
                    } catch (H5::Exception error) {
                
                        QString errMsgTitle = QString("Classifier Params");
                        QString errMsgText = QString("In parameter file, %1").arg(this->classifier_file);
                        errMsgText += QString(" Beh not present, %1").arg(QString::fromStdString(beh[nbeh]));
                        QMessageBox::critical(this, errMsgTitle, errMsgText);

                    }                   
                }
            }

            nbeh_present = std::count(beh_present.begin(), beh_present.end(), 1);
            file.close();

        }

        // catch failure caused by the H5File operations
        catch( H5::Exception error )
        {

            QString errMsgTitle = QString("Classifier Params");
            QString errMsgText = QString("In parameter file, %1").arg(this->classifier_file);
            errMsgText += QString(" error in function, %1").arg(QString::fromStdString(error.getFuncName()));
            QMessageBox::critical(this, errMsgTitle, errMsgText);
            return;
                
        }

    }  


    void beh_class::loadclassifier_model() 
    {
 
        RtnStatus rtnstatus; 
        std::string class_file = this->classifier_file.toStdString();
        int num_beh = beh_present.size();
        for(int ncls = 0;ncls < num_beh;ncls++)
        {

            if(beh_present[ncls]) 
            {

                rtnstatus = readh5(class_file, this->model_params, this->model[ncls], ncls);
                if(rtnstatus.success)
                    this->isClassifierPathSet = true;

            }
        }
    }


    //https://support.hdfgroup.org/HDF5/doc/cpplus_RM/readdata_8cpp-example.html
    RtnStatus beh_class::readh5(std::string filename, std::vector<std::string> &model_params, 
                                boost_classifier &data_out, int beh_id) 
    {

        RtnStatus rtnstatus;
        int nparams = model_params.size();
        std::vector<float*> model_data = {&data_out.cls_alpha.data()[0],
                                           &data_out.cls_dim.data()[0],
                                           &data_out.cls_dir.data()[0],
                                           &data_out.cls_error.data()[0],
                                           &data_out.cls_tr.data()[0] };
        int rank, ndims;
        hsize_t dims_out[2];
        try 
        {

            // load model params into the model
            for(int paramid = 0; paramid < nparams; paramid++)
            {  
		H5::H5File file(filename, H5F_ACC_RDONLY);
                H5::Group multbeh = file.openGroup(beh[beh_id]);
		H5::DataSet dataset = multbeh.openDataSet(model_params[paramid]);
		H5::DataSpace dataspace = dataset.getSpace();
		rank = dataspace.getSimpleExtentNdims();
		ndims = dataspace.getSimpleExtentDims(dims_out,NULL);
		H5::DataSpace memspace(rank,dims_out);
		dataset.read(model_data[paramid], H5::PredType::IEEE_F32LE, memspace, dataspace);
		file.close();
		rtnstatus.success = true;
            } 
        }
        

        // catch failure caused by the H5File operations
        catch( H5::Exception error )
        {
            QString errMsgTitle = QString("Classifier Params");
            QString errMsgText = QString("Parameter file, %1").arg(QString::fromStdString(filename));
            errMsgText += QString("error in function, %1").arg(QString::fromStdString(error.getFuncName()));
            QMessageBox::critical(this, errMsgTitle, errMsgText);
            rtnstatus.success = false;
            return rtnstatus;
        }
        return rtnstatus;

    }


    void beh_class::translate_mat2C(HOGShape *shape_side, HOGShape *shape_front) 
    {


	//shape of hist side
	unsigned int side_x = shape_side->x;
        unsigned int side_y = shape_side->y;
	unsigned int side_bin = shape_side->bin;

	//shape of hist front 
	unsigned int front_x = shape_front->x;
	unsigned int front_y = shape_front->y;
	unsigned int front_bin = shape_front->bin;

	// translate index from matlab to C indexing  
	unsigned int rollout_index, rem;
	unsigned int ind_k, ind_j, ind_i;
	unsigned int index;
	int dim;
	rem = 0;
	int flag = 0;
	int numWkCls = model[0].cls_alpha.size();

        int num_beh = beh_present.size();
        for(int ncls = 0;ncls < num_beh;ncls++)
        {
           
            if(beh_present[ncls])
            {

		for(int midx = 0; midx < numWkCls; midx ++) {

		    dim = this->model[ncls].cls_dim[midx];
		    flag = 0;

		    if(dim > ((side_x+front_x) * side_y * side_bin) ) { // checking if feature is hog/hof

			rollout_index = dim - ( (side_x + front_x) * side_y * side_bin) - 1;
			flag = 3;

		    } else {

			rollout_index = dim - 1;
			flag = 1;

		    }

		    ind_k = rollout_index / ((side_x + front_x) * side_y); // calculate index for bin
		    rem = rollout_index % ((side_x + front_x) * side_y); // remainder index for patch index

		    if(rem > 0) {

			ind_i = (rem) / side_y; // divide by second dim because that is first dim for matlab. This gives 
				     // index for first dim.
			ind_j = (rem) % side_y;

		    } else {

			ind_i = 0;

			ind_j = 0;

		    }

		    if(ind_i >= side_x) { // check view by comparing with size of first dim of the view

			ind_i = ind_i - side_x;
			flag = flag + 1;

		    }

		    if(flag == 1) {  // book keeping to check which feature to choose

			index = ind_k*side_x*side_y + ind_j*side_x + ind_i;
			this->translated_index[ncls][midx] = index;
			this->flag[ncls][midx] = flag;

		    } else if(flag == 2) {

			index = ind_k*front_x*front_y + ind_j*front_x + ind_i;
			this->translated_index[ncls][midx] = index;
			this->flag[ncls][midx] = flag;

		    } else if(flag == 3) {

			index = ind_k*side_x*side_y + ind_j*side_x + ind_i;
			this->translated_index[ncls][midx] = index;
			this->flag[ncls][midx] = flag;

		    } else if(flag == 4) {

			index = ind_k*front_x*front_y + ind_j*front_x + ind_i;
			this->translated_index[ncls][midx] = index;
			this->flag[ncls][midx] = flag;

	            }
                } 
	    }
        }
    }


    // boost score from a single stump of the model 
    void beh_class::boost_compute(float &scr, std::vector<float> &features, int ind,
			   int num_feat, int feat_len, int dir, float tr, float alpha) 
    {

	float addscores = 0.0;
	if(dir > 0) {

	    if(features[ind] > tr) {

	        addscores = 1;

	    } else {

		addscores = -1;
	    }

	    addscores = addscores * alpha;
	    scr = scr + addscores;

	} else {

	    if(features[ind] <= tr) {

	       addscores = 1;

	    } else {

	       addscores = -1;
	    }

	    addscores = addscores * alpha;
	    scr = scr + addscores;

	}

    }


    void beh_class::boost_classify(std::vector<float> &scr, std::vector<float> &hogs_features,
			 std::vector<float> &hogf_features, std::vector<float> &hofs_features,
			 std::vector<float> &hoff_features, struct HOGShape *shape_side,
			 struct HOFShape *shape_front, int feat_len,
			 std::vector<boost_classifier> &model) 
    {

	//shape of hist side
	unsigned int side_x = shape_side->x;
	unsigned int side_y = shape_side->y;
	unsigned int side_bin = shape_side->bin;

	//shape of hist front 
	unsigned int front_x = shape_front->x;
	unsigned int front_y = shape_front->y;
	unsigned int front_bin = shape_front->bin;

	//index variables
	unsigned int rollout_index, rem;
	unsigned int ind_k, ind_j, ind_i;
	unsigned int num_feat, index;
	int dir, dim;
	float alpha, tr;

	//rem = 0;
	//int flag = 0;

	int numWkCls = model[0].cls_alpha.size();
        int num_beh = beh_present.size();
        std::fill(scr.begin(), scr.end(), 0.0);

        for(int ncls = 0;ncls < num_beh;ncls++)
        {

            if(beh_present[ncls])
            {

	        // translate index from matlab to C indexing
		for(int midx = 0; midx < numWkCls; midx ++) {

		    dim = model[ncls].cls_dim[midx];
		    dir = model[ncls].cls_dir[midx];
		    alpha = model[ncls].cls_alpha[midx];
		    tr = model[ncls].cls_tr[midx];

		    if(this->flag[ncls][midx] == 1) {  // book keeping to check which feature to choose

			index = this->translated_index[ncls][midx];
			num_feat = side_x * side_y * side_bin;
			boost_compute(scr[ncls], hofs_features, index, num_feat, feat_len, dir, tr, alpha);

		    } else if(this->flag[ncls][midx] == 2) {

			index = this->translated_index[ncls][midx];
			num_feat = front_x * front_y * front_bin;
			boost_compute(scr[ncls], hoff_features, index, num_feat, feat_len, dir, tr, alpha);

		    } else if(this->flag[ncls][midx] == 3) {

		        index = this->translated_index[ncls][midx];
		        num_feat = side_x * side_y * side_bin;
		        boost_compute(scr[ncls], hogs_features, index, num_feat, feat_len, dir, tr, alpha);

		    } else if(this->flag[ncls][midx] == 4) {

			index = this->translated_index[ncls][midx];
			num_feat = front_x * front_y * front_bin;
			boost_compute(scr[ncls], hogf_features, index, num_feat, feat_len, dir, tr, alpha);
                    }

	        }

	    }

        }
 
    }


    bool beh_class::pathExists(hid_t id, const std::string& path)
    {

        return H5Lexists( id, path.c_str(), H5P_DEFAULT ) > 0;

    }


}

