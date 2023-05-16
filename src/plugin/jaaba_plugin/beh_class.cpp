
#include "beh_class.hpp"
#include <iostream>
#include <QDebug>
#include <fstream>
#include <algorithm>
#include <functional>
#include <iomanip>

#define DEBUG 0

namespace bias {

    //DEBUG variables test
    static int test_frameCount=1347;
    static int test_classifier_index=0;

    beh_class::beh_class(QWidget *parent): QDialog(parent) {}
  

    void beh_class::allocate_model() 
    {

        hsize_t dims_out[2] = {0};
        H5::Exception::dontPrint();
        try 
        {
           
            // open classfier file
			//int rank;
			int ndims;
			H5::H5File file(this->classifier_file, H5F_ACC_RDONLY);

            // check the number of behaviors present
            size_t num_beh = beh.size();

            //initialize other arrays
            translated_index.resize(num_beh);
            flag.resize(num_beh);
            finalscore.score.resize(num_beh,0);
            predScoreSide.score.resize(num_beh, 0);
            predScoreFront.score.resize(num_beh, 0);

            for(unsigned int nbeh =0;nbeh < num_beh;nbeh++)
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
                        QString errMsgText = QString("In parameter file, %1").arg(QString::fromStdString(this->classifier_file));
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
            QString errMsgText = QString("In parameter file, %1").arg(QString::fromStdString(this->classifier_file));
            errMsgText += QString(" error in function, %1").arg(QString::fromStdString(error.getFuncName()));
            QMessageBox::critical(this, errMsgTitle, errMsgText);
            return;
                
        }

    }  


    void beh_class::loadclassifier_model() 
    {
 
        RtnStatus rtnstatus; 
        std::string class_file = this->classifier_file;
        size_t num_beh = beh_present.size();
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
        size_t nparams = model_params.size();
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
		unsigned int dim;
		rem = 0;
		int flag = 0;
		size_t numWkCls = model[0].cls_alpha.size();

        size_t num_beh = beh_present.size();
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

    void beh_class::translate_featureIndexes(HOGShape *shape_side, HOGShape *shape_front, bool isSide)
    {

        //shape of hist side
        unsigned int side_x = shape_side->x;
        unsigned int side_y = shape_side->y;
        unsigned int side_bin = shape_side->bin;

        //shape of hist front 
        unsigned int front_x = shape_front->x;
        unsigned int front_y = shape_front->y;
        unsigned int front_bin = shape_front->bin;

        unsigned int side_shape = side_x * side_y * side_bin;
        unsigned int front_shape = front_x * front_y * front_bin;
        unsigned int feat_dims_side = side_shape * 2;
        unsigned int feat_dims_front = front_shape * 2;


        // translate index from matlab to C indexing  
        unsigned int rollout_index, rem;
        unsigned int ind_k, ind_j, ind_i;
        unsigned int index;
        unsigned int dim;
        rem = 0;
        int flag = 0;
        int start_range, end_range;
        vector<int> featrange;
        int feat_dims;

        if (isSide)
        {
            this->translated_featureindexes.resize(feat_dims_side);
            featrange.resize(feat_dims_side);
            feat_dims = feat_dims_side;
            start_range = 1;
            end_range = feat_dims_side / 2;
            for (int i = start_range; i <= end_range; i++)
            {
                featrange[i - 1] = i;
            }
            start_range = (feat_dims_side) / 2 + (feat_dims_front) / 2 + 1;
            end_range = start_range + (feat_dims_side / 2) - 1;
            for (int i = start_range; i <= end_range; i++)
            {
                featrange[i - 1] = i;
            }
        }
        else {
            feat_dims = feat_dims_front;
            this->translated_featureindexes.resize(feat_dims_front);
            featrange.resize(feat_dims_front);
            start_range = (feat_dims_side / 2) + 1;
            end_range = start_range + (feat_dims_front / 2) - 1;
            for (int i = start_range; i <= end_range; i++)
            {
                featrange[i - 1] = i;
            }
            start_range = (feat_dims_side)+(feat_dims_front / 2) + 1;
            end_range = start_range + (feat_dims_front / 2) - 1;
            for (int i = start_range; i <= end_range; i++)
            {
                featrange[i - 1] = i;
            }

        }

        for (int dim_idx = 0; dim_idx < feat_dims; dim_idx++) {

            flag = 0;
            dim = featrange[dim_idx];
            if (dim > ((side_x + front_x) * side_y * side_bin)) { // checking if feature is hog/hof

                rollout_index = dim - ((side_x + front_x) * side_y * side_bin) - 1;
                flag = 3;

            }
            else {

                rollout_index = dim - 1;
                flag = 1;

            }

            ind_k = rollout_index / ((side_x + front_x) * side_y); // calculate index for bin
            rem = rollout_index % ((side_x + front_x) * side_y); // remainder index for patch index

            if (rem > 0) {

                ind_i = (rem) / side_y; // divide by second dim because that is first dim for matlab. This gives 
                // index for first dim.
                ind_j = (rem) % side_y;

            }
            else {

                ind_i = 0;

                ind_j = 0;

            }

            if (ind_i >= side_x) { // check view by comparing with size of first dim of the view

                ind_i = ind_i - side_x;
                flag = flag + 1;

            }

            if (flag == 1) {  // book keeping to check which feature to choose

                index = ind_k * side_x*side_y + ind_j * side_x + ind_i;
                this->translated_featureindexes[dim_idx] = index;

            }
            else if (flag == 2) {

                index = ind_k * front_x*front_y + ind_j * front_x + ind_i;
                this->translated_featureindexes[dim_idx] = index;

            }
            else if (flag == 3) {

                index = ind_k * side_x*side_y + ind_j * side_x + ind_i;
                this->translated_featureindexes[dim_idx] = index;

            }
            else if (flag == 4) {

                index = ind_k * front_x*front_y + ind_j * front_x + ind_i;
                this->translated_featureindexes[dim_idx] = index;

            }
        }
        if (isSide)
            write_translated_indexes("./translatedindexes_mat2C_side.csv", featrange, feat_dims);
        else
            write_translated_indexes("./translatedindexes_mat2C_front.csv", featrange, feat_dims);

    }



    void beh_class::write_translated_indexes(string filename, vector<int>& index_vector, int feat_dim)
    {

        std::ofstream x_out;
        x_out.open(filename.c_str(), std::ios_base::app);


        /*size_t numWkCls = model[0].cls_alpha.size();
        //size_t num_beh = beh_present.size();
        size_t num_beh = 1;

        for (int ncls = 0; ncls < num_beh; ncls++)
        {
            std::cout << "started writing " << std::endl;
            if (ncls == 0)
                x_out << "Matlab" << "," << "Cuda" << "\n";
            if (beh_present[ncls])
            {
                for (int midx = 0; midx < numWkCls; midx++) {

                    dim = this->model[ncls].cls_dim[midx];
                    translated_dim = this->translated_index[ncls][midx];
                    x_out << dim << "," << translated_dim;
                    x_out << "\n";
                }
            }
            std::cout << "Finished Writing indxes " << std::endl;
        }*/

        int translated_dim, dim;
        for (int i = 0; i < feat_dim; i++)
        {
            if (i == 0)
                x_out << "Matlab" << "," << "Cuda" << "\n";
            dim = index_vector[i];
            translated_dim = this->translated_featureindexes[i];
            x_out << dim << "," << translated_dim;
            x_out << "\n";
        }

        x_out.close();
    }

	void beh_class::getviewandfeature(HOGShape *shape_side, HOGShape *shape_front)
	{
		//shape of hist side
		unsigned int side_x = shape_side->x;
		unsigned int side_y = shape_side->y;
		unsigned int side_bin = shape_side->bin;

		//shape of hist front 
		unsigned int front_x = shape_front->x;
		unsigned int front_y = shape_front->y;
		unsigned int front_bin = shape_front->bin;

		unsigned int feat_dim_side = shape_side->x * shape_side->y*shape_side->bin;
		unsigned int feat_dim_front = shape_front->x* shape_front->y*shape_front->bin;
		//std::cout << "feat_dim_side = " << feat_dim_side << std::endl;
		//std::cout << "feat_dim_front = " << feat_dim_front << std::endl;
		//std::cout <<"limits: "<<"1 "<< feat_dim_side<<" "<< (feat_dim_side + feat_dim_front)
		//	<< " " << ((2 * feat_dim_side) + feat_dim_front) << std::endl;

		size_t numWkCls = model[0].cls_alpha.size();
		size_t num_beh = beh_present.size();
		int cls_dim = 0;
		int flag = 0;
		int cuda_feat_dim = 0;

		for (int beh_id = 0; beh_id < num_beh; beh_id++)
		{
			if (beh_present[beh_id]) {
				for (int cls_id = 0; cls_id < numWkCls; cls_id++)
				{
					cls_dim = this->model[beh_id].cls_dim[cls_id];
					cuda_feat_dim = cls_dim-1;
					
					if(cls_dim > 0 && cls_dim <= feat_dim_side ){
						flag = 1;
					}
					else if (cls_dim > feat_dim_side && cls_dim <= (feat_dim_side + feat_dim_front) ) {
						cuda_feat_dim -= feat_dim_side;
						flag = 2;
					}
					else if(cls_dim > (feat_dim_side + feat_dim_front) && cls_dim <= ((2*feat_dim_side) + feat_dim_front)) {
						cuda_feat_dim -= (feat_dim_side + feat_dim_front);
						flag = 3;
					}else {
						cuda_feat_dim -= ((2 * feat_dim_side) + feat_dim_front);
						flag = 4;
					}
					this->translated_index[beh_id][cls_id] = cuda_feat_dim;
					this->flag[beh_id][cls_id] = flag;
					//std::cout << "beh " << beh_id << ", cls_id = " << cls_id << ", cls_dim = " << cls_dim << ", flag = " << flag << std::endl;
                    //std::cout << "cuda dim " << cuda_feat_dim << ", matlab dim = " << cls_dim  << ", flag = " << flag << std::endl;
				}
			}
		}
	}

    // boost score from a single stump of the model 
    void beh_class::boost_compute(float &scr, std::vector<float> &features, int ind,
			   int num_feat, int feat_len, int dir, float tr, float alpha, int framecount, int cls_idx ) 
    {
      
		float addscores = 0.0;
		if(dir > 0) {

			if(features[ind] > tr) {

				addscores = 1;

			} else {

				addscores = -1;
			}
#if DEBUG
            if (framecount == test_frameCount && cls_idx == test_classifier_index)
                printf("%.7f: feat val, %7f: addscore val, %.7f: tr,  %.7f scr before", features[ind], addscores, tr,scr);
#endif
            addscores = addscores * alpha;
            scr = scr + addscores;

#if DEBUG
            if (framecount == test_frameCount && cls_idx == test_classifier_index)
                printf("%.7f scr after, %.7f", scr, alpha);
#endif

		} else {

			if(features[ind] <= tr) {

				addscores = 1;

			} else {

				addscores = -1;
			}
#if DEBUG
            if (framecount == test_frameCount && cls_idx == test_classifier_index)
                printf("%.7f: feat val, %.7f: addscore val, %.7f: tr,  %.7f scr before", features[ind], addscores, tr, scr);
#endif			
            addscores = addscores * alpha;
			scr = scr + addscores;

#if DEBUG
            if (framecount == test_frameCount && cls_idx == test_classifier_index)
                printf("%.7f scr after, %.7f",  scr, alpha);
#endif
		}

    }

    void beh_class::boost_classify_side(std::vector<float> &scr, std::vector<float> &hogs_features,
			  std::vector<float> &hofs_features, struct HOGShape *shape_side,
			  struct HOFShape *shape_front, int feat_len,
			  std::vector<boost_classifier> &model, int frameCount) 
    {
		//std::cout << "Entered boost classify side " << std::endl;
		//shape of hist side
		unsigned int side_x = shape_side->x;
		unsigned int side_y = shape_side->y;
		unsigned int side_bin = shape_side->bin;

		//shape of hist front 
		unsigned int front_x = shape_front->x;
		unsigned int front_y = shape_front->y;
		unsigned int front_bin = shape_front->bin;

		//index variables
		//unsigned int rollout_index, rem;
		//unsigned int ind_k, ind_j, ind_i;
		unsigned int num_feat, index;
		int dir, dim;
		float alpha, tr;

		bool haveprinted_hof = false;
		bool haveprinted_hog = false;
		float scr_before;

		//rem = 0;
		//int flag = 0;

		size_t numWkCls = model[0].cls_alpha.size();
        size_t num_beh = beh_present.size();
        std::fill(scr.begin(), scr.end(), 0.0);

#if DEBUG
        std::ofstream x_out;
        string file = "test_side.csv";
        
        if(frameCount == test_frameCount)
            x_out.open(file.c_str(), std::ios_base::app);
#endif
		num_feat = side_x * side_y * side_bin;
        //std::cout << "numWkCls " << numWkCls << "num_beh " << num_beh << std::endl;

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
					scr_before = scr[ncls];

					//std::cout << "ncls = " << ncls << " midx = " << midx << std::endl;
					//std::cout<<"flag " << this->flag[ncls][midx] << std::endl;

					if(this->flag[ncls][midx] == 1) {  // book keeping to check which feature to choose

						index = this->translated_index[ncls][midx];
						boost_compute(scr[ncls], hofs_features, index, num_feat, feat_len, dir, tr, alpha,
                            frameCount, midx);

#if DEBUG
						if (!haveprinted_hof && frameCount == test_frameCount) {
							/*std::cout << "side hof weak classifier = " << midx
								<< ", feat number = " << dim
						   	<< ", translated index = " << index
								<< ", feat value = " << hofs_features[index]
								<< ", thresh = " << tr
								<< ", dir = " << dir
								<< ", alpha = " << alpha
								<< ", scr before = " << scr_before
								<< ", scr after = " << scr[ncls]
								<< std::endl;
					          haveprinted_hof = true;*/
                            x_out << midx << "," << scr[ncls] << "\n";
						}
#endif

					} else if(this->flag[ncls][midx] == 3) {

						index = this->translated_index[ncls][midx];
						num_feat = side_x * side_y * side_bin;
						boost_compute(scr[ncls], hogs_features, index, num_feat, feat_len, dir, tr, alpha,
                            frameCount, midx);

#if DEBUG
						if (!haveprinted_hog && frameCount == test_frameCount) {
							/*std::cout << "side hog weak classifier = " << midx
						   		<< ", feat number = " << dim
								<< ", translated index = " << index
								<< ", feat value = " << hogs_features[index]
								<< ", thresh = " << tr
								<< ", dir = " << dir
								<< ", alpha = " << alpha
								<< ", scr before = " << scr_before
								<< ", scr after = " << scr[ncls]
								<< std::endl;
							    haveprinted_hog = true;*/
                            x_out << midx << "," << scr[ncls] << "\n";
						}
#endif
					} 

				}
                //haveprinted_hog = true;
                //haveprinted_hof = true;

			}

        }
#if DEBUG
        if(frameCount == test_frameCount)
            x_out.close();
#endif
    }

    void beh_class::boost_classify_front(std::vector<float> &scr, std::vector<float>& hogf_features,
        std::vector<float>& hoff_features, struct HOGShape *shape_side,
        struct HOFShape *shape_front, int feat_len,
        std::vector<boost_classifier> &model, int frameCount)
    {
		//std::cout << "Entered boost classify front " << std::endl;
        //shape of hist side
        unsigned int side_x = shape_side->x;
        unsigned int side_y = shape_side->y;
        unsigned int side_bin = shape_side->bin;

        //shape of hist front 
        unsigned int front_x = shape_front->x;
        unsigned int front_y = shape_front->y;
        unsigned int front_bin = shape_front->bin;

        //index variables
        //unsigned int rollout_index, rem;
        //unsigned int ind_k, ind_j, ind_i;
        unsigned int num_feat, index;
        int dir, dim;
        float alpha, tr;

		bool haveprinted_hof = false;
		bool haveprinted_hog = false;
		float scr_before;

        //rem = 0;
        //int flag = 0;

        size_t numWkCls = model[0].cls_alpha.size();
        size_t num_beh = beh_present.size();
        std::fill(scr.begin(), scr.end(), 0.0);
        
#if DEBUG
        std::ofstream x_out;
        string file = "test_front.csv";

        if(frameCount == test_frameCount)
            x_out.open(file.c_str(), std::ios_base::app);
#endif

		num_feat = front_x * front_y * front_bin;

		//std::cout << "num_beh = " << num_beh << std::endl;

        for (int ncls = 0; ncls < num_beh; ncls++)
        {
             
            if (beh_present[ncls])
            {

                // translate index from matlab to C indexing
                for (int midx = 0; midx < numWkCls; midx++) {

                    dim = model[ncls].cls_dim[midx];
                    dir = model[ncls].cls_dir[midx];
                    alpha = model[ncls].cls_alpha[midx];
                    tr = model[ncls].cls_tr[midx];
					scr_before = scr[ncls];

                    if (this->flag[ncls][midx] == 2) {  // book keeping to check which feature to choose

                        index = this->translated_index[ncls][midx];
						boost_compute(scr[ncls], hoff_features, index, num_feat, feat_len, dir, tr, alpha,
                            frameCount, midx);
#if DEBUG                        
						if (!haveprinted_hof && frameCount == test_frameCount) {
							/*std::cout << "front hof weak classifier = " << midx
								<< ", feat number = " << dim
								<< ", translated index = " << index
								<< ", feat value = " << hoff_features[index]
								<< ", thresh = " << tr
								<< ", dir = " << dir
							    << ", alpha = " << alpha
								<< ", scr before = " << scr_before
								<< ", scr after = " << scr[ncls]
								<< std::endl;*/
							//haveprinted_hof = true;
                            x_out << midx << "," << scr[ncls] << "\n";
						}
#endif                        

                    }
                    else if (this->flag[ncls][midx] == 4) {

                        index = this->translated_index[ncls][midx];
                        boost_compute(scr[ncls], hogf_features, index, num_feat, feat_len, dir, tr, alpha,
                            frameCount, midx);
		
#if DEBUG

                        if (!haveprinted_hog && frameCount == test_frameCount) {
						        /*std::cout << "front hog weak classifier = " << midx
								<< ", feat number = " << dim
								<<", translated index = "<<index
								<< ", feat value = " << hogf_features[index]
								<< ", thresh = " << tr
								<< ", dir = " << dir
								<< ", alpha = " << alpha
								<< ", scr before = " << scr_before
								<< ", scr after = " << scr[ncls]
								<< std::endl;*/
							//haveprinted_hog = true;
                            x_out << midx << "," << scr[ncls] << "\n";
						}
#endif // DEBUG
                    }

                }
                //haveprinted_hog = true;
                //haveprinted_hof = true;

            }

        }

#if DEBUG
        if(frameCount == test_frameCount)
            x_out.close();
#endif

    }

    void beh_class::addScores(std::vector<float>& scr_side, 
                              std::vector<float>& scr_front)
    {
        size_t num_beh = beh_present.size();
        for (int ncls = 0; ncls < num_beh; ncls++)
        {
            if (beh_present[ncls])
            {
               
                finalscore.score[ncls] = scr_side[ncls] + scr_front[ncls];
            }
        }
    }


    bool beh_class::pathExists(hid_t id, const std::string& path)
    {

        return H5Lexists( id, path.c_str(), H5P_DEFAULT ) > 0;

    }


}
