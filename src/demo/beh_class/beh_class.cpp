#include "beh_class.hpp"



void beh_class::allocate_model() {

    H5::H5File file(this->classifier_file.toStdString(), H5F_ACC_RDONLY);
    int rank, ndims;
    hsize_t dims_out[2];
    H5::DataSet dataset = file.openDataSet(this->model_params[0]);
    H5::DataSpace dataspace = dataset.getSpace();
    rank = dataspace.getSimpleExtentNdims();
    ndims = dataspace.getSimpleExtentDims(dims_out,NULL);

    this->model.cls_alpha.resize(dims_out[0]);
    this->model.cls_dim.resize(dims_out[0]);
    this->model.cls_dir.resize(dims_out[0]);
    this->model.cls_error.resize(dims_out[0]);
    this->model.cls_tr.resize(dims_out[0]);

    //initialize other arrays
    this->translated_index.resize(dims_out[0],0);
    this->flag.resize(dims_out[0]);

}


void beh_class::loadclassifier_model() {

    std::string class_file = this->classifier_file.toStdString();
    readh5(class_file, this->model_params[0], &this->model.cls_alpha.data()[0]);
    readh5(class_file, this->model_params[1], &this->model.cls_dim.data()[0]);
    readh5(class_file, this->model_params[2], &this->model.cls_dir.data()[0]);
    readh5(class_file, this->model_params[3], &this->model.cls_error.data()[0]);
    readh5(class_file, this->model_params[4], &this->model.cls_tr.data()[0]);

}

void beh_class::translate_mat2C(struct HOGShape *shape_side, HOFShape *shape_front) {

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
    int numWkCls = model.cls_alpha.size();
 
    for(int midx = 0; midx < numWkCls; midx ++) {

        dim = this->model.cls_dim[midx];
        flag = 0;

        if(dim > ((side_x+front_x) * side_y * side_bin) ) { // checking if feature is hog/hof

            rollout_index = dim - ( (side_x +front_x) * side_y * side_bin) - 1;
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
            this->translated_index[midx] = index;
            this->flag[midx] = flag;

        } else if(flag == 2) {

            index = ind_k*front_x*front_y + ind_j*front_x + ind_i;
            this->translated_index[midx] = index;
            this->flag[midx] = flag;

        } else if(flag == 3) {

            index = ind_k*side_x*side_y + ind_j*side_x + ind_i;
            this->translated_index[midx] = index;
            this->flag[midx] = flag;

        } else if(flag == 4) {

            index = ind_k*front_x*front_y + ind_j*front_x + ind_i;
            this->translated_index[midx] = index;
            this->flag[midx] = flag;

        } 

    }

}


// boost score from a single stump of the model 
void beh_class::boost_compute(std::vector<float> &scr, std::vector<float> &features, int ind,
                       int num_feat, int feat_len, int frame, int dir, float tr, float alpha) {


    std::vector<float> addscores(feat_len, 0.0);

    if(dir > 0) {

        if(features[frame * num_feat + ind] > tr) {

                addscores[frame] = 1;

        } else {

                addscores[frame] = -1;
        }

        addscores[frame] = addscores[frame] * alpha;
        scr[frame] = scr[frame] + addscores[frame];

    } else {

        if(features[frame * num_feat + ind] <= tr) {

           addscores[frame] = 1;

        } else {

           addscores[frame] = -1;
        }

        addscores[frame] = addscores[frame] * alpha;
        scr[frame] = scr[frame] + addscores[frame];

    }

}


void beh_class::boost_classify(std::vector<float> &scr, std::vector<float> &hogs_features,
                     std::vector<float> &hogf_features, std::vector<float> &hofs_features,
                     std::vector<float> &hoff_features, struct HOGShape *shape_side,
                     struct HOFShape *shape_front, int feat_len, int frame_id,
                     boost_classifier& model) {

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
    int numWkCls = model.cls_alpha.size();

    // translate index from matlab to C indexing  
    for(int midx = 0; midx < numWkCls; midx ++) {

        dim = model.cls_dim[midx];
        dir = model.cls_dir[midx];
        alpha = model.cls_alpha[midx];
        tr = model.cls_tr[midx];

        if(this->flag[midx] == 1) {  // book keeping to check which feature to choose

            index = this->translated_index[midx];
            num_feat = side_x * side_y * side_bin;
            boost_compute(scr, hofs_features, index, num_feat, feat_len, frame_id, dir, tr, alpha);

        } else if(this->flag[midx] == 2) {

            index = this->translated_index[midx];
            num_feat = front_x * front_y * front_bin;
            boost_compute(scr, hoff_features, index, num_feat, feat_len, frame_id, dir, tr, alpha);

        } else if(this->flag[midx] == 3) {

            index = this->translated_index[midx];
            num_feat = side_x * side_y * side_bin;
            boost_compute(scr, hogs_features, index, num_feat, feat_len, frame_id, dir, tr, alpha);

        } else if(this->flag[midx] == 4) {

            index = this->translated_index[midx];
            num_feat = front_x * front_y * front_bin;
            boost_compute(scr, hogf_features, index, num_feat, feat_len, frame_id, dir, tr, alpha);

        }

    }

}



