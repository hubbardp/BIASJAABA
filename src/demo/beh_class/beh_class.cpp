#include "beh_class.hpp"
#include <fstream>


beh_class::beh_class(){}

beh_class::beh_class(QString& cls_file) {
 
    classifier_file = cls_file;
    initialize();
}

void beh_class::initialize() {

    int err = 0;
    err = allocate_model();
    if (err < 0) {
        QString errMsgText = QString("Missing Classifier file %1").arg(classifier_file);
        errMsgText += QString(" - using default values");
        qDebug() << errMsgText;
        return;
    }
        
    err = loadclassifier_model();
    if (err < 0) {
        QString errMsgText = QString("Missing Classifier file %1").arg(classifier_file);
        errMsgText += QString(" - using default values");
        qDebug() << errMsgText;
        return;
    }
        
}

int beh_class::allocate_model()
{

    hsize_t dims_out[2] = { 0 };
    H5::Exception::dontPrint();
    try
    {

        // open classfier file
        //int rank;
        int ndims;
        H5::H5File file(classifier_file.toStdString(), H5F_ACC_RDONLY);

        // check the number of behaviors present
        size_t num_beh = beh.size();

        //initialize other arrays
        translated_index.resize(num_beh);
        flag.resize(num_beh);
        score.resize(num_beh, 0);
        
        for (unsigned int nbeh = 0; nbeh < num_beh; nbeh++)
        {
            
            if (pathExists(file.getId(), beh[nbeh]))
            {
                
                try
                {

                    //allocate the model
                    H5::Group multbeh = file.openGroup(beh[nbeh]);
                    H5::DataSet dataset = multbeh.openDataSet(model_params[0]);
                    H5::DataSpace dataspace = dataset.getSpace();
                    ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
                    model[nbeh].cls_alpha.resize(dims_out[0]);
                    model[nbeh].cls_dim.resize(dims_out[0]);
                    model[nbeh].cls_dir.resize(dims_out[0]);
                    model[nbeh].cls_error.resize(dims_out[0]);
                    model[nbeh].cls_tr.resize(dims_out[0]);

                    //initialize other arrays
                    translated_index[nbeh].resize(dims_out[0]);
                    flag[nbeh].resize(dims_out[0]);

                    beh_present[nbeh] = 1;
                    

                }catch (H5::Exception error) {

                    QString errMsgTitle = QString("Classifier Params");
                    QString errMsgText = QString("In parameter file, %1").arg(classifier_file);
                    errMsgText += QString(" Beh not present, %1").arg(QString::fromStdString(beh[nbeh]));
                    qDebug() << errMsgText;
                    return 0;

                }
            }
        }

        nbeh_present = std::count(beh_present.begin(), beh_present.end(), 1);
        file.close();

    }

    // catch failure caused by the H5File operations
    catch (H5::Exception error)
    {

        QString errMsgText = QString("In parameter file, %1").arg(classifier_file);
        errMsgText += QString(" error in function, %1").arg(QString::fromStdString(error.getFuncName()));
        qDebug() << errMsgText;
        return 0;

    }
    return 1;

}

int beh_class::loadclassifier_model()
{

    int rtnStatus = 0;
    std::string class_file = classifier_file.toStdString();
    size_t num_beh = beh_present.size();
    for (int ncls = 0; ncls < num_beh; ncls++)
    {

        if (beh_present[ncls])
        {

            rtnStatus = readh5(class_file, this->model_params, this->model[ncls], ncls);
            if (rtnStatus)
                isClassifierPathSet = true;
            else {
                qDebug() << "Beh not present";
                
            }
        }
    }
    return 1;
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
    int flag_id = 0;
    size_t numWkCls = model[0].cls_alpha.size();

    size_t num_beh = beh_present.size();
    for (int ncls = 0; ncls < num_beh; ncls++)
    {

        if (beh_present[ncls])
        {
            
            for (int midx = 0; midx < numWkCls; midx++) {

                dim = model[ncls].cls_dim[midx];
                flag_id = 0;

                if (dim > ((side_x + front_x) * side_y * side_bin)) { // checking if feature is hog/hof

                    rollout_index = dim - ((side_x + front_x) * side_y * side_bin) - 1;
                    flag_id = 3;

                }
                else {

                    rollout_index = dim - 1;
                    flag_id = 1;

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
                    flag_id = flag_id + 1;

                }

                if (flag_id == 1) {  // book keeping to check which feature to choose

                    index = ind_k * side_x*side_y + ind_j * side_x + ind_i;
                    translated_index[ncls][midx] = index;
                    flag[ncls][midx] = flag_id;

                }
                else if (flag_id == 2) {

                    index = ind_k * front_x*front_y + ind_j * front_x + ind_i;
                    translated_index[ncls][midx] = index;
                    flag[ncls][midx] = flag_id;

                }
                else if (flag_id == 3) {

                    index = ind_k * side_x*side_y + ind_j * side_x + ind_i;
                    translated_index[ncls][midx] = index;
                    flag[ncls][midx] = flag_id;

                }
                else if (flag_id == 4) {

                    index = ind_k * front_x*front_y + ind_j * front_x + ind_i;
                    translated_index[ncls][midx] = index;
                    flag[ncls][midx] = flag_id;

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
    if (dir > 0) {

        if (features[ind] > tr) {

            addscores = 1;

        }
        else {

            addscores = -1;
        }

        addscores = addscores * alpha;
        scr = scr + addscores;

    }
    else {

        if (features[ind] <= tr) {

            addscores = 1;

        }
        else {

            addscores = -1;
        }

        addscores = addscores * alpha;
        scr = scr + addscores;

    }

}


// boost score from a single stump of the model 
/*void beh_class::boost_compute(std::vector<float> &scr, std::vector<float> &features, int ind,
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
	//unsigned int rollout_index, rem;
    //unsigned int ind_k, ind_j, ind_i;
    unsigned int num_feat, index;
    int dir, dim;
    float alpha, tr;
    //rem = 0;
    //int flag = 0;
    int numWkCls = static_cast<int>(model.cls_alpha.size());

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

}*/

//https://support.hdfgroup.org/HDF5/doc/cpplus_RM/readdata_8cpp-example.html
int beh_class::readh5(std::string filename, std::vector<std::string> &model_params,
    boost_classifier &data_out, int beh_id)
{

    int rtnStatus;
    size_t nparams = model_params.size();
    std::vector<float*> model_data = { &data_out.cls_alpha.data()[0],
                                       &data_out.cls_dim.data()[0],
                                       &data_out.cls_dir.data()[0],
                                       &data_out.cls_error.data()[0],
                                       &data_out.cls_tr.data()[0] };
    int rank, ndims;
    hsize_t dims_out[2];
    try
    {

        // load model params into the model
        for (int paramid = 0; paramid < nparams; paramid++)
        {
            H5::H5File file(filename, H5F_ACC_RDONLY);
            H5::Group multbeh = file.openGroup(beh[beh_id]);
            H5::DataSet dataset = multbeh.openDataSet(model_params[paramid]);
            H5::DataSpace dataspace = dataset.getSpace();
            rank = dataspace.getSimpleExtentNdims();
            ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
            H5::DataSpace memspace(rank, dims_out);
            dataset.read(model_data[paramid], H5::PredType::IEEE_F32LE, memspace, dataspace);
            file.close();
            rtnStatus = 1;
        }
    }


    // catch failure caused by the H5File operations
    catch (H5::Exception error)
    {
        QString errMsgTitle = QString("Classifier Params");
        QString errMsgText = QString("Parameter file, %1").arg(QString::fromStdString(filename));
        errMsgText += QString("error in function, %1").arg(QString::fromStdString(error.getFuncName()));
        qDebug() << errMsgText;
        rtnStatus = 0;
        return rtnStatus;
    }
    return rtnStatus;

}


void beh_class::boost_classify(std::vector<float> &scr, std::vector<float> &hogs_features,
    std::vector<float> &hogf_features, std::vector<float> &hofs_features,
    std::vector<float> &hoff_features, struct HOGShape *shape_side,
    struct HOFShape *shape_front, int feat_len,
    std::vector<boost_classifier>& model)
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
    //unsigned int rollout_index, rem;
    //unsigned int ind_k, ind_j, ind_i;
    unsigned int num_feat, index;
    int dir, dim;
    float alpha, tr;

    //rem = 0;
    //int flag = 0;

    size_t numWkCls = model[0].cls_alpha.size();
    size_t num_beh = beh_present.size();
    std::fill(scr.begin(), scr.end(), 0.0);

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

                if (flag[ncls][midx] == 1) {  // book keeping to check which feature to choose

                    index = translated_index[ncls][midx];
                    num_feat = side_x * side_y * side_bin;
                    boost_compute(scr[ncls], hofs_features, index, num_feat, feat_len, dir, tr, alpha);

                }
                else if (flag[ncls][midx] == 2) {

                    index = translated_index[ncls][midx];
                    num_feat = front_x * front_y * front_bin;
                    boost_compute(scr[ncls], hoff_features, index, num_feat, feat_len, dir, tr, alpha);

                }
                else if (flag[ncls][midx] == 3) {

                    index = translated_index[ncls][midx];
                    num_feat = side_x * side_y * side_bin;
                    boost_compute(scr[ncls], hogs_features, index, num_feat, feat_len, dir, tr, alpha);

                }
                else if (flag[ncls][midx] == 4) {

                    index = translated_index[ncls][midx];
                    num_feat = front_x * front_y * front_bin;
                    boost_compute(scr[ncls], hogf_features, index, num_feat, feat_len, dir, tr, alpha);
                }

            }

        }

    }

}

bool beh_class::pathExists(hid_t id, const std::string& path)
{
    std::cout << (H5Lexists(id, path.c_str(), H5P_DEFAULT) > 0) << std::endl;
    return (H5Lexists(id, path.c_str(), H5P_DEFAULT) > 0);

}


void beh_class::write_score(std::string file, int framenum, float score)
{

    std::ofstream x_out;
    x_out.open(file.c_str(), std::ios_base::app);

    // write score to csv file
    //for(int frame_id = 0;frame_id < framenum;frame_id++)
    x_out << framenum << "," << score << "\n";

    x_out.close();

}

beh_class::~beh_class() {}



