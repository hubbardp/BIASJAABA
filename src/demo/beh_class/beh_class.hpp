#ifndef BEH_CLASS_HPP
#define BEH_CLASS_HPP 

#include "HOGHOF.hpp"
#include "utils.hpp"

#include "H5Cpp.h"
#include <vector>
#include <QString>

#include <iostream>

struct boost_classifier {

    std::vector<float> cls_dim;
    std::vector<float> cls_error;
    std::vector<float> cls_dir;
    std::vector<float> cls_tr;
    std::vector<float> cls_alpha;

};


class beh_class {

  public:

    int nframes=2498; // nframe prediction
    QString classifier_file;
    boost_classifier model;
    std::vector<std::string> model_params{"alpha","dim","dir","error","tr"};
    std::vector<float> scores;
    std::vector<int> translated_index; //translated mat style 2 C style indexing
    std::vector<int> flag; // book kepping for features

    void allocate_model();
    void loadclassifier_model();
    void translate_mat2C(struct HOGShape *shape_side, HOFShape *shape_front);
    void boost_classify(std::vector<float> &scr, std::vector<float> &hogs_features,
                        std::vector<float> &hogf_features, std::vector<float> &hofs_features,
                        std::vector<float> &hoff_features, struct HOGShape *shape_side,
                        HOFShape *shape_front, int feat_len, int frame_id, 
                        boost_classifier& model);
    void boost_compute(std::vector<float> &scr, std::vector<float> &features, int ind,
                        int num_feat, int feat_len, int frame_id, int dir, float tr, float alpha);

    
};
#endif  
