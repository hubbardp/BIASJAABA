#ifndef BEH_CLASS_HPP
#define BEH_CLASS_HPP 

#include "HOGHOF.hpp"
#include "utils.hpp"

#include "H5Cpp.h"
#include "H5Exception.h"
#include <string>
#include <vector>
#include <QString>


namespace bias {

   
    struct boost_classifier {

	std::vector<float> cls_dim;
	std::vector<float> cls_error;
	std::vector<float> cls_dir;
	std::vector<float> cls_tr;
	std::vector<float> cls_alpha;

    };


    class beh_class : public QDialog {

      public:

	
	int nframes=1; // nframe prediction
        int nbeh_present; // number of beh classification
        bool isClassifierPathSet = false;
        std::vector<float> score = {0,0,0,0,0,0};
	QString classifier_file;
	std::vector<boost_classifier> model = std::vector<boost_classifier>(6);
	std::vector<std::string> model_params{"alpha","dim","dir","error","tr"};
        std::vector<std::string> beh{"Lift","Handopen","Grab","Supinate","Chew","Atmouth"};
        std::vector<int>beh_present = {0,0,0,0,0,0};
	std::vector<std::vector<int>> translated_index; //translated mat style 2 C style indexing
	std::vector<std::vector<int>> flag; // book kepping for features
	beh_class(QWidget *parent);

	void allocate_model();
	void loadclassifier_model();
        RtnStatus readh5(std::string filename, std::vector<std::string> &model_params, boost_classifier &data_out,int beh_id);
        void translate_mat2C(HOGShape *shape_side, HOGShape *shape_front);
	void boost_classify(std::vector<float> &scr, std::vector<float> &hogs_features,
			    std::vector<float> &hogf_features, std::vector<float> &hofs_features,
			    std::vector<float> &hoff_features, struct HOGShape *shape_side,
			    HOFShape *shape_front, int feat_len, 
			    std::vector<boost_classifier> &model);
	void boost_compute(float &scr, std::vector<float> &features, int ind,
			   int num_feat, int feat_len, int dir, float tr, float alpha);
        bool pathExists(hid_t id, const std::string& path);

	
    };

}
#endif  
