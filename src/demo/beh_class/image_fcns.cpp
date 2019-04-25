#include "image_fcns.h"
#include <stdio.h>
#include <assert.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>


void parse_input_HOG(int nargs, char** arg_inputs, struct HOGParameters& params, 
                     struct HOGImage& img, int& num_frames, int& verbose, std::string& view,
                     std::string& mv_path, struct CropParams& crp_params){

    int opt = 0;
    int h,w,c,n,f,v,b,e,p;
    std::set<char>req_options = {'h','w','c','n','i'};
    std::set<char> options;
    std::string g = "side";
    std::string i;
    verbose = 0;f = 1;e = 0;b = 0;p = 0;

    // parse all the arguments from the command line
    while((opt = getopt(nargs,arg_inputs,"v:h:w:c:b:f:n:p:e:g:i:")) != -1) {

        options.insert(opt);

        switch(opt){

            case 'h': h = atoi(optarg);
                break;
            case 'w': w = atoi(optarg);
                break;
            case 'c': c = atoi(optarg);
                break;
            case 'n': n = atoi(optarg);
                break;
            case 'f': f = atoi(optarg);
                break;
            case 'v': v = atoi(optarg);
                verbose = v;
                break;
            case 'b': b = atoi(optarg);
                break;
            case 'p': p = atoi(optarg);
                break;
            case 'e': e = atoi(optarg);
                break;
            case 'g': g = optarg;
                break;
            case 'i': i = optarg;
                break;
            default: std::cout << "Missing arguments:<help>" 
                               << "-usage:-h <height> -w <width> -c <cellsize> "
                               << "-n <nbins> -f <num frames> -v <verbose> "
                               << "-b <ncells for cropsize> -p <no of ips>" 
                               << "-e <enable/diasble crop> -g <side/front view>"
                               << "-i <movie file path>";
                exit(-1);

        }

    }

    //check required arguments
    if(e == 1) { req_options.insert('b'); req_options.insert('p'); }
    IsSubset(req_options,options); 
    
    // populate the HOG/HOF structs from the parsed arguments
    img.h=h;
    img.w=w;
    img.pitch=w;
    img.type=hog_f32;
    img.buf=nullptr;   
  
    params.cell.w=c;
    params.cell.h=c;
    params.nbins=n;

    // populate the crop params
    crp_params.crop_flag=e;
    crp_params.ncells=b;
    crp_params.npatches=p;

    //general params
    num_frames = f;
    view = g;
    mv_path = i; 
    
}


void parse_input_HOF(int nargs, char** arg_inputs, struct HOFParameters& params,
                     int& num_frames, int& verbose, std::string& view, 
                     std::string& mv_path, struct CropParams& crp_params){

    int opt = 0;
    int h,w,c,n,f,v,b,e,p;
    std::set<char>req_options = {'h','w','c','n','i'};
    std::set<char> options;
    std::string g = "side";
    std::string i = "";
    float sigma_smooth= 3 ;
    float sigma_derivative = 1;
    float opt_threshold = 3e-6;
    verbose = 0;f = 1;e = 0;b = 0;p = 0;

    // parse all the arguments from the command line
    while((opt = getopt(nargs,arg_inputs,"v:h:w:c:b:f:n:s:d:o:p:e:g:i:")) != -1) {

        options.insert(opt);

        switch(opt){

            case 'h': h = atoi(optarg);
                break;
            case 'w': w = atoi(optarg);
                break;
            case 'c': c = atoi(optarg);
                break;
            case 'n': n = atoi(optarg);
                break;
            case 'f': f = atoi(optarg);
                break;
            case 'v': v = atoi(optarg);
                verbose = v;
                break;
            case 'b': b= atoi(optarg);
                break;
            case 's': sigma_smooth = atof(optarg);
                break;
            case 'd': sigma_derivative = atof(optarg);
                break;
            case 'o': opt_threshold = atof(optarg);
                break;  
            case 'p': p = atoi(optarg);
                break;
            case 'e': e = atoi(optarg);
                break;
            case 'g': g = optarg;
                break;
            case 'i': i = optarg; 
                break;    
            default: std::cout << "Missing arguments:<help>"
                               << "-usage:-h <height> -w <width> -c <cellsize> "
                               << "-n <nbins> -f <num frames> -v <verbose> "
                               << "-b <ncells for cropsize> -s <sigma smooth>" 
                               << "-d <sigma for derivative> -o <opt_flow threshold>" 
                               << "-p <no of interest points> -e <enable/diasble crop> "
                               << "-g <side/front view> -i <movie file path>";
                exit(-1);
        }

    }
    
    //check required arguments
    if(e == 1) { req_options.insert('b');req_options.insert('p'); }
    IsSubset(req_options,options);
     
    // lk params
    params.lk.sigma.derivative = sigma_derivative;
    params.lk.sigma.smoothing = sigma_smooth;
    params.lk.threshold = opt_threshold;

    // input params
    params.input.w=w;
    params.input.h=h;
    params.input.pitch=w;

    //hof params
    params.cell.h=c;
    params.cell.w=c;
    params.nbins=n;

    //crp params
    crp_params.crop_flag=e;
    crp_params.ncells=b;
    crp_params.npatches=p;

    //general params
    num_frames = f;
    view = g;
    mv_path = i;

}


void read_image(std::string filename, float* img, int w, int h){

    int count_row = 0;
    int count_col = 0;
    std::string lin;
    std::ifstream x_in(filename);

    // read image input from a csv file
    if(x_in){
        while(getline(x_in, lin)){
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
    }else{

         std::cout << "File not present.Enter a valid filename." << std::endl;
         exit(1);
    }
}


//https://support.hdfgroup.org/HDF5/doc/cpplus_RM/readdata_8cpp-example.html
void readh5(std::string filename, std::string dataset_name, float* data_out) {

    H5::H5File file(filename, H5F_ACC_RDONLY);
    int rank,ndims;
    hsize_t dims_out[2];


    H5::DataSet dataset = file.openDataSet(dataset_name);
    H5::DataSpace dataspace = dataset.getSpace();
    rank = dataspace.getSimpleExtentNdims();

    ndims = dataspace.getSimpleExtentDims(dims_out,NULL);
    std::cout << "rank " << rank << ", dimensions " <<
    (unsigned long)(dims_out[0]) << " x " <<
    (unsigned long)(dims_out[1]) << std::endl;

    H5::DataSpace memspace(rank,dims_out);
    dataset.read(data_out, H5::PredType::IEEE_F32LE, memspace, dataspace);
    file.close();
}


// write out the ouput as a 1d array
void write_histoutput(std::string file,float* out_img, unsigned w, unsigned h,unsigned nbins){

     std::ofstream x_out;
     x_out.open(file.c_str());

     // write hist output to csv file
     for(int k=0;k < nbins;k++){
         for(int i = 0;i < h;i++){
             for(int j = 0; j < w;j++){
                 x_out << out_img[k*w*h +i*w + j]; // i*h +j
                 if(j != w-1 || i != h-1 || k != nbins -1)
                     x_out << ",";
             }
         }
     }
}



/*void read_structfromcsv(std::string filename, struct interest_pnts* pts){

    std::string lin;
    std::string result;
    std::ifstream x_in(filename);
    int count=0;

     if(x_in){
         while(getline(x_in, lin)){ 
           if(count!=0){ // skip the line with headers
              std::stringstream iss(lin);
              for(int i=0;i<2;i++){
                  if(i<1){
                      if(std::getline(iss,result,','))
                        pts->side[count-1][0]=atoi(result.c_str());
                      if(std::getline(iss,result,','))
                        pts->side[count-1][1]=atoi(result.c_str()); 
                  }else{
                      if(std::getline(iss,result,','))
                        pts->front[count-1][0]=atoi(result.c_str());       
                      if(std::getline(iss,result,','))
                        pts->front[count-1][1]=atoi(result.c_str());      
                  } 
                           
              }
          
           } 
           count=count+1;
        }
    }
}*/

void readfromcsv(std::string filename, std::vector<int>& interest_pnts){

    std::string lin;
    std::string result;
    std::ifstream x_in(filename);
    int count_row=0;

    if(x_in){
        while(getline(x_in, lin)){ 
            std::stringstream iss(lin);
            if(std::getline(iss,result,','))
                interest_pnts[count_row]=atoi(result.c_str());
            if(std::getline(iss,result,','))
                interest_pnts[count_row+1]=atoi(result.c_str());
            count_row = count_row + 2; // read in x and y coordiante for each point
        } 
    }else{

        std::cout << "File not present.Enter a valid filename." << std::endl;
        exit(1);  
    } 
}


void write_output(std::string file,float* out_img, unsigned w, unsigned h) {

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


void clear_stringstream(std::stringstream& idx ,std::stringstream& dx ,std::stringstream& dy, 
                        std::stringstream& mag ,std::stringstream& th ,std::stringstream& hist){

    idx.str(std::string());
    hist.str(std::string());
    dx.str(std::string());
    dy.str(std::string());
    th.str(std::string());
    mag.str(std::string());

}

// function to check for mandatory arguements in getopt
void IsSubset(std::set<char> A, std::set<char> B) {

    std::set<char>::iterator it;
    for(auto f : A) {

       it = B.find(f);

       if (it == B.end()) {

           printf("Required Field %c .Please enter a valid value.\n",f);
           exit(1);
       }          
 
    } 
     
}

