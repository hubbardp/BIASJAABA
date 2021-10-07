#include "test_config.hpp"
#include <stdio.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

#include <QDebug>
#include <QString>


using namespace std;


int convertValtoInt(string val) {

    return stoi(val);
}

float convertToFloat(string val) {

    return stof(val);
}

void addCamDir(std::shared_ptr<TestConfig>& test_config) {

    if (test_config->numCameras == 2) {

        test_config->cam_dir = "multi";

    }else if (test_config->numCameras == 1){
        
        test_config->cam_dir = "single";
    }else {
        
        test_config->cam_dir = "";
    }
}

void copyConfigField(std::shared_ptr<TestConfig> test_config, int fieldname_index, string val) {


    std::cout << "fieldname index : " << fieldname_index << " val: " << val << std::endl;
    switch (fieldname_index) {

        case 1:
            test_config->numCameras = convertValtoInt(val);
            break;
        case 2:
            test_config->cam_suffix.push_back(val);
            break;
        case 3:
            test_config->dir_len = convertValtoInt(val);
            break;
        case 4:
            test_config->dir_list.push_back(val);
            break;
        case 5:
            test_config->numFrames = convertValtoInt(val);
            break;
        case 6:
            test_config->no_of_trials = convertValtoInt(val);
            break;
        case 7:
            test_config->framerate = convertValtoInt(val);
            break;
        case 8:
            test_config->latency_threshold = convertToFloat(val);
            break;
        case 9:
            test_config->cam_dir = val;
            break;
        case 10:
            test_config->nidaq_prefix = val;
            break;
        case 11:
            test_config->f2f_prefix = val;
            break;
        case 12:
            test_config->queue_prefix = val;
            break;
        case 13:
            test_config->plugin_prefix = val;
            break;
        case 14:
            test_config->logging_prefix = val;
            break;
        case 15:
            test_config->imagegrab_prefix = val;
            break;
        default:
            break;
    }

}

int read_testConfig(std::shared_ptr<TestConfig> test_config, ifstream& input_testconfig) {

    //index to iterate throuh config fieldnames
    int rowIdx = 0;
    string line, val;
    char fieldval[256];
    char fieldname[256];

    // Read data one line at a time
    while (std::getline(input_testconfig, line))
    {
        // Create a stringstream 
        std::stringstream ss(line);

        //get fielname from current line first column
        ss.getline(fieldname, 256, ',');

        // to check that we are reading the correct fieldname 
        if (test_config->fieldnames[rowIdx] != fieldname) {
                     
            std::cout << test_config->fieldnames[rowIdx] << " " << string(fieldname) <<  std::endl;
            QString errMsgText = QString("Test Config not present");               
            qDebug() << errMsgText;
            return -1;
        }
        else {
            //std::cout << test_config.fieldnames[rowIdx] << " " << string(fieldname) << std::endl;
        }

        rowIdx++;

        while (getline(ss, val, ',')) {
            
            //strip all char before the actual 
            //string value to be stored
            while(!isalpha(*val.begin()) && 
                  !isdigit(*val.begin()))
                val.erase(val.begin());
            while (!isalpha(val.back()) &&
                !isdigit(val.back()))
                val.pop_back();
            
            copyConfigField(test_config, rowIdx, val);

        }
        addCamDir(test_config);
       
    }  
    return 0;
}
