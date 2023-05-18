#ifndef PARSER_HPP
#define PARSER_HPP

#include <stdio.h>
#include <string>
#include <iostream>

using namespace std;

namespace bias {

    struct CmdLineParams {

        string output_dir = "";
        bool isVideo = false;
        bool saveFeat = false;
        bool compute_jaaba = false;
        bool classify_scores = false;
        bool visualize = false;
        bool isSkip = false;
        int numframes = 0;
        int wait_thres = 1500;
        int window_size = 5;
    };
    
    void print(CmdLineParams& cmdlineparams);

    /*class ParseCmdLine
    {
        public:
            ParseCmdLine();
            void parser(int argc, char *argv[], CmdLineParams& cmdlineparams);

    };*/
}
#endif