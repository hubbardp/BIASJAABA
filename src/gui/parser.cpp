#include "parser.hpp"

namespace bias{

    /*ParseCmdLine::ParseCmdLine(){}

    void ParseCmdLine::parser(int argc, char *argv[], CmdLineParams& cmdlineparams) {

        int opt;

        // put ':' in the starting of the
        // string so that program can 
        //distinguish between '?' and ':' 
        while ((opt = getopt(argc, argv,":o:isclv")) != -1)
        {
            switch (opt)
            {
                case 'o':
                    cmdlineparams.output_dir = optarg;
                    break;
                case 'i':
                    cmdlineparams.isVideo = true;
                    break;
                case 's':
                    cmdlineparams.saveFeat = true;
                    break;
                case 'c':
                    cmdlineparams.compute_jaaba = true;
                    break;
                case 'l':
                    cmdlineparams.classify_scores = true;
                    break;
                case 'v':
                    cmdlineparams.visualize = true;
                    break;
                case ':':
                    printf("Required argument %c", opt);
                    break;
                case '?' :
                    //printf("unknown option : %c\n", optopt);
                    break;
            }
        }

        // optind is for the extra arguments
        // which are not parsed
        /*for (; optind < argc; optind++) {
            printf(“extra arguments : %s\n”, argv[optind]);
        }
    }*/
    

    void print(CmdLineParams& cmdlineparams)
    {
        std::cout << "Command line arguments\n"
            << "Output Dir: " << cmdlineparams.output_dir
            << "\nis Video:" << cmdlineparams.isVideo
            << "\nsave features:" << cmdlineparams.saveFeat
            << "\ncompute jaaba: " << cmdlineparams.compute_jaaba
            << "\nclassify scores: " << cmdlineparams.classify_scores
            << "\nvisualize " << cmdlineparams.visualize
            << "\nnumframes " << cmdlineparams.numframes
            << "\nisskip " << cmdlineparams.isSkip
            << "\n wait threshold" << cmdlineparams.wait_thres // wait time between jaaba views for computing a score
            << "\ nwindow_size " << cmdlineparams.window_size // averaging window size hoghof features
            << "\n DEBUG " << cmdlineparams.debug
            << "\n com port" << cmdlineparams.comport
            << std::endl;
    }


}