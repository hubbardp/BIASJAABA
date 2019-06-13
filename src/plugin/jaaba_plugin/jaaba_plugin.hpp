#ifndef JAABA_PLUGIN_WINDOW_HPP
#define JAABA_PLUGIN_WINDOW_HPP

#include "ui_jaaba_plugin.h"
#include "bias_plugin.hpp"
#include "rtn_status.hpp"
#include "HOGHOF.hpp"
#include <QMainWindow>
#include <QPointer>
#include <QThreadPool>
#include <QThread>
//test development
#include <fstream>
#include "H5Cpp.h"

namespace bias
{
    class HOGHOF;
    class CameraWindow;

    class JaabaPlugin : public BiasPlugin, public Ui::JaabaPluginDialog
    {
        Q_OBJECT

        public:

            static const QString PLUGIN_NAME;
            static const QString PLUGIN_DISPLAY_NAME;

            JaabaPlugin(int numberOfCameras,QWidget *parent=0);

            virtual QString getName();
            virtual QString getDisplayName();
            cv::Mat getCurrentImage();
            virtual void processFrames(QList<StampedImage> frameList);
            void initHOGHOF(QPointer<HOGHOF> hoghof);
            void genFeatures(QPointer<HOGHOF> hoghof, int frameCount);

        protected:

            QPointer<HOGHOF> HOGHOF_side;
            QPointer<HOGHOF> HOGHOF_front;

        private:

            int nviews_;
            bool detectStarted;
            bool save;
            bool stop_save; 
         
            void initialize();
            void setupHOGHOF();
            void connectWidgets();
            int getNumberofViews();
            void updateWidgetsOnLoad();
            void checkviews();
            void detectEnabled();

            //test development
            void copy_features1d(int frame_num, int num_elements, 
                            std::vector<float> &vec_feat, float* array_feat);
            int createh5(std::string filename, int frame_num,
                          int num_frames, int hog_elements, int hof_elements,
                          std::vector<float> hog, std::vector<float> hof);
            void create_dataset(H5::H5File& file, std::string key,
                          std::vector<float> features, int num_frames, int num_elements);
            void stop();
            void write_output(std::string file,float* out_img, unsigned w, unsigned h);
            void read_image(std::string filename, float* img, int w, int h);
            
        signals:

            void init(); 

        private slots:

            void SideViewCheckBoxChanged(int state);
            void FrontViewCheckBoxChanged(int state);
            void reloadButtonPressed();
            void detectClicked();
            void saveClicked();
    
    };

}
#endif
