#ifndef JAABA_PLUGIN_WINDOW_HPP
#define JAABA_PLUGIN_WINDOW_HPP

#include "ui_jaaba_plugin.h"
#include "camera_window.hpp"
//#include "bias_plugin.hpp"

#include "rtn_status.hpp"
#include "frame_data.hpp"
#include "camera_facade.hpp"
#include "HOGHOF.hpp"
#include "beh_class.hpp"
#include "process_scores.hpp"
#include "image_label.hpp"
//#include "shape_data.hpp"

#include <QMainWindow>
#include <QMessageBox>
#include <QPointer>
#include <QThreadPool>
#include <QThread>

#include<ctime>
#include "win_time.hpp"
#include "parser.hpp"

#ifdef WITH_SPIN
#include "camera_device_spin.hpp"
#endif

//test development
//#include <fstream>
//#include <string>
//#include "H5Cpp.h"
//#include "video_utils.hpp"
//#include <opencv2/highgui/highgui.hpp>

namespace bias
{
    class HOGHOF;
    class beh_class;
    class CameraWindow;
    

    class JaabaPlugin : public BiasPlugin, public Ui::JaabaPluginDialog
    {
        Q_OBJECT

        public:

            static const QString PLUGIN_NAME;
            static const QString PLUGIN_DISPLAY_NAME;
            static const unsigned int DEFAULT_TIMING_BUFFER_SIZE;

            JaabaPlugin(string camera_id, 
                QPointer<QThreadPool> threadPoolPtr,
                std::shared_ptr<Lockable<GetTime>> gettime,
                CmdLineParams& cmdlineparams,
                QWidget *parent=0);

            //void resetTrigger();

            virtual void finalSetup();
            virtual QString getName();
            virtual QString getDisplayName();
            virtual void processFrames(StampedImage stampedImage);
            //virtual void processFrames();
            virtual void reset();
            virtual void stop();
            virtual void stopThread();
            cv::Mat getCurrentImage();
            int getLaserTrigger();
            virtual void setupNIDAQ(std::shared_ptr <Lockable<NIDAQUtils>> nidaq_task,
                                    bool testConfigEnabled, string trial_info,
                                    std::shared_ptr<TestConfig> testConfig);
            virtual void setImageQueue(std::shared_ptr<LockableQueue<StampedImage>> pluginImageQueuePtr,
                                       std::shared_ptr<LockableQueue<unsigned int>> skippedFramesPluginPtr);
            
            virtual void setScoreQueue(std::shared_ptr<LockableQueue<PredData>> selfScoreQueuePtr,
                                       std::shared_ptr<LockableQueue<PredData>> partnerScoreQueuePtr);
            //made it virtual void because of the need to access the function through base class pointer.
            virtual void gpuInit();
            virtual void gpuDeinit();
            virtual QVariantMap getConfigAsMap();
            virtual RtnStatus setConfigFromMap(QVariantMap configMap);
            virtual void initializeParamsProcessScores();
            virtual void setTrialNum(string trialnum);
            virtual void setHOGHOFShape();
            virtual void setWriteScoreFlag();

            QPointer<ProcessScores> processScoresPtr_self;
            QPointer<ProcessScores> processScoresPtr_partner;

            int scoreCount = 0;
            int side_skip_count = 0;
            int front_skip_count = 0;
            bool gpuInitialized = false;

            //jaaba config params 
            string plugin_file;
            string hog_file;
            string hof_file;
            string crop_file;
            string classifier_filename;
            int window_size;
            int cuda_device;
            int num_behs;
            vector<string> beh_names;

        protected:
 
            bool laserOn; 
            bool mesPass;
            unsigned int cameraNumber_;
            unsigned int partnerCameraNumber_;
            HOGShape partner_hogshape_;
            HOFShape partner_hofshape_;
            HOGShape hogshape_;

            PredData predScoreSide_;
            PredData predScoreFront_;
 
            //QPointer<beh_class> classifier;
            //QPointer<VisPlots> visplots;
            QPointer<QThreadPool> threadPoolPtr_;

            CmdLineParams cmdlineparams_;
            std::shared_ptr<Lockable<GetTime>> gettime_;   
            std::shared_ptr<Lockable<NIDAQUtils>> nidaq_task_;
            std::shared_ptr<TestConfig>testConfig_;
            std::shared_ptr<LockableQueue<unsigned int>> skippedFramesPluginPtr_;
            //std::shared_ptr<LockableQueue<PredData>> sideScoreQueuePtr_;
            //std::shared_ptr<LockableQueue<PredData>> frontScoreQueuePtr_;
            std::shared_ptr<LockableQueue<PredData>> selfScoreQueuePtr_;
            std::shared_ptr<LockableQueue<PredData>> partnerScoreQueuePtr_;

            QSharedPointer<QList<QPointer<CameraWindow>>> cameraWindowPtrList_;
            std::shared_ptr<Lockable<Camera>> cameraPtr_;
            QPointer<CameraWindow> getPartnerCameraWindowPtr();
            std::shared_ptr<LockableQueue<StampedImage>> partnerPluginImageQueuePtr_;
            unsigned int getPartnerCameraNumber();

            //void updateTrigStateInfo();
            //RtnStatus connectTriggerDev();

            void allocate_testVec();
            void refill_testVec();

        private:

            int nviews_;
            int nDevices_;
            bool save;
            bool stop_save;
            bool detectStarted = false;
            unsigned int numskippedFrames_=0;
            int image_height=0;
            int image_width=0;
            double tStamp=0.0;
            int64_t process_time=0;
            TimeStamp cam_ofs={0,0};
            int partner_frameCount_;
            bool score_calculated_;
            int frameSkip;
            uint64_t fstfrmtStampRef_;
            double timeStamp_;
            bool process_frame_time;
            size_t hog_num_elements;
            size_t hof_num_elements;
            string view_;

            unsigned long numMessageSent_;
            unsigned long numMessageReceived_;
                         
            // Trigger parameters
            bool triggerEnabled;
            bool triggerArmedState;

            bool testConfigEnabled_;
            string trial_num_;

            JaabaConfig jab_conf;
            string config_file_dir;
            unsigned int camera_serial_id;
            
            unordered_map<string, unsigned int> camera_list;
            unordered_map<unsigned int, string> jab_crop_list;

            int processedFrameCount;
            QPointer<HOGHOF> HOGHOF_self;
            QPointer<HOGHOF> HOGHOF_partner;

            //test
            QString file_frt; // video input for front
            QString file_sde; // video input for side

            std::vector<int64_t> time_cur;
            std::vector<int64_t> ts_pc;// pc time
            std::vector<std::vector<uInt32>>ts_nidaq; // nidaq timings
            std::vector<unsigned int> queue_size; // queue size
            std::vector<int64_t> ts_gpuprocess_time;// gpu process timings
            std::vector<int64_t> ts_nidaqThres;// nmidaq threshold for spiked frames
            std::vector<int64_t> ts_jaaba_start;// test
            std::vector<int64_t> ts_jaaba_end;
            std::vector<double> imageTimeStamp;

            //test
            std::vector<std::vector<float>>hoghof_feat;
            std::vector<std::vector<float>>hoghof_feat_avg;

            //std::vector<PredData>scores;
            //int no_of_skips = 10;
            //priority_queue<int, vector<int>, greater<int>>skipframes_view1; // side skips
            //priority_queue<int, vector<int>, greater<int>>skipframes_view2; // front skips
            //std::vector<int> skipframes1;
            //std::vector<int> skipframes2;

            std::vector<float>laserRead = { 0,0,0,0,0,0 };
            std::vector<double>timeofs;
            std::vector<double>timestd;
			vector<vector<float>> translated_indexes;
          
            bool pluginReady();
            bool isSender();
            bool isReceiver(); 
            //void triggerLaser();   
            int getNumberofViews();
            int getNumberOfDevices();
            //void updateWidgetsOnLoad();
            void initialize(CmdLineParams& cmdlineparams);
            void setupHOGHOF();
            void setupClassifier();
            void connectWidgets();
            void detectEnabled();
            double convertTimeStampToDouble(TimeStamp curr, TimeStamp init);
            void getFormatSettings();
            //void gpuInit(); 
            void loadConfig();
            void average_windowFeatures(vector<float>&hog_feat, vector<float>&hof_feat,
                                        vector<float>& hog_feat_avg, vector<float>& hof_feat_avg,
                                        int window_size);
            void subtractLastwindowfeature(vector<float>& hog_past, vector<float>& hof_past, 
                                           vector<float>& hog_feat_avg, vector<float>& hof_feat_avg);

            void processFrame_inPlugin(StampedImage stampedImage);
            //void processFramePass();
            //void initiateVidSkips(priority_queue<int, vector<int>, greater<int>>& skip_frames);
            void write_score_final(std::string file, unsigned int numFrames,
                vector<PredData>& pred_score);
            bool jaabaSkipFrame(uint64_t& ts_cur, uint64_t& ts_prev,
                int& frameCount, uint64_t wait_thres);
            void waitForEmptyHOGHOFAvgQueue(LockableQueue<vector<float>>& avg_que);

            //void writeAllFeatures(string filename, vector<float>& feat_out,
            //    int feat_size);
            //void saveAvgwindowfeatures(vector<vector<float>>& hoghof_feat, QPointer<HOGHOF> hoghof_obj,
            //                           int frameCount,string filename);
            //void saveFeatures(string filename, QPointer<HOGHOF> hoghof_obj,//vector<float>& feat_out,
            //    int hog_num_elements, int hof_num_elements);
			//void paintEvent(QPainter& paitner);

            int numframes_;
            //command line variables
            string output_feat_directory;
            bool isVideo;
            bool saveFeat;
            bool compute_jaaba;
            bool classify_scores;
            bool visualize;
            bool isDebug;

            //test
            uint64_t start_prev = 0;

        signals:

            void partnerImageQueue(std::shared_ptr<LockableQueue<StampedImage>> partnerPluginImageQueuePtr);
            void processSide(bool side);
            void processFront(bool front);
            void passHOGShape(QPointer<HOGHOF> partner_hogshape);
            void passScore(PredData predSore);
            void passFrameRead(int64_t frameRead, int);
            void passScoreDone(bool score_cal);
            void doNotProcess(unsigned int frameCount);
            void framecountMatchError(unsigned int, QString);

        private slots:

            //void SideViewCheckBoxChanged(int state);
            //void FrontViewCheckBoxChanged(int state);
            void onPartnerPlugin(std::shared_ptr<LockableQueue<StampedImage>> partnerPluginImageQueuePtr);
            //void trigResetPushButtonClicked();
            //void trigEnabledCheckBoxStateChanged(int state);
            void receiveHOGShape(QPointer<HOGHOF> partner_hoghof);
            void scoreCompute(PredData predScore);
            void receiveFrameRead(int64_t frameReadtime, int frameCount);
            void scoreCalculated(bool score_cal);
            void setSkipFrameProcess(unsigned int frameCount);
            void setgpuInitializeFlag();
            //void reloadButtonPressed();
            //void detectClicked();
            //void saveClicked();
            //void receiveFrameNum(unsigned int frameReadNum);
    };

}
#endif
