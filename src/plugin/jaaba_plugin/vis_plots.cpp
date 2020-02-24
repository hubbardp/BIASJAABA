#include "vis_plots.hpp"
#include <cuda_runtime_api.h>
#include <iostream>

namespace bias {


    int VisPlots::DEFAULT_LIVEPLOT_UPDATE_DT = 200;
    double VisPlots::DEFAULT_LIVEPLOT_TIME_WINDOW = 10.0;
    double VisPlots::DEFAULT_LIVEPLOT_SIGNAL_WINDOW = 50.0;


    VisPlots::VisPlots(QCustomPlot *livePlotPtr, QObject *parent) : QObject(parent)
    {

        livePlotPtr_ = livePlotPtr;
        initialize();
    } 


    void VisPlots::initialize()
    {

        stopped_ = false;
        livePlotUpdateDt_ = DEFAULT_LIVEPLOT_UPDATE_DT;
        livePlotTimeWindow_ = DEFAULT_LIVEPLOT_TIME_WINDOW;
        livePlotSignalWindow_ = DEFAULT_LIVEPLOT_SIGNAL_WINDOW;

        //triggerEnabled = true;
        //triggerArmedState = true;

        // Setup live plot
        livePlotPtr_ -> addGraph();
        livePlotPtr_ -> addGraph();
        livePlotPtr_ -> addGraph();
        livePlotPtr_ -> addGraph();
        livePlotPtr_ -> addGraph();
        livePlotPtr_ -> addGraph();
        livePlotPtr_ -> graph(0) -> setName("Lift");
        livePlotPtr_ -> graph(1) -> setName("Handopen");
        livePlotPtr_ -> graph(2) -> setName("Grab");
        livePlotPtr_ -> graph(3) -> setName("Supinate");
        livePlotPtr_ -> graph(4) -> setName("Chew");
        livePlotPtr_ -> graph(5) -> setName("Atmouth"); 
        livePlotPtr_ -> graph(0) -> setPen(QPen(QColor(0,0,255,255),0.5));
        livePlotPtr_ -> graph(1) -> setPen(QPen(QColor(0,255,0,255),0.5));
        livePlotPtr_ -> graph(2) -> setPen(QPen(QColor(255,0,0,255),0.5));
        livePlotPtr_ -> graph(3) -> setPen(QPen(QColor(0,255,255,255),0.5));
        livePlotPtr_ -> graph(4) -> setPen(QPen(QColor(255,0,255,255),0.5));
        livePlotPtr_ -> graph(5) -> setPen(QPen(QColor(100,100,100,255),0.5));
        livePlotPtr_ -> xAxis -> setRange(-livePlotTimeWindow_,0);
        livePlotPtr_ -> yAxis -> setRange(-livePlotSignalWindow_, livePlotSignalWindow_);
        livePlotPtr_ -> xAxis -> setLabel("time (sec)");

        //legend
        livePlotPtr_->legend->setVisible(true);
        QFont legendFont = QFont();
        legendFont.setPointSize(10);
        livePlotPtr_->legend->setFont(legendFont);
        livePlotPtr_->legend->setSelectedFont(legendFont);
        livePlotPtr_->legend->setSelectableParts(QCPLegend::spItems); 
        livePlotPtr_ ->setAutoAddPlottableToLegend(true); 
        livePlotPtr_ -> replot();

        livePlotUpdateTimerPtr_ = new QTimer(this);
        livePlotUpdateTimerPtr_ -> start(livePlotUpdateDt_);
        connect(livePlotUpdateTimerPtr_, SIGNAL(timeout()), this, SLOT(stop()));

    }


    void VisPlots::stop()
    {

        livePlotUpdateTimerPtr_ -> stop();
        stopped_ = true;

    }
 

    void VisPlots::run()
    {

        bool done = false;

        // Set thread priority to idle - only run when no other thread are running
        //QThread *thisThread = QThread::currentThread();
        //thisThread -> setPriority(QThread::NormalPriority);

        acquireLock();
        stopped_ = false;
        releaseLock();
   
        long timer=0; 
        while(!done)
        {

            if(livePlotUpdateTimerPtr_ -> isActive())  
            {

                if(timer < livePlotUpdateDt_)
                {

                    timer++;

                } else {


                    if(livePlotTimeVec_.empty())
		    {
                        timer=0;   
			continue;
		    }

		    acquireLock();
		    double lastTime = livePlotTimeVec_.last();
		    double firstTime = livePlotTimeVec_.first();

		    if (lastTime < firstTime)
		    {
                        timer=0;
			livePlotTimeVec_.clear();
			livePlotSignalVec_Lift.clear();
			livePlotSignalVec_Handopen.clear();
			livePlotSignalVec_Grab.clear();
			livePlotSignalVec_Atmouth.clear();
			livePlotSignalVec_Supinate.clear();
			livePlotSignalVec_Chew.clear();
			releaseLock();
			continue;
		    }

		    while (lastTime - firstTime > livePlotTimeWindow_)
		    {
			livePlotTimeVec_.pop_front();
			livePlotSignalVec_Lift.pop_front();
			livePlotSignalVec_Handopen.pop_front();
			livePlotSignalVec_Grab.pop_front();
			livePlotSignalVec_Atmouth.pop_front();
			livePlotSignalVec_Supinate.pop_front();
			livePlotSignalVec_Chew.pop_front();
			firstTime = livePlotTimeVec_.first();
		    }

		    if (lastTime < livePlotTimeWindow_)
		    {
			double windowStartTime= -livePlotTimeWindow_ + lastTime;
			livePlotPtr_ -> xAxis -> setRange(windowStartTime,lastTime);

		    }
		    else
		    {
			livePlotPtr_ -> xAxis -> setRange(firstTime, lastTime);

		    }

                   
		    livePlotPtr_ -> graph(0) -> addData(livePlotTimeVec_,livePlotSignalVec_Lift);
		    livePlotPtr_ -> graph(1) -> addData(livePlotTimeVec_,livePlotSignalVec_Handopen);
		    livePlotPtr_ -> graph(2) -> addData(livePlotTimeVec_,livePlotSignalVec_Grab);
		    livePlotPtr_ -> graph(3) -> addData(livePlotTimeVec_,livePlotSignalVec_Supinate);
		    livePlotPtr_ -> graph(4) -> addData(livePlotTimeVec_,livePlotSignalVec_Chew);
                    livePlotPtr_ -> graph(5) -> addData(livePlotTimeVec_, livePlotSignalVec_Atmouth);
		    livePlotPtr_ -> replot();
                    timer=0;
		    releaseLock();
		    
                }  

	    }

            acquireLock();
            done = stopped_;
            releaseLock();
 
        }

    }

}
