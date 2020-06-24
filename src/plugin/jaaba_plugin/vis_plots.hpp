#ifndef VIS_PLOTS_HPP
#define VIS_PLOTS_HPP

#include "bias_plugin.hpp"
#include "qcustomplot.h"
#include <QFont>

namespace bias 
{

    class VisPlots : public QObject, public QRunnable, public Lockable<Empty>
    {

           
        Q_OBJECT
  
        public:

            static int DEFAULT_LIVEPLOT_UPDATE_DT;
            static double DEFAULT_LIVEPLOT_TIME_WINDOW;
            static double DEFAULT_LIVEPLOT_SIGNAL_WINDOW;

            QCustomPlot* livePlotPtr_;
            QVector<double> livePlotTimeVec_;
            QVector<double> livePlotSignalVec_Lift;
            QVector<double> livePlotSignalVec_Handopen;
            QVector<double> livePlotSignalVec_Grab;
            QVector<double> livePlotSignalVec_Chew;
            QVector<double> livePlotSignalVec_Supinate;
            QVector<double> livePlotSignalVec_Atmouth;
 
            VisPlots(QCustomPlot *livePlotPtr, QObject *parent);
            void stop();

        private:

            bool stopped_;        
            int livePlotUpdateDt_;
            double livePlotTimeWindow_;
            double livePlotSignalWindow_;
            
            QPointer<QTimer> livePlotUpdateTimerPtr_;
            //QElapsedTimer* livePlotUpdateTimerPtr_;

            void initialize();           
            void run();


    };

}

#endif

