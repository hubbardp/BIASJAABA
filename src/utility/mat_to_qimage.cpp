#include "mat_to_qimage.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


namespace bias
{
    static QVector<QRgb> colorTable = createColorTable();

    QImage matToQImage(const cv::Mat& mat)
    {
        if (mat.type()==CV_8UC1)
        {
            const uchar *qImageBuffer = (const uchar*) mat.data;
            QImage img = QImage(
                    qImageBuffer, 
                    static_cast<int>(mat.cols), 
                    static_cast<int>(mat.rows), 
                    static_cast<int>(mat.step), 
                    QImage::Format_Indexed8
                    );

            img.setColorTable(colorTable);
            return img;
        }
        else if (mat.type()==CV_16UC1)
        {
            cv::Mat matBGR = cv::Mat(mat.size(), CV_8UC3, cv::Scalar(0,0,0));
            //cvtColor(mat,matBGR,CV_GRAY2BGR);
            cvtColor(mat, matBGR, cv::COLOR_GRAY2BGR);
            const uchar *qImageBuffer = (const uchar*)mat.data;
            QImage img = QImage(
                    qImageBuffer, 
                    static_cast<int>(matBGR.cols), 
                    static_cast<int>(matBGR.rows), 
                    static_cast<int>(matBGR.step), 
                    QImage::Format_RGB888
                    );
            return img.rgbSwapped();
        }
        else if (mat.type()==CV_8UC3)
        {
            const uchar *qImageBuffer = (const uchar*)mat.data;
            QImage img = QImage(
                    qImageBuffer, 
                    static_cast<float>(mat.cols), 
                    static_cast<float>(mat.rows), 
                    static_cast<float>(mat.step), 
                    QImage::Format_RGB888
                    );
            return img.rgbSwapped();
        }
        else
        {
            // --------------------------------------------------------------------
            // TO DO ... need some error handling here
            // --------------------------------------------------------------------
            return QImage();
        }
    }

    QVector<QRgb> createColorTable() 
    {
        QVector<QRgb> colorTable;
        for (int i=0; i<256; i++)
        {
            colorTable.push_back(qRgb(i,i,i));
        }
        return colorTable;
    }

} // namespace bias

