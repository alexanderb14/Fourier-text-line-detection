/*
 *      Author: alexanderb
 */

#include <stdio.h>
#include <vector>
#include <complex>
#include <memory>

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "util.h"

using namespace cv;
using namespace std;

//-----------------------------------------------------------------------------------------------
std::vector<float> getOrderedValues(Mat _mat) {
    Mat planes[2];
    cv::split(_mat, planes);

    std::vector<float> pointValues;
    for (int r = 0; r < _mat.rows; r++) {
        for (int c = 0; c < _mat.cols; c++) {
            pointValues.push_back(planes[0].at< std::complex<float> >(r,c).real());
            pointValues.push_back(planes[0].at< std::complex<float> >(r,c).imag());
        }
    }
    std::sort(pointValues.begin(), pointValues.end(), by_pointValue());

    return pointValues;
}

//-----------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
    // read image from file + error handling
    Mat I;

    if (argc == 1) {
        cout << "No image provided! Usage: '" <<  argv[0] << " [path to image]'" << endl << "Using default image now: txt.jpg" << endl;

        I = imread("txt.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    } else {
        I = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    }
    imshow("Input Image", I);    // Show the result


    // (1) Compute DFT
    Mat planes[] = {Mat_<float>(I), Mat::zeros(I.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);

    dft(complexI, complexI);


    // (2) Preserve maxima only
    Mat thresholdedI;
    complexI.copyTo(thresholdedI);
    thresholdedI = cv::Scalar(255);

    vector<float> orderedValues = getOrderedValues(complexI);
    int thres = orderedValues[1]/3;

    for (int r = 0; r < complexI.rows; r++) {
        for (int c = 0; c < complexI.cols; c++) {
            complex<float> val = complexI.at< complex<float> >(r,c);
            if(val.imag() < thres && val.real() < thres) {
                thresholdedI.at< complex<float> >(r,c) = 0;
            } else {
                thresholdedI.at< complex<float> >(r,c) = val;
            }
        }
    }


    // (3) Compute iDFT on filtered complex image
    Mat rI;
    cv::idft(thresholdedI, rI, cv::DFT_REAL_OUTPUT|cv::DFT_SCALE);
    rI.convertTo(rI, CV_8U);
    // imshow("idft", rI);

    // (4) Apply Threshold to get binary result
    double min, max;
    cv::minMaxLoc(rI, &min, &max);

    cv::threshold(rI, rI, max*0.82, 255, cv::THRESH_BINARY_INV);
    // imshow("rI", rI);

    // (5) Draw bounding boxes
    Mat finalI;
    cvtColor(I, finalI, CV_GRAY2RGB);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours( rI, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    vector<RotatedRect> minRect( contours.size() );

    for (int i = 0; i < contours.size(); i++) { 
        minRect[i] = minAreaRect( Mat(contours[i]) );
    }

    for (int i = 0; i< contours.size(); i++) {
        Point2f rect_points[4]; minRect[i].points( rect_points );

        if (contourArea(contours[i]) > contours.size())
            for (int j = 0; j < 4; j++)
                line( finalI, rect_points[j], rect_points[(j+1)%4], cv::Scalar(0, 0, 255), 2, 8 ); 
    } 

    imshow("finalI", finalI);

    waitKey();
    return 0;
}
