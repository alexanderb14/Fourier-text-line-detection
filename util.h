/*
 *      Author: alexanderb
 */

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct pointData { 
    float value;

    Point point;
};

struct by_cornerResponse { 
    bool operator()(pointData const &left, pointData const &right) { 
        return left.value > right.value;
    }
};

struct by_pointValue { 
    bool operator()(float const &left, float const &right) { 
        return left > right;
    }
};

struct Derivatives {
	Mat Ix;
	Mat Iy;
	Mat Ixy;
};

class Util {
public:
	static void DisplayImage(Mat& img);
	static void DisplayMat(Mat& img);
	static void DisplayPointVector(vector<Point> vp);
  static void DisplayFloatVector(vector<float> vf);

	static Mat MarkInImage(Mat& img, vector<pointData> points, int radius);
};
