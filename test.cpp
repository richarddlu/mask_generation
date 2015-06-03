#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(){

	Mat A(2,2,CV_8U);
	A.at<uchar>(0,0) = 1;
	A.at<uchar>(0,1) = 2;
	A.at<uchar>(1,0) = 3;
	A.at<uchar>(1,1) = 4;
	Mat B = A.reshape(0, 4);
	cout<<B<<endl;

	return 0;
}