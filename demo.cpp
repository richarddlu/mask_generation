#include <iostream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

bool flag = false;
Mat img_show;
Mat mask;
Point* pre;

void mouseCallBackFunc(int event, int x, int y, int flags, void* userdata) {
	if(event == EVENT_LBUTTONDOWN) {
		flag = !flag;
		pre = NULL;
	}
	if(event == EVENT_MOUSEMOVE) {
		if(flag == true) {
			if(pre != NULL) {
				Point* cur = new Point(x, y);
				line(img_show, *pre, *cur, Scalar(255, 0, 0), 5);
				line(mask, *pre, *cur, Scalar(255), 5);
				pre = cur;
				imshow("demo", img_show);
				imshow("mask", mask);
			} else {
				pre = new Point(x, y);
			}
		}
	}
}

void matReshape32FC3(const Mat& src, Mat& dst, int numRows) {
	int numCols = src.size().height*src.size().width/numRows;
	dst.create(numRows, numCols, src.type());
	int row_count = 0;
	int col_count = 0;;
	for(int h = 0; h < src.size().height; h++) {
		for(int w = 0; w < src.size().width; w++) {
			dst.at<Vec3f>(row_count,col_count) = src.at<Vec3f>(h,w);
			col_count++;
			if(col_count >= numCols) {
				col_count = 0;
				row_count++;
			}
		}
	}
}

void imgBGRKMeans(const Mat& img, Mat& colors, int K) {
	Mat points_temp;
	if(img.isContinuous())
		points_temp = img.reshape(0, img.size().height*img.size().width);
	else
		matReshape32FC3(img, points_temp, img.size().height*img.size().width);
	Mat points;
	points_temp.convertTo(points, CV_32FC3);
	Mat labels;
	kmeans(points, K, labels, TermCriteria(TermCriteria::EPS+TermCriteria::COUNT,10,1.0), 3, KMEANS_PP_CENTERS, colors);
}

int main() {
	Mat img = imread("lena.jpg");
	img_show = img.clone();

	// initialize mask
	mask = Mat::zeros(img.size(), CV_8UC1);

	Mat colors;
	imgBGRKMeans(img, colors, 5);

	namedWindow("demo");
	namedWindow("mask");

	setMouseCallback("demo", mouseCallBackFunc, NULL);

	imshow("demo", img);
	imshow("mask", mask);

	waitKey(0);

	return 0;
}