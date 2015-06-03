#include "color_constraint.h"

#include <iostream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

bool flag = false;
Mat img_show;
Mat mask;
Point* pre;

// Macro Constants
#define __VORBOSE__
#define NUM_EXPECTED_SAMPLES 54
#define SIGMA 0.02

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
			} else {
				pre = new Point(x, y);
			}
		}
	}
}

int main() {
	Mat img = imread("images/sea.jpg");
	img_show = img.clone();

	// initialize mask
	mask = Mat::zeros(img.size(), CV_8UC1);

	namedWindow("demo");
	namedWindow("new mask");

	setMouseCallback("demo", mouseCallBackFunc, NULL);

	imshow("demo", img);

	waitKey(0);

	Mat new_mask;
	colorConstraint(img, mask, new_mask, NUM_EXPECTED_SAMPLES, SIGMA, false, true);
	imshow("new mask", new_mask);

	waitKey(0);

	return 0;
}