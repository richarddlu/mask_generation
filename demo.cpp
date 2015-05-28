#include <opencv2/opencv.hpp>

// using namespace std;
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

int main() {
	Mat img = imread("lena.jpg");
	img_show = img.clone();

	// initialize mask
	mask = Mat::zeros(img.size(), CV_8UC1);

	namedWindow("demo");
	namedWindow("mask");

	setMouseCallback("demo", mouseCallBackFunc, NULL);

	imshow("demo", img);
	imshow("mask", mask);

	waitKey(0);

	return 0;
}