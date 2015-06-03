#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include "nnls.h"

using namespace std;
using namespace cv;

void colorConstraint(const Mat& img, const Mat& mask, Mat& new_mask, int expectedNumSamples, double sigma, bool useNonNegative, bool binary);
