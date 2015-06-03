#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include "nnls.h"

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

void imgKMeans(const Mat& img, Mat& colors, int K, Mat& labels) {
	Mat points_temp;
	if(img.isContinuous())
		points_temp = img.reshape(0, img.size().height*img.size().width);
	else
		matReshape32FC3(img, points_temp, img.size().height*img.size().width);
	Mat points;
	points_temp.convertTo(points, CV_32FC3);
	kmeans(points, K, labels, TermCriteria(TermCriteria::EPS+TermCriteria::COUNT,1000,1.0), 3, KMEANS_PP_CENTERS, colors);
}

void calculatePDF(const Mat& labels, vector<float>& PDF, int K) {
	vector<int> counts(K);
	PDF.clear();
	PDF.resize(K);
	for(int i = 0; i < K; i++) {
		counts[i] = 0;
		PDF[i] = 0;
	}
	for(int i = 0; i < labels.size().height; i++) {
		counts[labels.at<int>(i,1)]++;
	}
	for(int i = 0; i < K; i++)
		PDF[i] = counts[i] / (float)labels.size().height;
}

void sample(const Mat& img, const Mat& mask, const Mat& labels, const vector<float>& PDF, vector<Vec3b>& samples, int NS) {
	vector<int> labels_sele;
	vector<Vec3b> colors_sele;
	for(int h = 0; h < mask.size().height; h++) {
		for(int w = 0; w < mask.size().width; w++) {
			if(mask.at<uchar>(h,w) != 0) {
				labels_sele.push_back(labels.at<int>(h*mask.size().width+w,0));
				colors_sele.push_back(img.at<Vec3b>(h,w));
			}
		}
	}
	
	vector<vector<Vec3b> > colors_sele_cluskters;
	vector<bool> labels_visited(labels_sele.size());
	vector<float> PDF_sele;
	vector<float> PMF_sele_scaled;
	for(int i = 0; i < labels_visited.size(); i++) {
		labels_visited[i] = false;
	}
	int visit_count = 0;
	while(visit_count < labels_visited.size()) {
		int cur_label = -1;
		vector<Vec3b> colors_sele_cluskter;
		for(int i = 0; i < labels_sele.size(); i++) {
			if(labels_visited[i] == false) {
				if(cur_label == -1) {
					cur_label = labels_sele[i];
					colors_sele_cluskter.push_back(colors_sele[i]);
					PDF_sele.push_back(PDF[cur_label]);
					labels_visited[i] = true;
					visit_count++;
				} else {
					if(labels_sele[i] == cur_label) {
						colors_sele_cluskter.push_back(colors_sele[i]);
						labels_visited[i] = true;
						visit_count++;
					}
				}
			}
		}
		colors_sele_cluskters.push_back(colors_sele_cluskter);
	}

	//scale PDF in selected region
	float sum_pdf_sele = 0;
	for(int i = 0; i < PDF_sele.size(); i++) {
		sum_pdf_sele += PDF_sele[i];
	}
	for(int i = 0; i < PDF_sele.size(); i++) {
		PDF_sele[i] *= 1 / sum_pdf_sele;
	}
	for(int i = 0; i < PDF_sele.size(); i++) {
		if(i == 0)
			PMF_sele_scaled.push_back(PDF_sele[i]);
		else
			PMF_sele_scaled.push_back(PDF_sele[i]+PMF_sele_scaled[i-1]);
	}

	// Importance sampling
	RNG rng(45438978);
	samples.clear();
	for(int i = 0; i < NS; i++) {
		double rn = rng.uniform(0.0,1.0);
		int index = -1;
		for(int j = 0; j < PMF_sele_scaled.size(); j++) {
			if(rn < PMF_sele_scaled[j]) {
				index = j;
				break;
			}
		}
		if(index == -1)	// if the last PMF is less than 1 caused by floating error
			index = PMF_sele_scaled.size() - 1;
		int r_index = rng.uniform(0, colors_sele_cluskters[index].size());
		samples.push_back(colors_sele_cluskters[index][r_index]);
	}
}

void sampleUniform(const Mat& img, const Mat& mask, vector<Vec3b>& samples, int NS) {
	vector<Vec3b> selectedColors;
	for(int h = 0; h < mask.size().height; h++) {
		for(int w = 0; w < mask.size().width; w++) {
			if(mask.at<uchar>(h,w) != 0) {
				selectedColors.push_back(img.at<Vec3b>(h,w));
			}
		}
	}

	RNG rng(454387648);
	samples.clear();
	for(int i = 0; i < NS; i++) {
		int r_index = rng.uniform(0, selectedColors.size());
		samples.push_back(selectedColors[r_index]);
	}
}

// Matrix A must be empty.
void constructEquations(const Mat& img, const Mat& mask, const vector<Vec3b>& samples, Mat& A, Mat& b, double sigma)
{
	int numSamples = samples.size();

	// Construct matrix A row by row
	int nMP = 0;	// number of mask pixels
	for(int h = 0; h < img.size().height; h++) {
		for(int w = 0; w < img.size().width; w++) {
			if(mask.at<uchar>(h,w) != 0) {
				nMP++;
				Mat row(1, numSamples, CV_64FC1);
				Vec3b f = img.at<Vec3b>(h,w);
				for(int i = 0; i < numSamples; i++) {
					Vec3b fi = samples[i];
					double r = norm(f - fi);
					double rf = exp(-sigma * r * r);
					row.at<double>(0,i) = rf;
				}
				A.push_back(row);
			}
		}
	}

	// Construct matrix B
	b = Mat::ones(nMP, 1, CV_64F);
}

double RBF(const Mat&a, const vector<Vec3b>& f_i, Vec3b f, double sigma)
{
	double result = 0;
	for(int i = 0; i < f_i.size(); i++) {
		double r = norm(f - f_i[i]);
		double rf = a.at<double>(i,0) * exp(-sigma * r * r);
		result += rf;
	}
	return result;
}

void solveCoefficients(const Mat& A, const Mat& b, Mat& a)
{
	// Construct mda, m and n
	int m = A.size().height;
	int n = A.size().width;
	int mda = m;

	// Construct array for A
	double* arrA;
	if(A.isContinuous()) {
		arrA = (double*)A.data;
	}
	else {
		arrA = (double*)malloc(m*n*8);
		for(int h = 0; h < m; h++)
			memcpy(&(arrA[h*n]), A.ptr<double>(h), n*8);
	}

	// Contruct array for b
	double* arrb;
	if(b.isContinuous()) {
		arrb = (double*)b.data;
	}
	else {
		arrb = (double*)malloc(m*8);
		for(int h = 0; h < m; h++)
			arrb[h] = b.at<double>(h,0);
	}

	// Construct array for a
	double* arra;
	arra = (double*)malloc(n*8);

	// COnstruct working space array
	double rnorm;
	double* w = (double*)malloc(n*8);
	double* zz = (double*)malloc(m*8);
	int* index = (int*)malloc(n*4);

	// Call nnls
	int mode;
	nnls(arrA, mda, m, n, arrb, arra, &rnorm, w, zz, index, &mode, 1000);
	cout<<mode<<endl;

	// Construct return matrix a
	a = *(new Mat(n, 1, CV_64F, arra));
}

void computeSimilarityMap(const Mat& img, const Mat& mask, const vector<Vec3b>& samples, Mat& sMap)
{
	// Construct RBF equations
	Mat A, b;
	constructEquations(img, mask, samples, A, b, 42.5);
	// Solve RBF coefficients
	Mat a;
	solve(A, b, a, DECOMP_SVD);
	// solveCoefficients(A, b, a);
	cout<<a<<endl;
	// Construct similarity map
	sMap = Mat::ones(img.size(), CV_64F);
	for(int h = 0; h < img.size().height; h++) {
		for(int w = 0; w < img.size().width; w++) {
			sMap.at<double>(h,w) = RBF(a, samples, img.at<Vec3b>(h,w), 42.5);
		}
	}
}

int main() {
	Mat img = imread("sky.jpg");
	img_show = img.clone();

	// initialize mask
	mask = Mat::zeros(img.size(), CV_8UC1);

	namedWindow("demo");
	namedWindow("mask");
	namedWindow("sMap");

	setMouseCallback("demo", mouseCallBackFunc, NULL);

	imshow("demo", img);
	imshow("mask", mask);

	waitKey(0);

	// Convert to YCrCb
	// cvtColor(img, img, CV_BGR2YCrCb);

	Mat centers;
	Mat labels;
	imgKMeans(img, centers, 27, labels);

	vector<float> PDF;
	calculatePDF(labels, PDF, 27);
	
	vector<Vec3b> samples;
	// sample(img, mask, labels, PDF, samples, 54);
	sampleUniform(img, mask, samples, 54);

	// Compute similarity map
	Mat sMap;
	computeSimilarityMap(img, mask, samples, sMap);
	double min;
	double max;
	minMaxLoc(sMap, &min, &max);
	cout<<min<<endl;
	cout<<max<<endl;
	double diff = max - min;
	sMap.convertTo(sMap, CV_64F, 1.0/diff, -min/diff);
	imshow("sMap", sMap);

	waitKey(0);

	return 0;
}