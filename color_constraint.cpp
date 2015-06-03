#include "color_constraint.h"

void getSelectedColors(const Mat& img, const Mat& mask, vector<Vec3b>& selectedColors)
{
	selectedColors.clear();
	for(int h = 0; h < mask.size().height; h++) {
		for(int w = 0; w < mask.size().width; w++) {
			if(mask.at<uchar>(h,w) != 0) {
				selectedColors.push_back(img.at<Vec3b>(h,w));
			}
		}
	}
}

void sampleUniform(const vector<Vec3b>& selectedColors, vector<Vec3b>& samples, int NS)
{
	RNG rng(getTickCount());
	samples.clear();
	for(int i = 0; i < NS; i++) {
		int r_index = rng.uniform(0, selectedColors.size());
		samples.push_back(selectedColors[r_index]);
	}
}

// Matrix A must be empty.
void constructEquations(const vector<Vec3b>& selectedColors, const vector<Vec3b>& samples, Mat& A, Mat& b, double sigma)
{
	int numSelectedColors = selectedColors.size();
	int numSamples = samples.size();

	// Construct matrix A row by row
	for(int i = 0; i < numSelectedColors; i++) {
		Mat row(1, numSamples, CV_64F);
		Vec3b f = selectedColors[i];
		for(int j = 0; j < numSamples; j++) {
			Vec3b fi = samples[j];
			double r = norm(f, fi, NORM_L2);
			double rf = exp(-sigma * r * r);
			row.at<double>(0,j) = rf;
		}
		A.push_back(row);
	}

	// Construct matrix B
	b = Mat::ones(numSelectedColors, 1, CV_64F);
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

double RBF(const Mat&a, const vector<Vec3b>& f_i, Vec3b f, double sigma)
{
	double result = 0;
	for(int i = 0; i < f_i.size(); i++) {
		double r = norm(f, f_i[i], NORM_L2);
		double rf = a.at<double>(i,0) * exp(-sigma * r * r);
		result += rf;
	}
	return result;
}

void computeSimilarityMap(const Mat& img, const vector<Vec3b>& selectedColors, const vector<Vec3b>& samples, Mat& sMap, double sigma, bool useNonNegative)
{
	// Construct RBF equations
	Mat A, b;
	constructEquations(selectedColors, samples, A, b, sigma);
	// Solve RBF coefficients
	Mat a;
	if(useNonNegative)
		solveCoefficients(A, b, a);
	else
		solve(A, b, a, DECOMP_QR);
	// Construct similarity map
	sMap = Mat::ones(img.size(), CV_64F);
	for(int h = 0; h < img.size().height; h++) {
		for(int w = 0; w < img.size().width; w++) {
			sMap.at<double>(h,w) = RBF(a, samples, img.at<Vec3b>(h,w), sigma);
		}
	}
}

void matReshape(const Mat& src, Mat& dst, int numRows) {
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

void binarize(const Mat& sMap, Mat& mask)
{
	Mat sMapF;
	sMap.convertTo(sMapF, CV_32FC3);
	Mat points;
	if(sMapF.isContinuous())
		points = sMapF.reshape(0, sMapF.size().height*sMap.size().width);
	else
		matReshape(sMapF, points, sMapF.size().height*sMap.size().width);
	
	Mat labels;
	Mat colors;
	kmeans(points, 2, labels, TermCriteria(TermCriteria::EPS+TermCriteria::COUNT,1000,0.001), 3, KMEANS_PP_CENTERS, colors);
	int foregroundLabel = 0;
	if(colors.at<float>(1,0) >= colors.at<float>(0,0))
		foregroundLabel = 1;

	mask = Mat::zeros(sMap.size(), CV_8U);
	for(int h = 0; h < mask.size().height; h++) {
		for(int w = 0; w < mask.size().width; w++) {
			if(labels.at<int>(h*mask.size().width+w,0) == foregroundLabel)
				mask.at<uchar>(h,w) = 255;
		}
	}
}

void colorConstraint(const Mat& img, const Mat& mask, Mat& new_mask, int expectedNumSamples, double sigma, bool useNonNegative, bool binary)
{
	// Convert to YCrCb
	Mat imgYCrCb;
	cvtColor(img, imgYCrCb, CV_BGR2YCrCb);

	vector<Vec3b> selectedColors;
	getSelectedColors(imgYCrCb, mask, selectedColors);
	int numSelectedColors = selectedColors.size();
#ifdef __VORBOSE__
	cout<<"Number of Selected Colors: "<<numSelectedColors<<endl;
#endif
	if(numSelectedColors < 1) {
#ifdef __VORBOSE__
		cout<<"No Color Selected."<<endl;
#endif
		return;
	}
	int numSamples = min(numSelectedColors, expectedNumSamples);
#ifdef __VORBOSE__
	cout<<"Number of Samples: "<<numSamples<<endl;
#endif
	
	// sampling
	vector<Vec3b> samples;
	sampleUniform(selectedColors, samples, numSamples);

	// Compute similarity map
	Mat sMap;
	computeSimilarityMap(imgYCrCb, selectedColors, samples, sMap, sigma, useNonNegative);
	double min;
	double max;
	minMaxLoc(sMap, &min, &max);
#ifdef __VORBOSE__
	cout<<"Minimum Similarity is: "<<min<<endl;
	cout<<"Maximum Similarity is: "<<max<<endl;
#endif
	double diff = max - min;
	sMap.convertTo(sMap, CV_64F, 1.0, -min);
	sMap.convertTo(sMap, CV_64F, 1.0/diff);
	imshow("sMap", sMap);
	if(binary)
		binarize(sMap, new_mask);
	else
		new_mask = sMap;
}