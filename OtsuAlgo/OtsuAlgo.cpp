// install opencv library
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "omp.h"

// install cpp library
#include <string>
#include <stdio.h>

// create namespace
using namespace std;
using namespace cv;

#define Gamma 3

// Otsu Implemented

int OTSU(cv::Mat srcImage) {
	int nCols = srcImage.cols;
	int nRows = srcImage.rows;
	int threshold = 0;

	// init the parameters

	int nSumPix[256];
	float nProDis[256];
	for (int i = 0; i < 256; i++) {
		nSumPix[i] = 0;
		nProDis[i] = 0;
	}

	// Count the number of every pixel in the whole painting in the grayscale set

	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			nSumPix[(int)srcImage.at<uchar>(i, j)]++;
		}
	}

	// Calculate the probability distribution that each gray level accounts for the image

	for (int i = 0; i < 256; i++) {
		nProDis[i] = (float)nSumPix[i] / (nCols * nRows);
	}

	// Traverse the gray level [0, 255] and calculate the value under the maximum inter-class variance

	float w0, w1, u0_temp, u1_temp, u0, u1, delta_temp;
	double delta_max = 0.0;

	for (int i = 0; i < 256; i++) {

		// initial variable
		w0 = w1 = u0 = u1 = u0_temp = u1_temp = delta_temp = 0;

		for (int j = 0; j < 256; j++) {
			// background part

			if (j <= i) {
				w0 += nProDis[j];
				u0_temp += j * nProDis[j];
			}

			// foreground part
			else {
				w1 += nProDis[j];
				u1_temp += j * nProDis[j];
			}
		}

		// Calculate an average of 2 types grayscale
		u0 = u0_temp / w0;
		u1 = u1_temp / w1;

		// Find the data under the maximum between-class variance at once
		delta_temp = (float)(w0 * w1 * pow((u0 - u1), 2)); // variance between foreground and background

		if (delta_temp > delta_max) {
			delta_max = delta_temp;
			threshold = i;
		}
	}
	return threshold;
}

int main()
{
	double itime, ftime, exec_time;
	itime = omp_get_wtime();
	namedWindow("srcGray", 0);
	resizeWindow("srcGray", 640, 480);
	namedWindow("otsuResultImage", 0);
	resizeWindow("otsuResultImage", 640, 480);
	namedWindow("dst", 0);
	resizeWindow("dst", 640, 480);

	// Reading and judgment the image

	cv::Mat srcImage;
	srcImage = cv::imread("khaw.jpg");
	if (!srcImage.data) {
		return -1;
	}
	cv::Mat srcGray;
	cvtColor(srcImage, srcGray, cv::COLOR_RGB2GRAY);
	imshow("srcGray", srcGray);

	// call the otsu algorithm to get the image
	int otsuThreshold = OTSU(srcGray);

	// declared the result output of image
	cv::Mat otsuResultImage = cv::Mat::zeros(srcGray.rows, srcGray.cols, CV_8UC1);

	// use the obtained value to perform a binarization operation
	for (int i = 0; i < srcGray.rows; i++) {
		for (int j = 0; j < srcGray.cols; j++) {

			// high pixel value judgment
			if (srcGray.at<uchar>(i, j) > otsuThreshold) {
				otsuResultImage.at<uchar>(i, j) = 255;
			}
			else
			{
				otsuResultImage.at<uchar>(i, j) = 0;
			}
		}
	}

	ftime = omp_get_wtime();
	exec_time = ftime - itime;

	cout << "\n\n\n";
	cout << "Total Final Result";
	cout << "==================";
	cout << "hello!!";
	printf("\n\nTime taken is %f", exec_time);

	// show the output 
	cv::imshow("otsuResultImage", otsuResultImage);
	waitKey(0);
	return 0;
}