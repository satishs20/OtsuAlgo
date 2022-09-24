// install opencv library
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "omp.h"
#include "mpi.h"


// install cpp library
#include <string>
#include <stdio.h>

// create namespace
using namespace std;
using namespace cv;

#define Gamma 3

// Otsu Implemented

int OTSU(cv::Mat srcImage, int argc, char** argv) {
	int nCols = srcImage.cols;
	int nRows = srcImage.rows;
	int threshold = 0;

	namedWindow("outImage", 0);
	resizeWindow("outImage", 640, 480);

	// the total size of the image matrix (rows * columns * channels):
	size_t imageTotalSize;

	// partial size (how many bytes will be sent to each process):
	size_t imagePartialSize;




	// partial buffer, to contain the image.
    // 'uchar' means 'unsigned char', i.e. an 8-bit value, because each pixel in an image is a byte (0..255)
	uchar* partialBuffer;


	// start the MPI part
	MPI_Init(&argc, &argv);

	// get the world size and current rank:
	int size;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	

	


	// get the total size of the image matrix (rows * columns * channels)
	imageTotalSize = srcImage.step[0] * nRows;

	// get partial size (how many bytes are sent to each process):
	imagePartialSize = imageTotalSize / size;


	// allocate the partial buffer:
	partialBuffer = new uchar[imagePartialSize];



	// scatter the image between the processes:
	MPI_Scatter(srcImage.data, imagePartialSize, MPI_UNSIGNED_CHAR, partialBuffer, imagePartialSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
	/*MPI_Scatter(void* send_data,int send_count,MPI_Datatype send_datatype,void* recv_data,int recv_count,MPI_Datatype recv_datatype,int root,MPI_Comm communicator)*/
	
	// synchronize the image processing:
	MPI_Barrier(MPI_COMM_WORLD);


	// Traverse the gray level [0, 255] and calculate the value under the maximum inter-class variance

	float w0, w1, u0_temp, u1_temp, u0, u1, delta_temp;
	double delta_max = 0.0;

	for (int i = 0; i < 256; i++) {

		// initial variable
		w0 = w1 = u0 = u1 = u0_temp = u1_temp = delta_temp = 0;

		for (int j = 0; j < 256; j++) {
			// background part

			if (j <= i) {
				w0 += partialBuffer[j];;
				u0_temp += j * partialBuffer[j];;
			}

			// foreground part
			else {
				w1 += partialBuffer[j];
				u1_temp += j * partialBuffer[j];;
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

	//output image
	cv::Mat outImage;

	// initialize the output image (only need to do it in the ROOT process)
	if (rank == 0)
	{
		outImage = cv::Mat(srcImage.size(), srcImage.type(), CV_8UC1);
	}

	// use the obtained value to perform a binarization operation
	for (int i = 0; i < srcImage.rows; i++) {
		for (int j = 0; j < srcImage.cols; j++) {

			// high pixel value judgment
			if (srcImage.at<uchar>(i, j) > threshold) {
				outImage.at<uchar>(i, j) = 255;
			}
			else
			{
				outImage.at<uchar>(i, j) = 0;
			}
		}
	}

	
	
	// and now we finally send the partial buffers back to the ROOT, gathering the complete image:
	MPI_Gather(outImage.data, imagePartialSize, MPI_UNSIGNED_CHAR, partialBuffer, imagePartialSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	// Display image, only in the ROOT process
	if (rank == 0)
	{
		cv::imshow( "outImage", outImage );
		
	}
	
	return threshold;
}

int main(int argc, char** argv)
{
	double itime, ftime, exec_time;
	itime = omp_get_wtime();
	namedWindow("srcGray", 0);
	resizeWindow("srcGray", 640, 480);


	// Reading and judgment the image

	cv::Mat srcImage;
	srcImage = cv::imread("S.jpg");
	if (!srcImage.data) {
		return -1;
	}
	cv::Mat srcGray;
	cvtColor(srcImage, srcGray, cv::COLOR_RGB2GRAY);
	imshow("srcGray", srcGray);

	// call the otsu algorithm to get the image
	OTSU(srcGray, argc, argv);

	// declared the result output of image
	

	// use the obtained value to perform a binarization operation
	

	ftime = omp_get_wtime();
	exec_time = ftime - itime;

	cout << "\n\n\n";
	cout << "Total Final Result";
	cout << "==================";
	cout << "hello!!";
	printf("\n\nTime taken is %f", exec_time);
	waitKey(0);

	
	return 0;
}
