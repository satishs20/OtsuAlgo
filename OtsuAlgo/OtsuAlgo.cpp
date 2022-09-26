
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

int main(int argc, char** argv) {

    // the input image
    cv::Mat srcImage;

    //output image, where we will save the results:
    cv::Mat outImage;

    //input image row/column
    int nCols = srcImage.cols;
    int nRows = srcImage.rows;
    int threshold = 0;

   

    // the total size of the image matrix (rows * columns * channels):
    size_t imageTotalSize = 0;
    // partial size (how many bytes will be sent to each process):
    size_t imagePartialSize = 0;
    // 'uchar' means 'unsigned char', i.e. an 8-bit value, because each pixel in an image is a byte (0..255)
    uchar* partialBuffer;

    
    std::clock_t start; //timer
    double duration;
    start = std::clock(); //curr time
   

    

    // get the world size and current rank:
    int size;
    int rank;
    // Initialise the MPI part
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    

   
    
    // read the image and its properties in the ROOT process:
    if (rank == 0)
    {
        //window for original grayscale image and otsu image
        namedWindow("srcGray", 0);
        resizeWindow("srcGray", 640, 480);
        namedWindow("otsuResultImage", 0);
        resizeWindow("otsuResultImage", 640, 480);

        //Reading the image
        cv::Mat srcGray;
        srcGray = cv::imread("khaw.jpg");
        
        //grayscaling the image
        cvtColor(srcGray, srcImage, cv::COLOR_RGB2GRAY);
        imshow("srcGray", srcImage);

        // check if image is empty:
        if (srcImage.empty())
        {
            std::cerr << "Image is empty, terminating!" << std::endl;
            return -1;
        }


        // get the total size of the image matrix (rows * columns * channels)
        imageTotalSize = srcImage.step[0] * srcImage.rows;


        // check if we can evenly divide the image bytes by the number of processes
        // the image.total() method returns the number of elements, i.e. (rows * cols)
        if (srcImage.total() % size)
        {
            std::cerr << "Cannot evenly divide the image between the processes. Choose a different number of processes!" << std::endl;
            return -2;
        }

        // get partial size (how many bytes are sent to each process):
        imagePartialSize = imageTotalSize / size;


        std::cout << "The image will be divided into blocks of " << imagePartialSize << " bytes each" << std::endl;
    }

    // send the "partial size" from #0 to other processes:
    MPI_Bcast(&imagePartialSize, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);


    // synchronize the processes here, to make sure that the sizes are initialized:
    MPI_Barrier(MPI_COMM_WORLD);


    // allocate the partial buffer:
    partialBuffer = new uchar[imagePartialSize];

    // synchronize the processe here, to make sure each process has allocated the buffer:
    MPI_Barrier(MPI_COMM_WORLD);

    // scatter the image between the processes:
    MPI_Scatter(srcImage.data, imagePartialSize, MPI_UNSIGNED_CHAR, partialBuffer, imagePartialSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

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
            uchar* B = &partialBuffer[j];
            if (j <= i) {
                w0 += B[j];
                u0_temp += j * B[j];
            }

            // foreground part
            else {
                w1 += B[j];
                u1_temp += j * B[j];
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
    // synchronize the image processing:
    MPI_Barrier(MPI_COMM_WORLD);

    // initialize the output image (only need to do it in the ROOT process)
    if (rank == 0)
    {
        outImage = cv::Mat(srcImage.size(), srcImage.type());
    }

   
   
    // and now finally send the partial buffers back to the ROOT, gathering the complete image:
    MPI_Gather(partialBuffer, imagePartialSize, MPI_UNSIGNED_CHAR, outImage.data, imagePartialSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
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
    
    
    // show the output 
    
        cout << "\n\n\n";
        
        duration = (std::clock()- start) / (double)CLOCKS_PER_SEC;
        std::cout << "Operation took: " << duration << " seconds using " << size << " cores" << std::endl;
        cv::imshow("otsuResultImage", outImage);
    }


    


    waitKey(0);
    delete[]partialBuffer;
    MPI_Finalize();
    return 0;
  
}


