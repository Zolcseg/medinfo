/**
 * @function calcHist_Demo.cpp
 * @brief Demo code to use the function calcHist
 * @author
 */

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

#include "utilities.h"

using namespace std;
using namespace cv;

/**
 * @function main
 */


int main(int argc, char** argv)
{
	//! [Load image]
	CommandLineParser parser(argc, argv, "{@input | ../data/lena.jpg | input image}");
	Mat src = imread(parser.get<String>("@input"), IMREAD_COLOR);
	if (src.empty())
	{
		return -1;
	}
	//! [Load image]


	// pixel szintu manipulacio
	Mat pxl_img;
	src.copyTo(pxl_img);
	cv::MatIterator_<cv::Vec3b> it, end;
	for (it = pxl_img.begin<cv::Vec3b>(), end = pxl_img.end<cv::Vec3b>(); it != end; ++it) {
		uchar &r = (*it)[2];
		uchar &g = (*it)[1];
		uchar &b = (*it)[0];
		// Modify r, g, b values
		r = 255 - r;
		g = 255 - g;
		b = 255 - b;
	}

	Mat bitwise;
	Mat grey;
	cvtColor(src, grey, cv::COLOR_RGB2GRAY);

	Mat dst;
	equalizeHist(grey, dst);

	bitwise_not(src, bitwise);

	imshow("Source image", src);
	imshow("bitwise inverted", bitwise);
	imshow("coverted grayscale", grey);
	imshow("Extended grayscale img", extended_hist_image(src));
	imshow("Equalized grayscale img", dst);

	imshow("calcHist headerbol", create_histogram(src));


	waitKey();
	//! [Display]

	return 0;
}
