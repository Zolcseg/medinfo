#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

Mat invert_img(Mat const& input)
{
	return cv::Scalar::all(255) - input;
}

Mat extended_hist_image(Mat image_src) {

	Mat grey;
	cvtColor(image_src, grey, cv::COLOR_RGB2GRAY);
	imshow("stuff", grey);

	//! [Establish the number of bins]
	int histSize = 256;
	//! [Establish the number of bins]

	float range[] = { 0, 256 }; //the upper boundary is exclusive
	const float* histRange = { range };

	//! [Set histogram param]
	bool uniform = true, accumulate = false;
	//! [Set histogram param]

	Mat hist;
	calcHist(&grey, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	uint min = 0;
	for (int i = 0; i < 256; i++) {
		if (hist.at<int>(i) != 0) { min = i; break; }
	}
	uint max = 255;
	for (int i = 255; i > 0; i--) {
		if (hist.at<int>(i) != 255) { max = i; break; }
	}

	Mat grey_extended;
	grey.copyTo(grey_extended);
	int rows = grey_extended.rows;
	int cols = grey_extended.cols;
	for (int x = 0; x < cols; x++) {
		for (int y = 0; y < rows; y++) {
			uint in = grey_extended.at<uchar>(y, x);
			float result = (in - min) * 255 / (max - min);
			grey_extended.at<uchar>(y, x) = (uint)result;
		}
	}

	return grey_extended;
}


Mat create_histogram(Mat image_src) {

	//! [Separate the image in 3 places ( B, G and R )]
	vector<Mat> bgr_planes;
	split(image_src, bgr_planes);
	//! [Separate the image in 3 places ( B, G and R )]

	//! [Establish the number of bins]
	int histSize = 256;
	//! [Establish the number of bins]

	//! [Set the ranges ( for B,G,R) )]
	float range[] = { 0, 256 }; //the upper boundary is exclusive
	const float* histRange = { range };
	//! [Set the ranges ( for B,G,R) )]

	//! [Set histogram param]
	bool uniform = true, accumulate = false;
	//! [Set histogram param]

	//! [Compute the histograms]
	Mat b_hist, g_hist, r_hist;
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	//! [Compute the histograms]

	//! [Draw the histograms for B, G and R]
	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	//! [Draw the histograms for B, G and R]

	//! [Normalize the result to ( 0, histImage.rows )]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	//! [Normalize the result to ( 0, histImage.rows )]

	//! [Draw for each channel]
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	return histImage;
}

Mat create_grey_histogram(Mat image_src) {

	//! [Establish the number of bins]
	int histSize = 256;
	//! [Establish the number of bins]

	//! [Set the ranges ( for B,G,R) )]
	float range[] = { 0, 256 }; //the upper boundary is exclusive
	const float* histRange = { range };
	//! [Set the ranges ( for B,G,R) )]

	//! [Set histogram param]
	bool uniform = true, accumulate = false;
	//! [Set histogram param]

	//! [Compute the histograms]
	Mat  g_hist;
	calcHist(&image_src, 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);

	//! [Compute the histograms]

	//! [Draw the histograms for B, G and R]
	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));

	//! [Normalize the result to ( 0, histImage.rows )]
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	//! [Normalize the result to ( 0, histImage.rows )]

	//! [Draw for each channel]
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	return histImage;
}

