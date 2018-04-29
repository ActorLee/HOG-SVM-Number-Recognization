#include <opencv2/opencv.hpp>  
 #include<fstream>
#include<iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>  
#include <opencv2/imgproc.hpp>  
#include "opencv2/imgcodecs.hpp"  
#include <opencv2/highgui.hpp>  
#include <opencv2/ml.hpp> 
#include<time.h>
#include<io.h>

using namespace std;
using namespace cv;
using namespace cv::ml;


int numberOfSample = 20;
int hog_dim = 1;
int w = 32,h = 32;
int nbins = 3;
int blocksize_w = 16;
int blocksize_h =16;
int b_stride_w = 8;
int b_stride_h = 8;
int cellsize_w = 8;
int cellsize_h = 8;



clock_t  img_start, img_end;
double imgprocess_time_train,imgprocess_time_test;



Mat image_processing(string path) {
	img_start = clock();
	vector<vector<int>> a;
	Mat imag, result;
	imag = imread(path, 0);    
	result = imag.clone();
	threshold(imag, result, 30, 1.0, CV_THRESH_BINARY);
	for (int i = 0; i < result.rows; i++)
	{
		for (int j = 0; j < result.cols; j++)
		{
			if (result.at<uchar>(i, j) == 1.0) {
				vector<int>b;
				b.push_back(i);
				b.push_back(j);
				a.push_back(b);


			}

		}
	}


	
	int maxh = 0, minh = 99, maxw = 0, minw = 99;
	for (int i = 0; i <a.size(); i++)
	{
		if (a[i][0] > maxh)maxh = a[i][0];
		if (a[i][0] < minh)minh = a[i][0];
		if (a[i][1] > maxw)maxw = a[i][1];
		if (a[i][1] < minw)minw = a[i][1];
	}
	Rect rect1(minw, minh, maxw - minw, maxh - minh);
	Mat img;
	img = imag(rect1);
	img_end = clock();
	imgprocess_time_train += double(img_end - img_start);
	return img;

}


Mat image_processing_mat(Mat path) {
	img_start = clock();
	vector<vector<int>> a;
	Mat imag, result;
	imag = path;    
	result = imag.clone();  
	threshold(imag, result, 30, 1.0, CV_THRESH_BINARY);
	for (int i = 0; i < result.rows; i++)
	{
		for (int j = 0; j < result.cols; j++)
		{
			if (result.at<uchar>(i, j) == 1.0) {
				vector<int>b;
				b.push_back(i);
				b.push_back(j);
				a.push_back(b);


			}

		}
	}



	int maxh = 0, minh = 99, maxw = 0, minw = 99;
	for (int i = 0; i <a.size(); i++)
	{
		if (a[i][0] > maxh)maxh = a[i][0];
		if (a[i][0] < minh)minh = a[i][0];
		if (a[i][1] > maxw)maxw = a[i][1];
		if (a[i][1] < minw)minw = a[i][1];
	}
	Rect rect1(minw, minh, maxw - minw, maxh - minh);
	Mat img;
	img = imag(rect1);
	img_end = clock();
	imgprocess_time_test += double(img_end-img_start);
	return img;
}



Mat gray(Mat src) {
	resize(src, src, Size(w,h));
	src.convertTo(src, CV_32F);
	sqrt(src, src);
	normalize(src, src, 0, 255, NORM_MINMAX, CV_8UC1);
	return src;
}




vector<float>  hog_path(string path) {
	HOGDescriptor hog(Size(w, h), Size(blocksize_w,blocksize_h), Size(b_stride_w, b_stride_h), Size(cellsize_w ,cellsize_h), nbins);
	Mat a = image_processing(path);
	Mat src = a;
	src=gray(src);
	vector<float> descriptors;//HOG描述子向量
	hog.compute(src, descriptors);
	hog_dim = descriptors.size();
	return descriptors;
}



vector<float>  hog_mat(Mat img) {
	HOGDescriptor hog(Size(w, h), Size(blocksize_w, blocksize_h), Size(b_stride_w, b_stride_h), Size(cellsize_w, cellsize_h), nbins);
	Mat a = image_processing_mat(img);
	Mat src = gray(a);
	vector<float> descriptors;//HOG描述子向量
	hog.compute(src, descriptors);
	hog_dim = descriptors.size();
	return descriptors;
}


Ptr<SVM>  train() {

    Mat featureVector(numberOfSample, hog_dim, CV_32FC1);
    Mat labelsMat(numberOfSample, 1, CV_32SC1);
	vector<string> imagePath;
	vector<int> imageClass;
	string line;
	ifstream trainingData;
	trainingData.open("D:\\datasets\\number\\label.txt");
	while (getline(trainingData, line)) {
		int index = 0;
		index = line.find(",", index);
		string a = "D:/datasets/number/number_crop/";
		a.append(line.substr(0, index));
		string b = line.substr(index + 1);
		int b_int = std::stoi(b);
		imagePath.push_back(a);
		imageClass.push_back(b_int);
	}

	for (int i = 0; i < imagePath.size(); i++) {
		vector<float> descriptors;
		descriptors = hog_path(imagePath[i]);
		for (vector<float>::size_type j = 0; j <= descriptors.size() - 1; ++j)
		{
			featureVector.at<float>(i, j) = descriptors[j];
		}
		labelsMat.at<int>(i, 0) = imageClass[i];
	}

	//*************************************************************************************
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 50, FLT_EPSILON));
	svm->train(featureVector, ROW_SAMPLE, labelsMat);
	svm->save("D:\\datasets\\number\\Detect.xml");
	cout << "train over"<< endl;
	return svm;
}


int detect(string path,Ptr<SVM> svm) {
	Mat test1(1,hog_dim,CV_32FC1);
	Mat test2(1,hog_dim, CV_32FC1);
	Mat test3(1, hog_dim, CV_32FC1);
	vector<float> descriptors1;
	vector<float> descriptors2;
	vector<float> descriptors3;
	Mat img = imread(path,0);
	Rect rect1(28, 105, 19, 50);
	Rect rect2(111, 104, 19, 52);
	Rect rect3(193, 104, 19, 54);
	Mat roi1 =img(rect1);
	Mat roi2 = img(rect2);
	Mat roi3 = img(rect3);
	descriptors1 = hog_mat(roi1);
	descriptors2 = hog_mat(roi2);
	descriptors3 = hog_mat(roi3);
	for (vector<float>::size_type j = 0; j <= descriptors1.size() - 1; ++j)
	{
		test1.at<float>(0, j) = descriptors1[j];
	}
	for (vector<float>::size_type j = 0; j <= descriptors2.size() - 1; ++j)
	{
		test2.at<float>(0, j) = descriptors2[j];
	}
	for (vector<float>::size_type j = 0; j <= descriptors3.size() - 1; ++j)
	{
		test3.at<float>(0, j) = descriptors3[j];
	}
	int *a=new int[3];
	a[0] = svm->predict(test1);
	a[1] = svm->predict(test2);
	a[2] = svm->predict(test3);
cout <<path<<"  "<< a[0]<<"  "<<a[1]<<"  "<<a[2] << endl;
	return 0;
}


int main()
{
	Mat img_init(Size(w,h),0);
	hog_mat(img_init);
	clock_t  startTime, endTime;
	startTime = clock();
	Ptr<SVM> svm = train();
	endTime = clock();
	cout << "hog_dim: " << hog_dim << endl;
	double totaltime = ((double)(endTime - startTime)-imgprocess_time_train) / CLOCKS_PER_SEC;
	cout << "train time: " << totaltime<<"s"<< endl;
	
	string line;
	ifstream testData;
	double alltime=0.000;
	testData.open("D:\\datasets\\number\\test.txt");
	while (getline(testData, line)) {
startTime = clock();
       detect("D:/datasets/number/all_pic/"+line,svm);
endTime = clock();
alltime += double(endTime - startTime);

	}
	
	totaltime =(alltime-imgprocess_time_test) / CLOCKS_PER_SEC;
	cout << "test time: " << totaltime<<"s"<< endl;
	
	cout << "image processiong time" << (imgprocess_time_train+imgprocess_time_test) / CLOCKS_PER_SEC <<"s"<< endl;
	system("pause");
	return 0;
}

