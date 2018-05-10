#include <opencv2/opencv.hpp>  
 #include<fstream>
#include<iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>  
#include <opencv2/highgui.hpp>  
#include <opencv2/ml.hpp> 
#include<time.h>
#include "tinyxml.h" 

using namespace std;
using namespace cv;
using namespace cv::ml;


int numberOfSample = 20;
int w = 32,h = 32;
int nbins = 9;
int blocksize_w = 16;
int blocksize_h =16;
int b_stride_w = 8;
int b_stride_h = 8;
int cellsize_w = 8;
int cellsize_h = 8;
int varcount = 0;
int hog_dim = nbins*((w -blocksize_w) / b_stride_w + 1)*((h - blocksize_h) / b_stride_h + 1)*(blocksize_w / cellsize_w)*(blocksize_h/ cellsize_h);

vector<vector<double>> svm_sv(45, vector<double>(varcount));
vector<double> svm_rho(45);

clock_t  img_start, img_end;
double imgprocess_time_train,imgprocess_time_test;


Mat gray(Mat src) {
	resize(src, src, Size(w, h));
	src.convertTo(src, CV_32F);
	sqrt(src, src);
	normalize(src, src, 0, 255, NORM_MINMAX, CV_32F);//CV_8UC1
	return src;
}



vector<float> get_cell_hog(Mat img,int nbins) {
	int theta_step = 180 / nbins;
	vector<float> cell_hog(nbins);
	float gradient = 0;
	float theta = 0;
	for (int i = 1; i < img.rows - 1; i++)
	{	
		for (int j = 1; j < img.cols - 1; j++)
		{
			float Gx, Gy;

			Gx = img.at<float>(i, j + 1) - img.at<float>(i, j - 1);
			Gy = img.at<float>(i + 1, j) - img.at<float>(i - 1, j);

			gradient= sqrt(Gx * Gx + Gy * Gy);//梯度模值  

           if (float(atan2(Gy, Gx) * 180 / CV_PI) >= 0) {
				theta= float(atan2(Gy, Gx) * 180 / CV_PI);
			}
			if (float(atan2(Gy, Gx) * 180 / CV_PI) < 0) {
				theta=180.00 + float(atan2(Gy, Gx) * 180 / CV_PI);
			}
			for (int i = 0; i < nbins; i++) {
				if (theta >= theta_step*i && theta < theta_step*(i + 1)) {
					cell_hog[i] += gradient;
				}
			}	
			//梯度方向[-180°，180°]  
		}
	}
	
	return cell_hog;
}


vector<float> get_hog(Mat img,int win_w,int win_h,int block_w,int block_h,int stride_w,int stride_h,int cell_w,int cell_h,int nbins) {
	
	int num_cell_w = block_w / cell_w;
	int num_cell_h = block_h / cell_h;
	vector<float> hog;

	for (int i = 0; i <win_h-stride_h; i = i + stride_h)
	{
		for (int j = 0; j < win_w-stride_w; j = j + stride_w)
		{

	        vector<float> block_vec;
			for (int x = 0; x < num_cell_h; x++)
			{
				for (int y = 0; y < num_cell_w; y++)
				{
					Rect cell(j + y*cell_w, i + x*cell_h , cell_w, cell_h);
					Mat cell_mat = img(cell);	
					vector<float> cell_vec = get_cell_hog(cell_mat, nbins);
					block_vec.insert(block_vec.end(), cell_vec.begin(),cell_vec.end());
				}
			}

			normalize(block_vec, block_vec, NORM_L2);
			hog.insert(hog.end(), block_vec.begin(), block_vec.end());
		}
	}



	return hog;

}


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
	imag = path;   //将读入的彩色图像直接以灰度图像读入  
	result = imag.clone();
	//进行二值化处理，选择30，200.0为阈值  
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








/*vector<float>  hog_path(string path) {
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
}*/


int  train() {

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
		Mat img = gray( image_processing(imagePath[i]));
		descriptors = get_hog(img,w,h,blocksize_w,blocksize_h,b_stride_w,b_stride_h,cellsize_w,cellsize_h,nbins);
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
	return 0;
}




vector<vector<double>> load_svm_sv()
{
	TiXmlDocument mydoc("D:\\datasets\\number\\Detect.xml");//xml文档对象  
	bool loadOk = mydoc.LoadFile();//加载文档  
	if (!loadOk)
	{
		cout << "could not load the test file.Error:" << mydoc.ErrorDesc() << endl;
		exit(1);
	}

	TiXmlElement *RootElement = mydoc.RootElement();  //根元素, Info  
	TiXmlElement *pEle = RootElement;
	TiXmlElement *StuElement = pEle->FirstChildElement();//第一个子元素  

	for (TiXmlElement *sonElement = StuElement->FirstChildElement(); sonElement; sonElement = sonElement->NextSiblingElement())
	{
		if (string(sonElement->Value()) == "var_count")
		{
			varcount = atoi(sonElement->FirstChild()->Value());
		}
	}


	vector<vector<double>> spport_vector(45, vector<double>(varcount));
	int index_sv = 0;
	vector<double> rho(45);


	for (TiXmlElement *sonElement = StuElement->FirstChildElement(); sonElement; sonElement = sonElement->NextSiblingElement())
	{
		if (string(sonElement->Value()) == "support_vectors")
		{
			for (TiXmlElement *node = sonElement->FirstChildElement(); node; node = node->NextSiblingElement())
			{
				TiXmlNode *txt = node->FirstChild();
				char *sv = (char*)(txt->Value());

				int index1 = 0;
				int index2 = 0;
				int num = 0;
				for (int i = 0; i <= strlen(sv); i++)
				{
					if (sv[i] == ' ')
					{
						num++;
						index2 = i;
						char *temp = new char[20];
						strncpy_s(temp, 20, sv + index1, index2 - index1);
						double a;
						stringstream(temp) >> a;
						spport_vector[index_sv][num - 1] = a;
						index1 = index2 + 1;
						if (num == varcount - 1)
						{
							temp = sv + index2 + 1;
							stringstream(temp) >> a;
							spport_vector[index_sv][num] = a;
						}
					}
				}
				index_sv++;
			}
		}
	}
	return spport_vector;
}


vector<double> load_svm_rho()
{
	TiXmlDocument mydoc("D:\\datasets\\number\\Detect.xml");//xml文档对象  
	bool loadOk = mydoc.LoadFile();//加载文档  
	if (!loadOk)
	{
		cout << "could not load the test file.Error:" << mydoc.ErrorDesc() << endl;
		exit(1);
	}

	TiXmlElement *RootElement = mydoc.RootElement();  //根元素, Info  
	TiXmlElement *pEle = RootElement;
	TiXmlElement *StuElement = pEle->FirstChildElement();//第一个子元素  

	for (TiXmlElement *sonElement = StuElement->FirstChildElement(); sonElement; sonElement = sonElement->NextSiblingElement())
	{
		if (string(sonElement->Value()) == "var_count")
		{
			varcount = atoi(sonElement->FirstChild()->Value());
		}
	}
	vector<double> rho(45);

	for (TiXmlElement *sonElement = StuElement->FirstChildElement(); sonElement; sonElement = sonElement->NextSiblingElement())
	{
		
		int num2 = 0;
		if (string(sonElement->Value()) == "decision_functions")
		{
			for (TiXmlElement *node = sonElement->FirstChildElement(); node; node = node->NextSiblingElement())
			{
				num2++;
				for (TiXmlElement *node_df = node->FirstChildElement(); node_df; node_df = node_df->NextSiblingElement())
				{
					if (string(node_df->Value()) == "rho")
					{
						double a;
						stringstream(node_df->FirstChild()->Value()) >> a;
						rho[num2 - 1] = a;
					}
				}
			}
		}
	}
	return rho;
}

int detect_mat(Mat img)
{

	vector<float> descriptor_test(hog_dim);
	descriptor_test = get_hog(img, w, h, blocksize_w, blocksize_h, b_stride_w, b_stride_h, cellsize_w, cellsize_h, nbins);
	
	double *score = new double[10];
	for (int a=0; a < 10; a++) {
		score[a] = 0.00;
	}
	int num = 0;
	for (int i = 0; i < 10; i++)
	{
		for (int j = i+1; j < 10; j++)
		{
			 double sum = 0.00;
			 for (int k = 0; k < varcount; k++)
	         {
		             sum += descriptor_test[k] * svm_sv[num][k];
	         }
	        sum = sum + svm_rho[num];
			if (sum >= 0)
			{
				score[i] += sum;
			}
			if (sum < 0)
			{
				score[j] -= sum;
			}
			num++;
		}
	}
	int max= 0;
	for (int i = 0; i < 10; i++)
	{
		if (score[i] > score[max])max = i;
	}
	return max;
}



int detect(string path) {
	Mat img = imread(path, 0);
	Rect rect1(28, 105, 19, 50);
	Rect rect2(111, 104, 19, 52);
	Rect rect3(193, 104, 19, 54);
	Mat roi1 = gray(image_processing_mat(img(rect1)));
	Mat roi2 = gray(image_processing_mat(img(rect2)));
	Mat roi3 = gray(image_processing_mat(img(rect3)));
	cout <<path<<" "<< detect_mat(roi1) <<" "<< detect_mat(roi2) << " "<<detect_mat(roi3) << endl;
	return 0;
}


int main()
{
    cout << "hog_dim: " << hog_dim << endl;
	clock_t  startTime, endTime;
	double totaltime=0.000;
	startTime = clock();
	train();
	endTime = clock();
	totaltime= ((double)(endTime - startTime)-imgprocess_time_train) / CLOCKS_PER_SEC;
	cout << "train time: " << totaltime<<"s"<< endl;

	clock_t loadstart, loadend;
	double loadTime=0.000;
	loadstart = clock();
	svm_sv = load_svm_sv();
	svm_rho = load_svm_rho();
	loadend = clock();
	loadTime = (double)(loadend- loadstart)  / CLOCKS_PER_SEC;

	string line;
	ifstream testData;
	double alltime=0.000;
	testData.open("D:\\datasets\\number\\test.txt");
	while (getline(testData, line)) {
 startTime = clock();
       detect("D:/datasets/number/all_pic/"+line);
endTime = clock();
alltime += double(endTime - startTime);

	}
	
	totaltime =(alltime-imgprocess_time_test) / CLOCKS_PER_SEC;	
	cout << "load mode time: " <<loadTime<<"s"<< endl;
	cout << "test time: " << totaltime-loadTime<<"s"<< endl;
	cout << "image processiong time" << (imgprocess_time_train+imgprocess_time_test) / CLOCKS_PER_SEC <<"s"<< endl;
	
system("pause");
return 0;
}





