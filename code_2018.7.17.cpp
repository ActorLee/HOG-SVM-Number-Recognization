#include <opencv2/opencv.hpp>  
 #include<fstream>
#include<iostream>
#include <string>
#include <vector>
#include<time.h>
#include "tinyxml.h" 

using namespace std;
using namespace cv;



int numberOfSample = 100;
int w = 32,h = 32;
int nbins = 9;
int blocksize_w = 16;
int blocksize_h =16;
int b_stride_w = 8;
int b_stride_h = 8;
int cellsize_w = 8;
int cellsize_h = 8;
string train_path = "D:\\datasets\\number\\new_train2.txt";
int hog_dim = nbins*((w -blocksize_w) / b_stride_w + 1)*((h - blocksize_h) / b_stride_h + 1)*(blocksize_w / cellsize_w)*(blocksize_h/ cellsize_h);
int varcount = hog_dim;
vector<vector<double>> svm_sv(45, vector<double>(varcount+1));
vector<double> svm_rho(45);

clock_t  img_start, img_end;
double imgprocess_time=0.000;

Mat RGB_func(Mat src) {
	resize(src, src, Size(w, h));
	int i, j;
	int cPointR, cPointG, cPointB, cPoint;//currentPoint;
	for (i = 1; i<src.rows; i++)
		for (j = 1; j<src.cols; j++)
		{
			cPointB = src.at<Vec3b>(i, j)[0];
			cPointG = src.at<Vec3b>(i, j)[1];
			cPointR = src.at<Vec3b>(i, j)[2];
			if (cPointR<220)
			{
				src.at<Vec3b>(i, j)[0] = 0;  //单通道是uchar，没有[0][1][2]
				src.at<Vec3b>(i, j)[1] = 0;
				src.at<Vec3b>(i, j)[2] = 0;
			}

		}
	cvtColor(src, src, CV_BGR2GRAY);

	return src;
}

Mat Gray_func(Mat src)
{

	resize(src, src, Size(w, h));
	int i,j;
	int point;
	for (i = 1; i<src.rows; i++)
		for (j = 1; j < src.cols; j++)
		{
			point = src.at<uchar>(i, j);

			if (point < 0)
			{
				src.at<uchar>(i, j) = 0;  //单通道是uchar，没有[0][1][2]
			}
		}
	return src;
}

Mat norm(Mat src) {	
	
src.convertTo(src, CV_32F);
	//sqrt(src, src);
	//normalize(src, src, 0, 255, NORM_MINMAX);
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



Mat image_processing_mat(Mat path) {
	resize(path, path, Size(w, h));
	img_start = clock();
	vector<vector<int>> a;
	Mat imag, result;
	imag = path; 
	
	result = imag.clone();
	//进行二值化处理，选择30，200.0为阈值  
	threshold(imag, result, 1, 1.0, CV_THRESH_BINARY);
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
	imgprocess_time += double(img_end-img_start);
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


/*Ptr<SVM>  train() {

    Mat featureVector(numberOfSample, hog_dim, CV_32FC1);
    Mat labelsMat(numberOfSample, 1, CV_32SC1);
	vector<string> imagePath;
	vector<int> imageClass;
	string line;
	ifstream trainingData;
	trainingData.open(train_path);
	while (getline(trainingData, line)) {
		int index = 0;
		index = line.find(",", index);
		string a = "D:/datasets/number/new_train/1/";
		a.append(line.substr(0, index));
		string b = line.substr(index + 1);
		int b_int = std::stoi(b);
		imagePath.push_back(a);
		imageClass.push_back(b_int);
	}

	for (int i = 0; i < imagePath.size(); i++) {
		vector<float> descriptors;
		Mat temp = imread(imagePath[i],0);
		Mat img =  norm(Gray_func(temp));
		descriptors = get_hog(img,w,h,blocksize_w,blocksize_h,b_stride_w,b_stride_h,cellsize_w,cellsize_h,nbins);
		for (vector<float>::size_type j = 0; j <= descriptors.size() - 1; ++j)
		{
			featureVector.at<float>(i, j) = descriptors[j];
		}
		labelsMat.at<int>(i, 0) = imageClass[i];
	}

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON));
	svm->train(featureVector, ROW_SAMPLE, labelsMat);
	svm->save("D:\\datasets\\number\\Detect.xml");
	cout << "train over"<< endl;
	return svm;
}*/



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
    vector<vector<double>> spport_vector(45, vector<double>(varcount));
	int index_sv = 0;
	for (TiXmlElement *sonElement = StuElement->FirstChildElement(); sonElement; sonElement = sonElement->NextSiblingElement())
	{
		if (string(sonElement->Value()) == "var_count")
		{
			varcount = atoi(sonElement->FirstChild()->Value());
		}
//*************************************
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
						a = atof(temp);
						spport_vector[index_sv][num - 1] = a;
						index1 = index2 + 1;
						if (num == varcount - 1)
						{
							temp = sv + index2 + 1;
							a = atof(temp);
							spport_vector[index_sv][num] = a;
						}
					}
				}
				index_sv++;
			}
		}
//************************************
		index_sv = 0;
		if (string(sonElement->Value()) == "decision_functions")
		{
			for (TiXmlElement *node = sonElement->FirstChildElement(); node; node = node->NextSiblingElement())
			{
				index_sv++;
				for (TiXmlElement *node_df = node->FirstChildElement(); node_df; node_df = node_df->NextSiblingElement())
				{
					if (string(node_df->Value()) == "rho")
					{
						double a;
						a = atof(node_df->FirstChild()->Value());
						spport_vector[index_sv - 1].push_back(a);
					}
				}
			}
		}

	}

	return spport_vector;
}

int classfier(int i,int j,vector<float> descriptor_test) {
	double sum = 0.00;
	int index = 0;
	int cls = 0;
	for (int x = 0; x < i; x++) {
		index = index + (9 - x);
	}
	index = index + j-i-1;
	for (int k = 0; k < varcount; k++)
	{
		sum += descriptor_test[k] * svm_sv[index][k];
	}
	sum = sum -svm_sv[index][varcount];
	if (sum >= 0) {
		cls = i;
	}
	if (sum < 0) {
		cls = j;
	}
	return cls;
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
	int max = 10;
	/*for (int i = 0; i < 10; i++)
	{
		for (int j = i+1; j < 10; j++)
		{
			 double sum = 0.00;
			 for (int k = 0; k < varcount; k++)
	         {
		             sum += descriptor_test[k] * svm_sv[num][k];
	         }
	        sum = sum + svm_sv[num][varcount];
			if (sum >= 0)
			{
				score[i] += 1;
			}
			if (sum < 0)
			{
				score[j] += 1;
			}
			num++;
		}
	}
	
	for (int i = 0; i < 10; i++)
	{
		if (score[i] > score[max])max = i;
	}*/
	int x = 0, y = 9;
	for (int i = 0; i < 8; i++)
	{
		if (classfier(x, y,descriptor_test) == x)
		{
			max= x;
			y--;
			continue;
		}
		if (classfier(x, y, descriptor_test) == y)
		{
			max = y;
			x++;
			continue;
		}
	}

	return max;
}



int detect(string path) {
	Mat img = imread(path, 0);
	Rect rect1(28, 105, 19, 50);
	Rect rect2(111, 104, 19, 52);
	Rect rect3(193, 104, 19, 54);
	Mat roi1 = norm(image_processing_mat(img(rect1)));
	Mat roi2 = norm(image_processing_mat(img(rect2)));
	Mat roi3 = norm(image_processing_mat(img(rect3)));
	cout <<path<<" "<< detect_mat(roi1) <<" "<< detect_mat(roi2) << " "<<detect_mat(roi3) << endl;
	return 0;
}


int main()
{	
	double img_process_train_time;
	double totaltime=0.000;
    cout << "hog_dim: " << hog_dim << endl;
	clock_t  startTime, endTime;
	startTime = clock();
	//Ptr<SVM>svm=train();
	endTime = clock();
	totaltime= ((double)(endTime - startTime)-imgprocess_time) / CLOCKS_PER_SEC;
	//cout << "train time: " << totaltime<<"s"<< endl;
	img_process_train_time = imgprocess_time;
	imgprocess_time = 0.000;

	clock_t loadstart, loadend;
	double loadTime=0.000;
	loadstart = clock();
	svm_sv = load_svm_sv();
	loadend = clock();
	loadTime = (double)(loadend- loadstart)  / CLOCKS_PER_SEC;

	string line;
	ifstream testData;
	double alltime=0.000;

	testData.open("D:\\datasets\\number\\new_test2.txt");
	while (getline(testData, line)) { 
       startTime = clock();
		Mat img_= imread("D:/datasets/number/new_test/" + line,0);
		Mat test_img =norm(Gray_func( img_));
		Mat featureVecto(1, hog_dim, CV_32FC1);
		vector<float> descriptors = get_hog(test_img, w, h, blocksize_w, blocksize_h, b_stride_w, b_stride_h, cellsize_w, cellsize_h, nbins);
		for (vector<float>::size_type j = 0; j <= hog_dim - 1; ++j)
		{
			featureVecto.at<float>(0, j) = descriptors[j];
		}

		//int y = svm->predict(featureVecto);
	 	int x = detect_mat(test_img);
		  cout << line << " "<<"svm "<<" "<< x<<endl;
		  
endTime = clock();
alltime += double(endTime - startTime);

	}
	

	/*Mat img = norm(imread("D:\\datasets\\number\\new_test\\0_1.jpg", 0));
	Mat featureVecto(1, hog_dim, CV_32FC1);
	vector<float> descriptors = get_hog(img, w, h, blocksize_w, blocksize_h, b_stride_w, b_stride_h, cellsize_w, cellsize_h, nbins);
	for (vector<float>::size_type j = 0; j <= hog_dim - 1; ++j)
	{
		featureVecto.at<float>(0, j) = descriptors[j];
	}
	int y = svm->predict(featureVecto);
	cout << y << endl;
	cout <<detect_mat(img) << endl;*/
	//totaltime =(alltime-imgprocess_time) / CLOCKS_PER_SEC;	
	//cout << "load mode time: " <<loadTime<<"s"<< endl;
	//cout << "test time: " << totaltime<<"s"<< endl;
   // cout << "image processiong time" << (imgprocess_time+img_process_train_time) / CLOCKS_PER_SEC <<"s"<< endl;

	//Mat src = imread("D:\\datasets\\number\\new_train\\1\\2_1.jpg",0);
	//src=Gray_func(src);
	
	//imshow("a", src);

	
	
	
	
	//waitKey(0);
system("pause");
return 0;
}