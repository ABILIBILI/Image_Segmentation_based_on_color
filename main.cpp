#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <typeinfo>
#include <svm.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

using namespace cv;
using namespace std;
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define FILE_NUM 8
#define POSITIVE_NUM 4
#define TRAIN_DATA_CACHE   (Train_Image_data.size() * 4 * sizeof(double)+Train_Image_data.size() * 4 * sizeof(int))
#define TRAIN_FLAG 0

void read_train_image(void);
void copy_data_to_svm(void);
Mat read_test_image(const struct svm_model *svmmodel);

struct point_HSV
{
	double H;
	double S;
	double V;
};


vector <Mat> srcImage,HSVImage;
Mat Image_HSV_Signal_Channal[FILE_NUM][3];
Mat test_Image, test_HSVImage[3],predic_img;
vector <point_HSV> Train_Image_data;
vector <double> lable_y;


svm_model HSV_model;
svm_node* x_space;
svm_node** train_data_x;

int main()
{

	const char* model_path = "F:\\»º´æ\\TEMP\\ConsoleApplication3\\model";


	read_train_image();
	cout<<"read_img_finish\n"<<endl;
	train_data_x = Malloc(svm_node *, TRAIN_DATA_CACHE);
	x_space = Malloc(struct svm_node, TRAIN_DATA_CACHE);

	copy_data_to_svm();
	cout << "copy_data_to_svm_finish" << endl;
	const char* error_msg;
	
	double temp[2] = { 1, 1 };

	const svm_problem svmprob = { lable_y.size(), &lable_y[0], { train_data_x } };
	const svm_parameter svmpar = { C_SVC, RBF, 0, 0.5, 0, 300, 0.05, 50, 0, NULL, temp, 0.1, 1, 0 };

	error_msg = svm_check_parameter(&svmprob, &svmpar);
	if (error_msg){
		cout << *error_msg << endl;
		return -1;
	}
	cout << "checek_finish" << endl;
#if TRAIN_FLAG
	
	 svm_model *svmmodel = svm_train(&svmprob, &svmpar);	cout << "train_finish" << endl;
	 svm_save_model( model_path , svmmodel);
	
	 return 0;
#else	
	 svm_model* svmmodel = svm_load_model(model_path);	cout << "load_finish" << endl;
#endif
	 free(train_data_x);
	free(x_space);

	predic_img = read_test_image(svmmodel);
	cout << "predic_finish" << endl;

	namedWindow("HSV", CV_WINDOW_NORMAL);
	imshow("HSV", predic_img);
	
	waitKey(0);

	return 0;
}

void read_train_image(void){

	char ImgfileName[200];
	Mat tempHSV;
	double temp_y = 0;
	for (int i = 1; i <= FILE_NUM; i++)
	{


		sprintf_s(ImgfileName,"F:\\»º´æ\\TEMP\\ConsoleApplication3\\train_data\\traindata%d.png", i);
		srcImage.push_back(imread(ImgfileName));
		cvtColor(srcImage[i-1], tempHSV, COLOR_RGB2HSV);
		HSVImage.push_back(tempHSV);
		split(tempHSV, Image_HSV_Signal_Channal[i]);

		if (i <= POSITIVE_NUM)
			temp_y = 1;
		else         
			temp_y = -1;


		for (size_t m = 0; m < Image_HSV_Signal_Channal[i][0].cols; m++)
		for (size_t n = 0; n < Image_HSV_Signal_Channal[i][0].rows; n++)
			{
				struct point_HSV temp_point_value;

				temp_point_value.H = Image_HSV_Signal_Channal[i][0].data[m*Image_HSV_Signal_Channal[i][0].rows + n];//+ m*Image_HSV_Signal_Channal[i][0].rows + n)
				temp_point_value.S = Image_HSV_Signal_Channal[i][1].data[m*Image_HSV_Signal_Channal[i][0].rows + n];
				temp_point_value.V = Image_HSV_Signal_Channal[i][2].data[m*Image_HSV_Signal_Channal[i][0].rows + n];

				Train_Image_data.push_back(temp_point_value);

				lable_y.push_back(temp_y);

			}

	}



}
void copy_data_to_svm(void){


	for (size_t i = 0, m = 0; i < TRAIN_DATA_CACHE&& m < Train_Image_data.size(); i += 4, m++)
	{
		x_space[i].index = (int)(1);
		x_space[i].value = (double)(Train_Image_data[m].H);
		train_data_x[i] = &x_space[i];

		x_space[i + 1].index = (int)(2);
		x_space[i + 1].value = (double)(Train_Image_data[m].S);
		train_data_x[i+1] = &x_space[i+1];

		x_space[i + 2].index = (int)(3);
		x_space[i + 2].value = (double)(Train_Image_data[m].V);
		train_data_x[i+2] = &x_space[i+2];

		x_space[i+3].index = int(-1);
		x_space[i+3].value = double(-1);
		train_data_x[i+3] = &x_space[i+3];
	}
}

Mat read_test_image(const struct svm_model *svmmodel){

	Mat tempHSV;
	unsigned char bin_temp[3] = { 0, 255,255 };
	


	test_Image=imread("F:\\»º´æ\\TEMP\\ConsoleApplication3\\00000.png");
	cvtColor(test_Image, tempHSV, COLOR_RGB2HSV);
	split(tempHSV, test_HSVImage);
	
	const int col = test_HSVImage[0].cols;
	const int row = test_HSVImage[0].rows;

	const int num = col*row;

	Mat predict_img_temp(row, col, CV_8UC1);



	int* temp = Malloc(int, num);


	for (size_t m = 0; m < test_HSVImage[0].rows; m++)
		for (size_t n = 0; n < test_HSVImage[0].cols; n++)
		{
			svm_node *testpoint = new svm_node[4];

			testpoint[0].index = 1;
			testpoint[0].value = test_HSVImage[0].data[m * col + n];
			testpoint[1].index = 2;
			testpoint[1].value = test_HSVImage[1].data[m * col + n];
			testpoint[2].index = 3;
			testpoint[2].value = test_HSVImage[2].data[m * col + n];
			testpoint[3].index = -1;


			predict_img_temp.data[m * col + n] = bin_temp[(int)(svm_predict(svmmodel,testpoint) + 1)];
			temp[m*col + n] = svm_predict(svmmodel, testpoint);

			
		}


	return predict_img_temp;
}