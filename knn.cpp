#include <opencv2/opencv.hpp>
#include <iostream>
#include <io.h>
#include <direct.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

/*训练Knearest*/
Ptr<KNearest> trainKnearest(Mat &data, Mat &lable, int K = 5)
{
	Ptr<TrainData> T_Data = TrainData::create(data, ROW_SAMPLE, lable);
	Ptr<KNearest> Model = KNearest::create();
	Model->setDefaultK(K);
	Model->setIsClassifier(true);
	Model->train(T_Data);
	//Model->save("D:/Cpp/Opencv/knn.xml");//将训练结果保存到xml文件，方便下次使用
	return Model;
}

/*分类预测*/
vector<float> predict(Ptr<KNearest> model, Mat &sample)//sample的元素必须经过一维序列化
{
	vector<float> result;
	int sampleNum = sample.rows;
	for (int i = 0; i < sampleNum; ++i)
	{
		float prediction = model->predict(sample.row(i));
		result.push_back(prediction);
	}
	return result;
}

/*遍历文件夹*/
void getFiles(string mainDir, vector<string> &files)
{
	files.clear();
	const char *dir = mainDir.c_str();
	_chdir(dir);
	intptr_t hFile;
	_finddata_t fileinfo;

	if ((hFile = _findfirst("*.*", &fileinfo)) != -1)
	{
		do
		{
			if (!(fileinfo.attrib & _A_SUBDIR))//找到文件  
			{
				char filename[_MAX_PATH];
				strcpy_s(filename, dir);
				strcat_s(filename, "\\");
				strcat_s(filename, fileinfo.name);
				string temfilename = filename;
				files.push_back(temfilename);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

/*新的resize函数，此函数在不改变原图比例的情况下，增加offset*/
void myResize(Mat src, Mat &dst, Size size, int offsetx = 0, int offsety = 0)
{
	Mat temp = Mat::zeros(max(src.rows, src.cols), max(src.rows, src.cols), CV_32F);//创建一个原图的最大正解正方形背景色图片
	Mat backGround = Mat::zeros(size, CV_32F);
	src.copyTo(temp(Range((temp.rows - src.rows) / 2, (temp.rows - src.rows) / 2 + src.rows),
		Range((temp.cols - src.cols) / 2, (temp.cols - src.cols) / 2 + src.cols)));
	resize(temp, temp, Size(size.width - 2 * offsetx, size.height - 2 * offsety));
	temp.copyTo(backGround(Range(offsetx, size.width - offsetx), Range(offsety, size.height - offsety)));
	dst = backGround;
}

int main()
{
	vector<string> fileName;
	string filePath;
	int roiSizeX = 28;//设定每个训练样本x方向大小
	int roiSizeY = 28;//设定每个训练样本y方向大小
	Mat temp;
	Mat imgData;
	Mat trainData;
	Mat lable;
	Mat trainLable;

	for (int i = 0; i < 10; ++i)
	{
		string mainPath = "...path/mnist/" + to_string(i);
		getFiles(mainPath, fileName);
		for (string name : fileName)
		{
			filePath = name;
			Mat img_source = imread(filePath, 0);
			temp = img_source.reshape(0, 1);//序列化temp，将temp由28x28矩阵，变成一维矩阵
			imgData.push_back(temp);//将temp push到trainDate mat中
			lable.push_back(i);//将每个temp对应的label push到label mat里
		}
	}
	imgData.convertTo(trainData, CV_32F);
	trainLable = lable;

	/*设置测试样本*/
	Mat testImg = imread("...path/test.JPG", 1);//打开测试图片
	Mat transMat;
	Mat flipImg;
	transMat = testImg.t();//将图片转置
	flip(transMat, flipImg, 0);//将图片沿着x轴翻转
	resize(flipImg, flipImg, Size(600, 700));//重新设定图片大小

	Mat grayImg;
	cvtColor(flipImg, grayImg, CV_BGR2GRAY);//转换到灰度图

	Mat blurImg;
	bilateralFilter(grayImg, blurImg, 3, 50, 50);//双边模糊，去掉一些噪点

	Mat threshImg;
	threshold(blurImg, threshImg, 50, 200, THRESH_BINARY_INV);//设置阈值，提取出数字内容

	Mat dilateKernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat dilateImg;
	morphologyEx(threshImg, dilateImg, MORPH_DILATE, dilateKernel);//膨胀操作，让数字内容更明显

	vector<vector<Point>> contours;
	findContours(dilateImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);//查找最外层轮廓
	drawContours(flipImg, contours, -1, Scalar(0, 0, 255), 1);//画出轮廓
	int minHeight = 20;//设置轮廓大小阈值，小于阈值的不画外接矩形
	int minWidth = 0;
	Mat testTemp;
	Mat samples;
	vector<Rect> Rects;
	/*遍历轮廓，画出外接矩形*/
	for (auto cont : contours)
	{
		Rect rect = boundingRect(cont);
		if (rect.height>minHeight&&rect.width>minWidth)
		{
			rectangle(flipImg, rect, Scalar(0, 255, 0));
			Rects.push_back(rect);
			dilateImg(rect).copyTo(testTemp);
			myResize(testTemp, testTemp, Size(roiSizeX, roiSizeY), 2, 2);
			testTemp = testTemp.reshape(0, 1);
			testTemp.convertTo(testTemp, CV_32F);
			samples.push_back(testTemp);
		}
	}
	/*训练和预测*/
	Ptr<KNearest> Model = trainKnearest(trainData, trainLable, 5);
	//Ptr<KNearest> Model = Algorithm::load<KNearest>("D:/Cpp/Opencv/knn.xml");//也可以选择从现有训练结果中直接载入
	vector<float> result = predict(Model, samples);

	for (float i : result)
	{
		cout << i << endl;
	}
	for (int i = 0; i < result.size(); ++i)
	{
		putText(flipImg, to_string((int)result[i]), Point(Rects[i].x, Rects[i].y), 1, 1, Scalar(255, 0, 0));
		//imshow("result", flipImg);
		//Mat dst;
		//dst=samples.row(i).reshape(0, roiSizeX);
		//imshow("dst", dst);
		//waitKey(0);
	}
	imshow("result", flipImg);
	waitKey(0);
	return 0;
}






