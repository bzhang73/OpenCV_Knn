#include <opencv2/opencv.hpp>
#include <iostream>
#include <io.h>
#include <direct.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

/*ѵ��Knearest*/
Ptr<KNearest> trainKnearest(Mat &data, Mat &lable, int K = 5)
{
	Ptr<TrainData> T_Data = TrainData::create(data, ROW_SAMPLE, lable);
	Ptr<KNearest> Model = KNearest::create();
	Model->setDefaultK(K);
	Model->setIsClassifier(true);
	Model->train(T_Data);
	//Model->save("D:/Cpp/Opencv/knn.xml");//��ѵ��������浽xml�ļ��������´�ʹ��
	return Model;
}

/*����Ԥ��*/
vector<float> predict(Ptr<KNearest> model, Mat &sample)//sample��Ԫ�ر��뾭��һά���л�
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

/*�����ļ���*/
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
			if (!(fileinfo.attrib & _A_SUBDIR))//�ҵ��ļ�  
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

/*�µ�resize�������˺����ڲ��ı�ԭͼ����������£�����offset*/
void myResize(Mat src, Mat &dst, Size size, int offsetx = 0, int offsety = 0)
{
	Mat temp = Mat::zeros(max(src.rows, src.cols), max(src.rows, src.cols), CV_32F);//����һ��ԭͼ��������������α���ɫͼƬ
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
	int roiSizeX = 28;//�趨ÿ��ѵ������x�����С
	int roiSizeY = 28;//�趨ÿ��ѵ������y�����С
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
			temp = img_source.reshape(0, 1);//���л�temp����temp��28x28���󣬱��һά����
			imgData.push_back(temp);//��temp push��trainDate mat��
			lable.push_back(i);//��ÿ��temp��Ӧ��label push��label mat��
		}
	}
	imgData.convertTo(trainData, CV_32F);
	trainLable = lable;

	/*���ò�������*/
	Mat testImg = imread("...path/test.JPG", 1);//�򿪲���ͼƬ
	Mat transMat;
	Mat flipImg;
	transMat = testImg.t();//��ͼƬת��
	flip(transMat, flipImg, 0);//��ͼƬ����x�ᷭת
	resize(flipImg, flipImg, Size(600, 700));//�����趨ͼƬ��С

	Mat grayImg;
	cvtColor(flipImg, grayImg, CV_BGR2GRAY);//ת�����Ҷ�ͼ

	Mat blurImg;
	bilateralFilter(grayImg, blurImg, 3, 50, 50);//˫��ģ����ȥ��һЩ���

	Mat threshImg;
	threshold(blurImg, threshImg, 50, 200, THRESH_BINARY_INV);//������ֵ����ȡ����������

	Mat dilateKernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat dilateImg;
	morphologyEx(threshImg, dilateImg, MORPH_DILATE, dilateKernel);//���Ͳ��������������ݸ�����

	vector<vector<Point>> contours;
	findContours(dilateImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);//�������������
	drawContours(flipImg, contours, -1, Scalar(0, 0, 255), 1);//��������
	int minHeight = 20;//����������С��ֵ��С����ֵ�Ĳ�����Ӿ���
	int minWidth = 0;
	Mat testTemp;
	Mat samples;
	vector<Rect> Rects;
	/*����������������Ӿ���*/
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
	/*ѵ����Ԥ��*/
	Ptr<KNearest> Model = trainKnearest(trainData, trainLable, 5);
	//Ptr<KNearest> Model = Algorithm::load<KNearest>("D:/Cpp/Opencv/knn.xml");//Ҳ����ѡ�������ѵ�������ֱ������
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






