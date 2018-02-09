#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/core/core.hpp> 
#include <iostream>
#include "cvaux.h" 
#include "cxcore.h"  
#include <fstream>
using namespace std;
using namespace cv;

    //匹配特征点坐标
    float fu1,fv1,fu2,fv2;
	int index1,index2;

    // 初始化SURF检测描述子
	cv::SurfFeatureDetector surfDetector(1000);
	cv::SurfDescriptorExtractor surfExtractor;

    // 关键点及特征描述矩阵声明
	vector<cv::KeyPoint> keyPoints1, keyPoints2;
	cv::Mat descriptorMat1, descriptorMat2;
    //匹配
	cv::FlannBasedMatcher matcher;
	vector< cv::DMatch > matches;
	std::vector<cv::DMatch> viewMatches;

	//相机参数
	Mat cameraK1 = (Mat_<float>(3, 4) << 4334.09568, 0, 959.50000, 0,
	0, 4334.09568, 511.50000,0,
	0, 0, 1, 0 );
	Mat cameraK2= (Mat_<float>(3, 4) << 4489.55770, 0, 801.86552, 0,
	0, 4507.13412, 530.72579, 0,
	0, 0, 1, 0);
	Mat cameraM1 = (Mat_<float>(4, 4) << 1,0,0,-518.97666,
		0,0.999635,-0.0270067,1.20629,
		0,0.0270067,0.999635, 9.14632,
		0,0,0,1);
	Mat cameraM2 = (Mat_<float>(4, 4) << 1,0,0,518.97666,
		0,0.999635,-0.0270067,-1.20629,
		0,0.0270067,0.999635, -9.14632,
		0,0,0,1);
	Mat M1(cameraK1*cameraM1.inv());
	Mat M2(cameraK2*cameraM2.inv());
	float equ_a[12],equ_b[4]; 
	CvMat *Mx = cvCreateMat(3, 1, CV_32FC1);

	ofstream outf; 
	
/********************************************************
* 函数名称： equ_parament
* 输入参数说明： none
* 输出参数说明： none
* 函数作用说明： 计算成像矩阵方程系数
* 
********************************************************/
void equ_parament()
{
	equ_a[0]=fu1*M1.at<float>(2,0)-M1.at<float>(0,0);
	equ_a[1]=fu1*M1.at<float>(2,1)-M1.at<float>(0,1);
	equ_a[2]=fu1*M1.at<float>(2,2)-M1.at<float>(0,2);
	equ_a[3]=fv1*M1.at<float>(2,0)-M1.at<float>(1,0);
	equ_a[4]=fv1*M1.at<float>(2,1)-M1.at<float>(1,1);
	equ_a[5]=fv1*M1.at<float>(2,2)-M1.at<float>(1,2);
	equ_a[6]=fu2*M2.at<float>(2,0)-M2.at<float>(0,0);
	equ_a[7]=fu2*M2.at<float>(2,1)-M2.at<float>(0,1);
	equ_a[8]=fu2*M2.at<float>(2,2)-M2.at<float>(0,2);
	equ_a[9]=fv2*M2.at<float>(2,0)-M2.at<float>(1,0);
	equ_a[10]=fv2*M2.at<float>(2,1)-M2.at<float>(1,1);
	equ_a[11]=fv2*M2.at<float>(2,2)-M2.at<float>(1,2);

	equ_b[0]=M1.at<float>(0,3)-fu1*M1.at<float>(2,3);
	equ_b[1]=M1.at<float>(1,3)-fv1*M1.at<float>(2,3);
	equ_b[2]=M2.at<float>(0,3)-fu2*M2.at<float>(2,3);
	equ_b[3]=M2.at<float>(1,3)-fv2*M2.at<float>(2,3);
	
}


/********************************************************
* 函数名称： cacSURFFeatureAndCompare
* 输入参数说明： srcImage1 左图 srcImage2 右图
* 输出参数说明： result 匹配度
* 函数作用说明： SURF图提取特征点，像匹配
* 
********************************************************/
// 计算图像的特征及匹配
float cacSURFFeatureAndCompare(cv::Mat srcImage1,
	cv::Mat srcImage2, float paraHessian)
{ 
	CV_Assert(srcImage1.data != NULL && srcImage2.data != NULL);
	// 转换为灰度
	cv::Mat grayMat1, grayMat2;
	cv::cvtColor(srcImage1, grayMat1, CV_RGB2GRAY);
	cv::cvtColor(srcImage2, grayMat2, CV_RGB2GRAY);

	// 计算surf特征关键点
	surfDetector.detect( grayMat1, keyPoints1 );
	surfDetector.detect( grayMat2, keyPoints2 );
	cout<<"图像1特征点个数:"<<keyPoints1.size()<<endl;  
    cout<<"图像2特征点个数:"<<keyPoints2.size()<<endl;  

	// 计算surf特征描述矩阵
	surfExtractor.compute(grayMat1, keyPoints1, descriptorMat1);
	surfExtractor.compute(grayMat2, keyPoints2, descriptorMat2);
	 cout<<"图像1特征描述矩阵大小："<<descriptorMat1.size()  
        <<"，特征向量个数："<<descriptorMat1.rows<<"，维数："<<descriptorMat1.cols<<endl;  
    cout<<"图像2特征描述矩阵大小："<<descriptorMat2.size()  
        <<"，特征向量个数："<<descriptorMat2.rows<<"，维数："<<descriptorMat2.cols<<endl;  

	Mat outimg1;
	drawKeypoints(grayMat1, keyPoints1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	namedWindow("img1_特征点", CV_WINDOW_NORMAL);
    imshow("img1_特征点",outimg1);
	Mat outimg2;
	drawKeypoints(grayMat1,keyPoints2, outimg2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	namedWindow("img2_特征点", CV_WINDOW_NORMAL);
    imshow("img2_特征点",outimg2);
	float result = 0;

	// 特征点匹配
	if (keyPoints1.size() > 0 && keyPoints2.size() > 0)
	{	
		// 计算特征匹配点
		matcher.match( descriptorMat1, descriptorMat2, matches);
		cout<<"Match个数："<<matches.size()<<endl;  
		// 最优匹配判断
		float minDist = 100;
		float maxDist = 0;
		for (int i = 0; i < matches.size(); i++)
		{ 
			if(matches[i].distance < minDist) 
				minDist = matches[i].distance;
			if (matches[i].distance > maxDist)
			{
				maxDist = matches[i].distance;
			}
		}
		// 计算距离特征符合要求的特征点
		int num = 0;
		std::cout << "minDist: "<< minDist << std::endl;
		std::cout << "maxDist: "<< maxDist << std::endl;

		for( int i = 0; i < matches.size(); i++ )
		{ 
			// 特征点匹配距离判断
			if(matches[i].distance <= minDist+(maxDist-minDist)/3)//+(maxDist-minDist)/2
			{
				result += matches[i].distance;
				viewMatches.push_back(matches[i]);
				num++;
			}
		}
		cout<<"goodMatch个数："<<viewMatches.size()<<endl;  
		// 匹配度计算
		result /= num;

		// 绘制特征点匹配结果
		cv::Mat matchMat;
		cv::drawMatches(srcImage1, keyPoints1, 
			srcImage2, keyPoints2, matches, matchMat); 
		//cv::namedWindow("matchMat", CV_WINDOW_NORMAL);
		//cv::imshow("matchMat", matchMat); 

		Mat img_goodmatch;//筛选后的匹配点图
        drawMatches(srcImage1, keyPoints1, 
			srcImage2, keyPoints2, viewMatches, img_goodmatch);
	    namedWindow("筛选后的匹配点对", CV_WINDOW_NORMAL);
        imshow("筛选后的匹配点对", img_goodmatch);
		//cv::waitKey(0);
	}
	return result;
}
int main ()
{
	// 读取源图像及待匹配图像
	cv::Mat srcImage1 = 
		cv::imread("carl.bmp", 1); 
	if (srcImage1.empty()) 
		return -1;
	cv::Mat srcImage2 = 
		cv::imread("carr.bmp", 1); 
	if (srcImage2.empty()) 
		return -1;


	float matchRate= cacSURFFeatureAndCompare(srcImage1, srcImage2,1000);
	std::cout <<"matchRate: " << matchRate << std::endl;

	outf.open("depth.txt",std::ios::out | std::ios::app);
	if (!outf.is_open())
        return 0;
	outf << "匹配点像素坐标（单位：像素）"  << endl;
	outf <<  "物体世界坐标XYZ(单位：米):" << endl;

	for (int i = 0; i < viewMatches.size(); i++)
	{
		cout << "\n匹配点序号： " << i << endl;
		outf << "\n匹配点序号： " << i << endl;

		//提取左右途中匹配点像素坐标
		index1=viewMatches.at(i).queryIdx;
		index2=viewMatches.at(i).trainIdx;
		fu1=keyPoints1.at(index1).pt.x;
		fv1=keyPoints1.at(index1).pt.y;
		fu2=keyPoints2.at(index2).pt.x;
		fv2=keyPoints2.at(index2).pt.y;
		//cout << "fu1 = \n" << fu1 << endl; 
		//cout << "fv1 = \n" << fv1 << endl;
		//cout << "fu2 = \n" << fu2 << endl; 
		//cout << "fv2 = \n" << fv2 << endl;

		outf << "左图匹配点像素坐标："  << endl;
		outf << "fu1 = " << fu1 << endl; 
		outf << "fv1 = " << fv1 << endl;
		outf << "右图匹配点像素坐标："  << endl;
		outf<< "fu2 = " << fu2 << endl; 
		outf <<"fv2 = " << fv2 << endl;

		//计算矩阵方程参数
		equ_parament();
		CvMat Ma = cvMat(4, 3, CV_32FC1, equ_a);
		CvMat Mb = cvMat(4, 1, CV_32FC1, equ_b);

		cout << endl << "ma:" << endl;
		//outf << endl << "ma:" << endl;
	    for (int ai = 0; ai < 4; ai++)
	    {
			const float *p = (const float*)(Ma.data.ptr + ai*Ma.step);
		cout << *p << "    " << *(p + 1) << "		" << *(p + 2) << endl;
		//outf<< *p << "    " << *(p + 1) << "		" << *(p + 2) << endl;
	    }

		cout << endl << "mb:" << endl;
		//outf << endl << "mb:" << endl;
	    for (int bi = 0; bi < 4; bi++)
	    {
			const float *p = (const float*)(Mb.data.ptr + bi*Mb.step);
		cout << *p << "    " << endl;
		//outf << *p << "    " << endl;
	    }

		//最小二乘法计算深度值
		cvSolve(&Ma, &Mb, Mx, CV_SVD );

		cout << endl << "mx:" << endl;
		outf <<  "物体世界坐标XYZ:" << endl;
		for(int xi=0; xi<3; xi++)    
       {          
		   const float *p = (const float*)(Mx->data.ptr + xi*Mx->step);
		 cout << *p /1000<< "    " << endl; 
		 outf << *p/1000 << "    " << endl;
       }     

	}
	outf.close();
  
	cv::waitKey();
	system("pause");
	return 0;
}

