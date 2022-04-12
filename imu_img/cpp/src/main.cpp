#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <unistd.h>
// #include "stdio.h"

using namespace std;
using namespace cv;

const Size IMG_SIZE = Size(1920, 1080);
const int IMG_NUM = 20;

// void calibrateCam(Mat &cameraMatrix, Mat &distCoeffs);

bool isExists(const std::string &name)
{
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}


//void getFiles( string path, vector<string>& files )
//{
//	//文件句柄
//	long   hFile   =   0;
//	//文件信息
//	struct _finddata_t fileinfo;
//	string p;
//	if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1)
//	{
//		do
//		{
//			//如果是目录,迭代之
//			//如果不是,加入列表
//			if((fileinfo.attrib &  _A_SUBDIR))
//			{
//				// if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
//				// 	getFiles( p.assign(path).append("\\").append(fileinfo.name), files );
//			}
//			else
//			{
//				files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
//			}
//		}while(_findnext(hFile, &fileinfo)  == 0);
//		_findclose(hFile);
//	}
//}

//void deleteInvalidImages(string imagesDir)
//{
//	Size boardSize = Size(9, 7);
//	vector<string> files;
//	getFiles(imagesDir, files);
//	for (int i = 0; i < files.size(); i++)
//	{
//		string imgPath;
//		imgPath = files[i];
//		if (!isExists(imgPath))
//			continue;
//
//		std::cout << "image path: " << imgPath << endl;
//		Mat img = imread(imgPath);
//
//		vector<Point2f> imageCornerBuff;
//		if (0 == findChessboardCorners(img, boardSize, imageCornerBuff))
//		{
//			cout << "can not find chessboard corners!\n"; // 找不到角点
//			remove(imgPath.c_str());
//		}
//	}
//}

//void calibrateCam(Mat &cameraMatrix, Mat &distCoeffs, string imagesDir)
//{
//	Size imageSize = IMG_SIZE;
//	Size boardSize = Size(9, 7);
//	vector<vector<Point2f>> imagesCorners;
//	vector<string> files;
//	getFiles(imagesDir, files);
//	for (int i = 0; i < files.size(); i++)
//	{
//		string imgPath;
//		imgPath = files[i];
//		if (!isExists(imgPath))
//		{
//			cout << "File doesn't exist!" << endl;
//			exit(-1);
//		}
//		cout << "image path: " << imgPath << endl;
//		Mat img = imread(imgPath);
//
//		vector<Point2f> imageCornerBuff;
//		if (0 == findChessboardCorners(img, boardSize, imageCornerBuff))
//		{
//			cout << "can not find chessboard corners!\n";
//			exit(1);
//		}
//		else
//		{
//			Mat gray;
//			cvtColor(img, gray, CV_RGB2GRAY);
//			/* 亚像素精确化 */
//			// image_points_buf 初始的角点坐标向量，同时作为亚像素坐标位置的输出
//			// Size(5,5) 搜索窗口大小
//			// （-1，-1）表示没有死区
//			// TermCriteria 角点的迭代过程的终止条件, 可以为迭代次数和角点精度两者的组合
//			cornerSubPix(gray, imageCornerBuff, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
//
//			imagesCorners.push_back(imageCornerBuff); // 保存亚像素角点
//
//			/* 在图像上显示角点位置 */
//			drawChessboardCorners(img, boardSize, imageCornerBuff, false); // 用于在图片中标记角点
//			imshow("Camera Calibration", img);							   // 显示图片
//			waitKey(100);												   //暂停
//		}
//	}
//
//	int CornerNum = boardSize.width * boardSize.height; // 每张图片上总的角点数
//	Size squareSize = Size(10, 10);						/* 实际测量得到的标定板上每个棋盘格的大小 */
//	vector<vector<Point3f>> objectPoints;				/* 保存标定板上角点的三维坐标 */
//
//	//Mat cameraMatrix; /* 摄像机内参数矩阵 */
//	vector<int> point_counts; // 每幅图像中角点的数量
//	//Mat distCoeffs;       /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
//	vector<Mat> tvecsMat; /* 每幅图像的旋转向量 */
//	vector<Mat> rvecsMat; /* 每幅图像的平移向量 */
//
//	/* 初始化标定板上角点的三维坐标 */
//	for (int t = 0; t < files.size(); t++)
//	{
//		vector<Point3f> tempPointSet;
//		for (int i = 0; i < boardSize.height; i++)
//		{
//			for (int j = 0; j < boardSize.width; j++)
//			{
//				Point3f realPoint;
//
//				/* 假设标定板放在世界坐标系中z=0的平面上 */
//				realPoint.x = i * squareSize.width;
//				realPoint.y = j * squareSize.height;
//				realPoint.z = 0;
//				tempPointSet.push_back(realPoint);
//			}
//		}
//		objectPoints.push_back(tempPointSet);
//	}
//
//	/* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */
//	for (int i = 0; i < files.size(); i++)
//	{
//		point_counts.push_back(boardSize.width * boardSize.height);
//	}
//
//	/* 开始标定 */
//	// object_points 世界坐标系中的角点的三维坐标
//	// image_points_seq 每一个内角点对应的图像坐标点
//	// image_size 图像的像素尺寸大小
//	// cameraMatrix 输出，内参矩阵
//	// distCoeffs 输出，畸变系数
//	// rvecsMat 输出，旋转向量
//	// tvecsMat 输出，位移向量
//	// 0 标定时所采用的算法
//	calibrateCamera(objectPoints, imagesCorners, imageSize, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
//}

int main(int argc, char** argv)
{
	string ROOTDIR;
	string imgdir;
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	cameraMatrix.at<float>(0,0) = 1001.5891335;
	cameraMatrix.at<float>(0,2) = 953.6128327;
	cameraMatrix.at<float>(1,1) = 1000.9244526;
	cameraMatrix.at<float>(1,2) = 582.04816056;
	cameraMatrix.at<float>(2,2) = 1;
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));
//	 distCoeffs.at<float>(0,0) = 3.2083739041580001e-01;
//	 distCoeffs.at<float>(0,1) = 2.2269550643173597e-01;
//	 distCoeffs.at<float>(0,2) = 8.8895447057740762e-01;
//	 distCoeffs.at<float>(0,3) = -2.8404775013002994e+00;
	 // distCoeffs.at<float>(0,4) = 4.867095044851689;
	ROOTDIR = "/mnt/d/human_data/0724/calib/";
    cout << "calib camera imu RT..." << endl;
    string result = ROOTDIR + "pair.txt";
    cout << "read 2d-3d points pairs..." << endl;
    ifstream pointPairFile(result);
    if (!pointPairFile.is_open())
    {
        cout << "open 2D-3D files failed" << endl;
        return -1;
    }

    vector<Point3f> obejctPoints;
    vector<Point2f> imagePoints;
    // todo
    for (int i = 0; i < 6; i++)
    {
        Point2f temp1;
        pointPairFile >> temp1.x >> temp1.y;
        Point3f temp2;
        pointPairFile >> temp2.x >> temp2.y >> temp2.z;
        obejctPoints.push_back(temp2);
        imagePoints.push_back(temp1);
    }
    cout << "2D Points:" << endl;
    cout << imagePoints << endl;
    cout << "3D Points:" << endl;
    cout << obejctPoints << endl
        << endl;
    pointPairFile.close();

    cout << "cal extra parameters..." << endl;
    Mat rvec, tvec;
    solvePnP(obejctPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    // solveP3P();
    // solvePoly();
    // solve();
    double rotD[9] = {0};
    Mat rotM(3, 3, CV_64FC1, rotD);
    Rodrigues(rvec, rotM);
    cout << "Rotation matrix:\n" << rotM << endl;
    cout << "Trans matrix:\n" << tvec << endl;
    ofstream eParamFile(ROOTDIR + "CamExtrinsic.txt");
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            eParamFile << rotM.ptr<double>(i)[j] << " ";
        }
        eParamFile << tvec.ptr<double>(i)[0] << endl;
    }
    eParamFile << "0 0 0 1";
    eParamFile.close();

    cout << "direction vector:" << endl;
    ofstream camDirFile(ROOTDIR + "CamDir.txt");
    // Mat xTemp = Mat(3, 3, CV_32FC1, Scalar::all(0));
    // xTemp << 1, 0 , 0;
    double xTemp[3] = {1, 0, 0};
    Mat xVec = rotM.inv() * Mat(3, 1, CV_64FC1, xTemp);
    camDirFile << xVec.t() << endl;
    cout << xVec.t() << endl;
    double yTemp[3] = {0, -1, 0};
    Mat yVec = rotM.inv() * Mat(3, 1, CV_64FC1, yTemp);
    camDirFile << yVec.t() << endl;
    cout << yVec.t() << endl;
    double zTemp[3] = {0, 0, -1};
    Mat zVec = rotM.inv() * Mat(3, 1, CV_64FC1, zTemp);
    camDirFile << zVec.t();
    cout << zVec.t() << endl;
    camDirFile.close();


    cout << "position of camera:" << endl;
    Mat camPos = rotM.inv() * (0 - tvec);
    cout << camPos << endl;
    ofstream camPosFile(ROOTDIR + "CamPos.txt");
    camPosFile << camPos;
    camPosFile.close();
}
    //cout << "偏航角：" << endl;
    //double theta_x = atan2(rotM.at<double>(2, 1), rotM.at<double>(2, 2));
    //double theta_y = atan2(-rotM.at<double>(2, 0),
    //	sqrt(rotM.at<double>(2, 1)*rotM.at<double>(2, 1) + rotM.at<double>(2, 2)*rotM.at<double>(2, 2)));
    //double theta_z = atan2(rotM.at<double>(1, 0), rotM.at<double>(0, 0));
    //cout << theta_x << " " << endl
    //	<< theta_y << " " << endl
    //	<< theta_z << endl;

//	for (int i = 1; i < argc; i++)
//	{
//		const char *currArg = argv[i];
//		if (strcmp(currArg, "--help") ==0 || strcmp(currArg, "-h") == 0)
//		{
//			printf("-d\t找出指定文件夹内有效的图片，无效的图片将被删除\n");
//			printf("-i\t标定内参步骤。指定根目录，棋盘格图片请保存在images文件夹下，结果将保存到result文件夹下\n");
//			printf("-e\t标定外参步骤。指定根目录，棋盘格图片请保存在images文件夹下，结果将保存到result文件夹下\n");
//			return 0;
//		}
//		if (strcmp(currArg, "-d") == 0 || strcmp(currArg, "-D") == 0)
//		{
//			cout << "Deleting invalid images..." << endl;
//			i++;
//			if (i >= argc)
//			{
//				printf("指定根目录路径错误！");
//				return -1;
//			}
//			imgdir = argv[i];
//			cout << "imgdir is:" << imgdir << endl;
//			deleteInvalidImages(imgdir);
//			return 0;
//		}
//		else if (strcmp(currArg, "-i") == 0 || strcmp(currArg, "-I") == 0)
//		{
//			cout << "标定相机内参..." << endl;
//			i++;
//			if (i >= argc)
//			{
//				printf("指定根目录路径错误！");
//				return -1;
//			}
//			ROOTDIR = argv[i];
//			cout << "ROOTDIR IS:" << ROOTDIR << endl;
//
//			calibrateCam(cameraMatrix, distCoeffs, ROOTDIR+"\\images");
//			string result = ROOTDIR + "\\result\\";
//			ofstream iParamFile(result + "intrinsic_matrix.txt");
//			ofstream distFile(result + "dist_coeff.txt");
//			iParamFile << cameraMatrix;
//			distFile << distCoeffs;
//			iParamFile.close();
//			distFile.close();
//			// distCoeffs = (Mat_<float>(1, 5) << -0.4782848119910399, 0.2015237767054037, 0.00201207874800318, -0.003078179793033205, 0.4735001159851723);
//			cout << "内参矩阵：" << endl;
//			cout << cameraMatrix << endl;
//			cout << "畸变系数：" << endl;
//			cout << distCoeffs << endl
//				<< endl;
//			return 0;
//		}
//		else if (strcmp(currArg, "-e") == 0 || strcmp(currArg, "-E") == 0)
//		{
//			/*
//			* 开始标定外参
//			*/
//			cout << "标定相机外参..." << endl;
//			i++;
//			if (i >= argc)
//			{
//				printf("指定根目录路径错误！");
//				return -1;
//			}
//			ROOTDIR = argv[i];
//			cout << "ROOTDIR IS:" << ROOTDIR << endl;
//			calibrateCam(cameraMatrix, distCoeffs, ROOTDIR+"\\images");
//
//			string result = ROOTDIR + "\\result\\";
//			cout << "读取2D-3D点对..." << endl;
//			ifstream pointPairFile(result + "pair.txt");
//			if (!pointPairFile.is_open())
//			{
//				cout << "打开2D-3D点对文件失败" << endl;
//				return -1;
//			}
//
//			vector<Point3f> obejctPoints;
//			vector<Point2f> imagePoints;
//
//			for (int i = 0; i < 7; i++)
//			{
//				Point2f temp1;
//				pointPairFile >> temp1.x >> temp1.y;
//				Point3f temp2;
//				pointPairFile >> temp2.x >> temp2.y >> temp2.z;
//				obejctPoints.push_back(temp2);
//				imagePoints.push_back(temp1);
//			}
//			cout << "2D Points：" << endl;
//			cout << imagePoints << endl;
//			cout << "3D Points：" << endl;
//			cout << obejctPoints << endl
//				<< endl;
//			pointPairFile.close();
//
//			cout << "求解外参..." << endl;
//			Mat rvec, tvec;
//			solvePnP(obejctPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);
//			// solveP3P();
//			// solvePoly();
//			// solve();
//			double rotD[9] = {0};
//			Mat rotM(3, 3, CV_64FC1, rotD);
//			Rodrigues(rvec, rotM);
//			cout << "旋转矩阵：\n" << rotM << endl;
//			cout << "平移矩阵：\n" << tvec << endl;
//			ofstream eParamFile(result + "CamExtrinsic.txt");
//			for (int i = 0; i < 3; i++)
//			{
//				for (int j = 0; j < 3; j++)
//				{
//					eParamFile << rotM.ptr<double>(i)[j] << " ";
//				}
//				eParamFile << tvec.ptr<double>(i)[0] << endl;
//			}
//			eParamFile << "0 0 0 1";
//			eParamFile.close();
//
//			cout << "方向向量：" << endl;
//			ofstream camDirFile(result + "CamDir.txt");
//			// Mat xTemp = Mat(3, 3, CV_32FC1, Scalar::all(0));
//			// xTemp << 1, 0 , 0;
//			double xTemp[3] = {1, 0, 0};
//			Mat xVec = rotM.inv() * Mat(3, 1, CV_64FC1, xTemp);
//			camDirFile << xVec.t() << endl;
//			cout << xVec.t() << endl;
//			double yTemp[3] = {0, -1, 0};
//			Mat yVec = rotM.inv() * Mat(3, 1, CV_64FC1, yTemp);
//			camDirFile << yVec.t() << endl;
//			cout << yVec.t() << endl;
//			double zTemp[3] = {0, 0, -1};
//			Mat zVec = rotM.inv() * Mat(3, 1, CV_64FC1, zTemp);
//			camDirFile << zVec.t();
//			cout << zVec.t() << endl;
//			camDirFile.close();
//
//
//			cout << "相机位置：" << endl;
//			Mat camPos = rotM.inv() * (0 - tvec);
//			cout << camPos << endl;
//			ofstream camPosFile(result + "CamPos.txt");
//			camPosFile << camPos;
//			camPosFile.close();
//			//cout << "偏航角：" << endl;
//			//double theta_x = atan2(rotM.at<double>(2, 1), rotM.at<double>(2, 2));
//			//double theta_y = atan2(-rotM.at<double>(2, 0),
//			//	sqrt(rotM.at<double>(2, 1)*rotM.at<double>(2, 1) + rotM.at<double>(2, 2)*rotM.at<double>(2, 2)));
//			//double theta_z = atan2(rotM.at<double>(1, 0), rotM.at<double>(0, 0));
//			//cout << theta_x << " " << endl
//			//	<< theta_y << " " << endl
//			//	<< theta_z << endl;
//		}
//		else
//		{
//			printf("Input error!\n");
//			return -1;
//		}
//	}
//	return 0;
//}
