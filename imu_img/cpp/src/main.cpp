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
//	//�ļ����
//	long   hFile   =   0;
//	//�ļ���Ϣ
//	struct _finddata_t fileinfo;
//	string p;
//	if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1)
//	{
//		do
//		{
//			//�����Ŀ¼,����֮
//			//�������,�����б�
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
//			cout << "can not find chessboard corners!\n"; // �Ҳ����ǵ�
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
//			/* �����ؾ�ȷ�� */
//			// image_points_buf ��ʼ�Ľǵ�����������ͬʱ��Ϊ����������λ�õ����
//			// Size(5,5) �������ڴ�С
//			// ��-1��-1����ʾû������
//			// TermCriteria �ǵ�ĵ������̵���ֹ����, ����Ϊ���������ͽǵ㾫�����ߵ����
//			cornerSubPix(gray, imageCornerBuff, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
//
//			imagesCorners.push_back(imageCornerBuff); // ���������ؽǵ�
//
//			/* ��ͼ������ʾ�ǵ�λ�� */
//			drawChessboardCorners(img, boardSize, imageCornerBuff, false); // ������ͼƬ�б�ǽǵ�
//			imshow("Camera Calibration", img);							   // ��ʾͼƬ
//			waitKey(100);												   //��ͣ
//		}
//	}
//
//	int CornerNum = boardSize.width * boardSize.height; // ÿ��ͼƬ���ܵĽǵ���
//	Size squareSize = Size(10, 10);						/* ʵ�ʲ����õ��ı궨����ÿ�����̸�Ĵ�С */
//	vector<vector<Point3f>> objectPoints;				/* ����궨���Ͻǵ����ά���� */
//
//	//Mat cameraMatrix; /* ������ڲ������� */
//	vector<int> point_counts; // ÿ��ͼ���нǵ������
//	//Mat distCoeffs;       /* �������5������ϵ����k1,k2,p1,p2,k3 */
//	vector<Mat> tvecsMat; /* ÿ��ͼ�����ת���� */
//	vector<Mat> rvecsMat; /* ÿ��ͼ���ƽ������ */
//
//	/* ��ʼ���궨���Ͻǵ����ά���� */
//	for (int t = 0; t < files.size(); t++)
//	{
//		vector<Point3f> tempPointSet;
//		for (int i = 0; i < boardSize.height; i++)
//		{
//			for (int j = 0; j < boardSize.width; j++)
//			{
//				Point3f realPoint;
//
//				/* ����궨�������������ϵ��z=0��ƽ���� */
//				realPoint.x = i * squareSize.width;
//				realPoint.y = j * squareSize.height;
//				realPoint.z = 0;
//				tempPointSet.push_back(realPoint);
//			}
//		}
//		objectPoints.push_back(tempPointSet);
//	}
//
//	/* ��ʼ��ÿ��ͼ���еĽǵ��������ٶ�ÿ��ͼ���ж����Կ��������ı궨�� */
//	for (int i = 0; i < files.size(); i++)
//	{
//		point_counts.push_back(boardSize.width * boardSize.height);
//	}
//
//	/* ��ʼ�궨 */
//	// object_points ��������ϵ�еĽǵ����ά����
//	// image_points_seq ÿһ���ڽǵ��Ӧ��ͼ�������
//	// image_size ͼ������سߴ��С
//	// cameraMatrix ������ڲξ���
//	// distCoeffs ���������ϵ��
//	// rvecsMat �������ת����
//	// tvecsMat �����λ������
//	// 0 �궨ʱ�����õ��㷨
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
    //cout << "ƫ���ǣ�" << endl;
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
//			printf("-d\t�ҳ�ָ���ļ�������Ч��ͼƬ����Ч��ͼƬ����ɾ��\n");
//			printf("-i\t�궨�ڲβ��衣ָ����Ŀ¼�����̸�ͼƬ�뱣����images�ļ����£���������浽result�ļ�����\n");
//			printf("-e\t�궨��β��衣ָ����Ŀ¼�����̸�ͼƬ�뱣����images�ļ����£���������浽result�ļ�����\n");
//			return 0;
//		}
//		if (strcmp(currArg, "-d") == 0 || strcmp(currArg, "-D") == 0)
//		{
//			cout << "Deleting invalid images..." << endl;
//			i++;
//			if (i >= argc)
//			{
//				printf("ָ����Ŀ¼·������");
//				return -1;
//			}
//			imgdir = argv[i];
//			cout << "imgdir is:" << imgdir << endl;
//			deleteInvalidImages(imgdir);
//			return 0;
//		}
//		else if (strcmp(currArg, "-i") == 0 || strcmp(currArg, "-I") == 0)
//		{
//			cout << "�궨����ڲ�..." << endl;
//			i++;
//			if (i >= argc)
//			{
//				printf("ָ����Ŀ¼·������");
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
//			cout << "�ڲξ���" << endl;
//			cout << cameraMatrix << endl;
//			cout << "����ϵ����" << endl;
//			cout << distCoeffs << endl
//				<< endl;
//			return 0;
//		}
//		else if (strcmp(currArg, "-e") == 0 || strcmp(currArg, "-E") == 0)
//		{
//			/*
//			* ��ʼ�궨���
//			*/
//			cout << "�궨������..." << endl;
//			i++;
//			if (i >= argc)
//			{
//				printf("ָ����Ŀ¼·������");
//				return -1;
//			}
//			ROOTDIR = argv[i];
//			cout << "ROOTDIR IS:" << ROOTDIR << endl;
//			calibrateCam(cameraMatrix, distCoeffs, ROOTDIR+"\\images");
//
//			string result = ROOTDIR + "\\result\\";
//			cout << "��ȡ2D-3D���..." << endl;
//			ifstream pointPairFile(result + "pair.txt");
//			if (!pointPairFile.is_open())
//			{
//				cout << "��2D-3D����ļ�ʧ��" << endl;
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
//			cout << "2D Points��" << endl;
//			cout << imagePoints << endl;
//			cout << "3D Points��" << endl;
//			cout << obejctPoints << endl
//				<< endl;
//			pointPairFile.close();
//
//			cout << "������..." << endl;
//			Mat rvec, tvec;
//			solvePnP(obejctPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, CV_EPNP);
//			// solveP3P();
//			// solvePoly();
//			// solve();
//			double rotD[9] = {0};
//			Mat rotM(3, 3, CV_64FC1, rotD);
//			Rodrigues(rvec, rotM);
//			cout << "��ת����\n" << rotM << endl;
//			cout << "ƽ�ƾ���\n" << tvec << endl;
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
//			cout << "����������" << endl;
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
//			cout << "���λ�ã�" << endl;
//			Mat camPos = rotM.inv() * (0 - tvec);
//			cout << camPos << endl;
//			ofstream camPosFile(result + "CamPos.txt");
//			camPosFile << camPos;
//			camPosFile.close();
//			//cout << "ƫ���ǣ�" << endl;
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
