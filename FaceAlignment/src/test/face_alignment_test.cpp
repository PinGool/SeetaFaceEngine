/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is an example of how to use SeetaFace engine for face alignment, the
 * face alignment method described in the following paper:
 *
 *
 *   Coarse-to-Fine Auto-Encoder Networks (CFAN) for Real-Time Face Alignment, 
 *   Jie Zhang, Shiguang Shan, Meina Kan, Xilin Chen. In Proceeding of the
 *   European Conference on Computer Vision (ECCV), 2014
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Jie Zhang (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <string.h>
#include <memory.h>
#ifdef __cplusplus
#include "opencv2/opencv.hpp"
#endif
#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "face_detection.h"
#include "face_alignment.h"

#ifdef _WIN32
std::string DATA_DIR = "../../data/";
//std::string MODEL_DIR = "../../model/";
std::string MODEL_DIR = "";
#else
std::string DATA_DIR = "./data/";
//std::string MODEL_DIR = "./model/";
std::string MODEL_DIR = "";
#endif

using namespace cv;
using namespace std;

//seeta::FaceDetection detector("../../../FaceDetection/model/seeta_fd_frontal_v1.0.bin");
seeta::FaceDetection detector("seeta_fd_frontal_v1.0.bin");

Mat getwarpAffineImg(Mat &src, seeta::FacialLandmark* landmarks)
{
	Mat oral; src.copyTo(oral);
	Point2d eyesCenter = Point2d((landmarks[0].x + landmarks[1].x) * 0.5f, (landmarks[0].y + landmarks[1].y) * 0.5f);

	double dy = (landmarks[1].y - landmarks[0].y);
	double dx = (landmarks[1].x - landmarks[0].x);
	double angle = atan2(dy, dx) * 180.0 / CV_PI; // Convert from radians to degrees.

	Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, 1.0);
	Mat rot;
	//
	warpAffine(src, rot, rot_mat, src.size());
	vector<Point2d> marks;

	//
	for (int n = 0; n<5/*landmarks.size()*/; n++)
	{
		Point2d p = Point2d(0, 0);
		p.x = rot_mat.ptr<double>(0)[0] * landmarks[n].x + rot_mat.ptr<double>(0)[1] * landmarks[n].y + rot_mat.ptr<double>(0)[2];
		p.y = rot_mat.ptr<double>(1)[0] * landmarks[n].x + rot_mat.ptr<double>(1)[1] * landmarks[n].y + rot_mat.ptr<double>(1)[2];
		marks.push_back(p);
		landmarks[n].x = p.x;
		landmarks[n].y = p.y;
	}

	return rot;
}

std::vector<seeta::FaceInfo> detectFace(IplImage *image, seeta::ImageData* image_data) {
	int im_width = image->width;
	int im_height = image->height;
	unsigned char* data = new unsigned char[im_width * im_height];
	unsigned char* data_ptr = data;
	unsigned char* image_data_ptr = (unsigned char*)image->imageData;
	int h = 0;
	for (h = 0; h < im_height; h++) {
		memcpy(data_ptr, image_data_ptr, im_width);
		data_ptr += im_width;
		image_data_ptr += image->widthStep;
	}

	//seeta::ImageData image_data;
	image_data->data = data;
	image_data->width = im_width;
	image_data->height = im_height;
	image_data->num_channels = 1;

	// Detect faces
	std::vector<seeta::FaceInfo> faces = detector.Detect(*image_data);

	int32_t face_num = static_cast<int32_t>(faces.size());

	if (face_num == 0)
	{
		delete[]data;
		cvReleaseImage(&image);
		//cvReleaseImage(&img_color);
		return *(new std::vector<seeta::FaceInfo>);
	}

	return faces;
}

bool procFaceImage(string fullpath, string path, string filename, string ext, string dst_path, string in_size)
{
  // Initialize face detection model
	//seeta::FaceDetection detector("../../../FaceDetection/model/seeta_fd_frontal_v1.0.bin");
  detector.SetMinFaceSize(40);
  detector.SetScoreThresh(2.f);
  detector.SetImagePyramidScaleFactor(0.8f);
  detector.SetWindowStep(4, 4);

  // Initialize face alignment model 
  seeta::FaceAlignment point_detector((MODEL_DIR + "seeta_fa_v1.1.bin").c_str());

  //load image
  IplImage *img_grayscale = NULL;
  img_grayscale = cvLoadImage(/*(DATA_DIR + "image_0001.jpg")*/fullpath.c_str(), 0);
  if (img_grayscale == NULL)
  {
    printf("%s\n", fullpath.c_str());
    printf("[0]img_grayscale == NULL\n");
    return false;
  }

  IplImage *outImg = NULL;
  while(img_grayscale->width  > 1024 + 1024 || img_grayscale->height > 768 + 512 ){
    outImg = cvCreateImage(cvSize(img_grayscale->width / 2, img_grayscale->height / 2), 
                                     img_grayscale->depth, 
                                     img_grayscale->nChannels);
    cvPyrDown(img_grayscale, outImg);
    img_grayscale = outImg;
  }

  /*IplImage *img_color = cvLoadImage((DATA_DIR + "image_0001.jpg").c_str(), 1);
  int pts_num = 5;*/

  printf("detectFace now!\n");
  seeta::ImageData image_data;
  std::vector<seeta::FaceInfo> faces = detectFace(img_grayscale, &image_data);
  if (faces.size() == (0)) {
	  printf("[1]detectFace error!\n");
	  return false;
  }
  printf("face number = %d\n",faces.size());

  printf("PointDetectLandmarks now!\n");

  string result_path = (/*path*/dst_path + "/" + filename + "_result." + ext);
  // Detect 5 facial landmarks
  seeta::FacialLandmark points[5];

  {
    IplImage *img_color = cvLoadImage(/*(DATA_DIR + "image_0001.jpg")*/fullpath.c_str(), 1);

    while(img_color->width  > 1024 + 1024 || img_color->height > 768 + 512 ){
    outImg = cvCreateImage(cvSize(img_color->width / 2, img_color->height / 2), 
                                     img_color->depth, 
                                     img_color->nChannels);
    cvPyrDown(img_color, outImg);
    img_color = outImg;
  }

    for(int idx = 0;idx < faces.size(); idx++){
      cvRectangle(img_color, cvPoint(faces[idx].bbox.x, faces[idx].bbox.y), cvPoint(faces[idx].bbox.x + faces[idx].bbox.width - 1, faces[idx].bbox.y + faces[idx].bbox.height - 1), CV_RGB(255, 0, 0));
    }
    cvSaveImage(result_path.c_str(), img_color);
    //printf("Show result image\n");
    //cvShowImage("result", img_color);
  }

  for(int idx = 0;idx < faces.size(); idx++){
    printf("Proc No.%d\n", idx);
  point_detector.PointDetectLandmarks(image_data, faces[idx], points);

  IplImage *img_color = cvLoadImage(/*(DATA_DIR + "image_0001.jpg")*/fullpath.c_str(), 1);
  int pts_num = 5;
  cv::Mat img = cv::cvarrToMat(img_color);
  Mat retImg = getwarpAffineImg(img, points);
    Mat dstResizeImg;
	IplImage* dstimg_tmp = NULL;
	int resize_num = 0;
    
  IplImage qImg = IplImage(retImg);

  char ch_idx[3] ={0};
  sprintf(ch_idx, "%d", idx);
  char ch_size[5] = {0};
  sprintf(ch_size, "%d", atoi(in_size.c_str()));

  IplImage *dst_gray = cvCreateImage(cvGetSize(&qImg), qImg.depth, 1);//
  cvCvtColor(&qImg, dst_gray, CV_BGR2GRAY);//
  seeta::ImageData image_data_inner;
  std::vector<seeta::FaceInfo> faces_inner = detectFace(dst_gray, &image_data_inner);
  if (faces_inner.size() == (0)) {
	  printf("[2]detectFace error!\n");
	  return false;
  }
  char ch_x1[5] = {0};
  char ch_y1[5] = {0};
  char ch_x2[5] = {0};
  char ch_y2[5] = {0};
  sprintf(ch_x1, "%d", faces_inner[idx].bbox.x);
  sprintf(ch_y1, "%d", faces_inner[idx].bbox.y);
  sprintf(ch_x2, "%d", faces_inner[idx].bbox.x + faces_inner[idx].bbox.width);
  sprintf(ch_y2, "%d", faces_inner[idx].bbox.y + faces_inner[idx].bbox.height);
  string save_path = (/*path*/dst_path + "/" + filename + "_crop_" + ch_size + "_" + ch_idx + "_" + ch_x1 + "_" + ch_y1 + "_" + ch_x2 + "_" + ch_y2 +"." + ext);
  try{
    cvSetImageROI(&qImg, cvRect(faces_inner[idx].bbox.x, faces_inner[idx].bbox.y, faces_inner[idx].bbox.width, faces_inner[idx].bbox.height));
    CvSize dst_size;
    if(in_size != ""){
  	  resize_num = atoi(in_size.c_str());
    }

    if(resize_num != 0){
	  dst_size.height = resize_num;
	  dst_size.width = resize_num;
	  dstimg_tmp = cvCreateImage(dst_size, IPL_DEPTH_8U, 3);
	  cvResize(&qImg, dstimg_tmp);
	  cvSaveImage(save_path.c_str(), dstimg_tmp);
    } else {
	  cvSaveImage(save_path.c_str(), &qImg);
    }
    cvResetImageROI(&qImg);
  }catch(...){
    printf("Exception occured!\n");

    
    cvSetImageROI(img_grayscale, cvRect(faces[idx].bbox.x, faces[idx].bbox.y, faces[idx].bbox.width, faces[idx].bbox.height));
    CvSize dst_size;
    if(in_size != ""){
  	  resize_num = atoi(in_size.c_str());
    }

    if(resize_num != 0){
	  dst_size.height = resize_num;
	  dst_size.width = resize_num;
	  dstimg_tmp = cvCreateImage(dst_size, IPL_DEPTH_8U, 3);
	  cvResize(img_grayscale, dstimg_tmp);
	  cvSaveImage(save_path.c_str(), dstimg_tmp);
    } else {
	  cvSaveImage(save_path.c_str(), img_grayscale);
    }
    cvResetImageROI(img_grayscale);
  }

  // Release memory
  cvReleaseImage(&img_color);
  delete[] image_data_inner.data;
  }
  
  //cvSaveImage(result_path, &qImg);
  cvReleaseImage(&img_grayscale);
  //delete[]data;
  delete[] image_data.data;
  return true;
}

void splitname(const char *szfullfilename, char *szpathname, char *szfilename, char *szextname)
{
    int i, j;
    
    i = 0;
    while (szfullfilename[i] != '\0')
        i++;
    while (szfullfilename[i] != '.')
        i--;
    
    j = 0;
    i++;
    while((szextname[j] = szfullfilename[i]) != '\0')
    {
        i++;
        j++;
    }
    i -= j;
    while (szfullfilename[i] != '\\' && szfullfilename[i] != '/' )
        i--;
    
    for (j = 0; j <= i; j++)
    {
        szpathname[j] = szfullfilename[j];
    }
    szpathname[j] = '\0';
    
    j = 0;
    i++;
    while((szfilename[j] = szfullfilename[i]) != '.')
    {
        i++;
        j++;
    }
    szfilename[j] = '\0';
}

void testsplit(){
    char szfullfilename[255] = "C:\\windows\\help.txt";
    char szpathname[255];
    char szfilename[255];
    char szextname[255];
    
    splitname(szfullfilename, szpathname, szfilename, szextname);
    
    printf("%s\n", szfullfilename);
    printf("path: %s\n", szpathname);
    printf("file: %s\n", szfilename);
    printf("ext: %s\n", szextname);
}

int isTxtFile(char* argv){
	int i, j;
    
    	i = 0;
    	while (argv[i] != '\0')
     	   i++;
    	while (argv[i] != '.')
     	   i--;
	j = 0;
    	i++;
	if(strcmp("txt", &argv[i]) == 0){
		return 1;
	} else {
		return 0;
	}
}

int main(int argc, char** argv)
{
	bool ret;
    	char args1[1024] = {0};
	char args2[1024] = {0};
	char args3[1024] = {0};
	int blank_num = 0;
	int index = 0;
	int idx = 0;
	int bTxtFile = 0;
	char txtLine[1024] = {0};
	int sucNum = 0;
	int errNum = 0;
	FILE* fp = NULL;
    	//testsplit();
	printf("argc = %d\n",argc);
    	for(int i = 0;i<argc;i++){
		printf("%s\n",argv[i]);
	}
	if(argc == 1){
		{

			fp = fopen("image.txt", "r");

if(fp == 0)printf("fp == 0\n");
			while(feof(fp) == 0){
				index = 0;
				blank_num = 0;
				memset(txtLine, 0, 1024);
				fgets(txtLine, 1024, fp);
puts(txtLine);
				{

				while(txtLine[index] != ' ' && txtLine[index] != 0){
					args1[index] = txtLine[index];
					index++;
				}

				if(index == 0){
					continue;
				}				

				if(txtLine[index] == 0){
					printf("[1] command is : command src_full_path dst_path [resize]\n");
					return 0;
				} else
				if(txtLine[index] == ' '){

					blank_num++;
				}

				index++;
				idx = index;
				while(txtLine[index] != ' ' && txtLine[index] != 0){
					args2[index - idx] = txtLine[index];
					index++;
				}

				if(txtLine[index] == ' '){

					blank_num++;
				}
				index++;
				idx = index;

				if(txtLine[index-1] != 0){

					while(txtLine[index] != 0){
						args3[index - idx] = txtLine[index];
						index++;
					}
				} 

		
				{

        				char szpathname[255];
        				char szfilename[255];
        				char szextname[255];

        				splitname( args1, szpathname, szfilename, szextname );

    					if(blank_num == 1){

      						ret = procFaceImage(args1, szpathname, szfilename, szextname, args2, "");
    					} else if(blank_num == 2){

      						ret = procFaceImage(args1, szpathname, szfilename, szextname, args2, args3);      
    					}

					if (ret == false) {
						printf("[0]ProcFaceImage failed!\n");
						errNum = errNum + 1;
					}
					else {
						printf("Done!\n");
						sucNum = sucNum + 1;
					}
					}
				}
			}

			fclose(fp);
			printf("sucNum = %d\n",sucNum);
			printf("errNum = %d\n",errNum);
		}
	} else
	if(argc == 2){
		
		bTxtFile = isTxtFile(argv[1]);
		if(bTxtFile == 1){
			fp = fopen(argv[1], "r");

			while(feof(fp) == 0){
				index = 0;
				blank_num = 0;
				memset(txtLine, 0, 1024);
				fgets(txtLine, 1024, fp);
puts(txtLine);
				{

				while(txtLine[index] != ' ' && txtLine[index] != 0){
					args1[index] = txtLine[index];
					index++;
				}

				if(index == 0){
					continue;
				}				

				if(txtLine[index] == 0){
					printf("[1] command is : command src_full_path dst_path [resize]\n");
					return 0;
				} else
				if(txtLine[index] == ' '){
					blank_num++;
				}

				index++;
				idx = index;
				while(txtLine[index] != ' ' && txtLine[index] != 0){
					args2[index - idx] = txtLine[index];
					index++;
				}

				if(txtLine[index] == ' '){
					blank_num++;
				}
				index++;
				idx = index;

				if(txtLine[index-1] != 0){
					while(txtLine[index] != 0){
						args3[index - idx] = txtLine[index];
						index++;
					}
				} 

		
				{

        				char szpathname[255];
        				char szfilename[255];
        				char szextname[255];

        				splitname( args1, szpathname, szfilename, szextname );

    					if(blank_num == 1){
      						ret = procFaceImage(args1, szpathname, szfilename, szextname, args2, "");
    					} else if(blank_num == 2){
      						ret = procFaceImage(args1, szpathname, szfilename, szextname, args2, args3);      
    					}

					if (ret == false) {
						printf("[1]ProcFaceImage failed!\n");
						errNum = errNum + 1;
					}
					else {
						printf("Done!\n");
						sucNum = sucNum + 1;
					}
					}
				}
			}

			fclose(fp);
			printf("sucNum = %d\n",sucNum);
			printf("errNum = %d\n",errNum);
		} else
		{

			while(argv[1][index] != ' ' && argv[1][index] != 0){
				args1[index] = argv[1][index];
				index++;
			}
      
			if(argv[1][index] == 0){
				printf("[2] command is : command src_full_path dst_path [resize]\n");
				return 0;
			} else
			if(argv[1][index] == ' '){
				blank_num++;
			}

			index++;
			idx = index;
			while(argv[1][index] != ' ' && argv[1][index] != 0){
				args2[index - idx] = argv[1][index];
				index++;
			}

			if(argv[1][index] == ' '){
				blank_num++;
			}
			index++;
			idx = index;

			if(argv[1][index-1] != 0){
				while(argv[1][index] != 0){
					args3[index - idx] = argv[1][index];
					index++;
				}
			} 

		
			{

        			char szpathname[255];
        			char szfilename[255];
        			char szextname[255];

        			splitname( args1, szpathname, szfilename, szextname );

    				if(blank_num == 1){
      					ret = procFaceImage(args1, szpathname, szfilename, szextname, args2, "");
    				} else if(blank_num == 2){
      					ret = procFaceImage(args1, szpathname, szfilename, szextname, args2, args3);      
    				}

				if (ret == false) {
					printf("[2]ProcFaceImage failed!\n");
				}
				else {
					printf("Done!\n");
				}
			}
		}

	} else

	if (argc != 4 && argc != 3 ) {
		printf("[3] command is : command src_full_path dst_path [resize]\n");
	}
	else {

        char szpathname[255];
        char szfilename[255];
        char szextname[255];

        splitname( argv[1], szpathname, szfilename, szextname );

    if(argc == 3){
      ret = procFaceImage(argv[1], szpathname, szfilename, szextname, argv[2], "");
    } else {
      ret = procFaceImage(argv[1], szpathname, szfilename, szextname, argv[2], argv[3]);      
    }

		if (ret == false) {
			printf("[3]ProcFaceImage failed!\n");
		}
		else {
			printf("Done!\n");
		}
	}
	//ret = procFaceImage("C:\\Users\\chengstone\\Desktop\\ML\\SeetaFaceEngine-master\\FaceAlignment\\data\\image_0001.jpg", "C:\\Users\\chengstone\\Desktop\\ML\\SeetaFaceEngine-master\\FaceAlignment\\data", "image_0001", "jpg");
	return 0;
}
