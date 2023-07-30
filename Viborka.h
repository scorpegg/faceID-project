#pragma once

#include <iostream>

#include <string>

#include <fstream>

#include <boost/filesystem.hpp>

#include <boost/algorithm/string/predicate.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/objdetect/objdetect.hpp>

#include <opencv2/face.hpp>

#include <filesystem>

#include <list>

#include <map>

using namespace std;

using namespace cv;

namespace bf = boost::filesystem;

using namespace cv::face;

float** main1()

{
  Mat pil_image;

  Mat pil_image1;

  string p;

  int nomer = 0;

  vector<Rect> faces;  //

  vector<vector<Point2f> > landmarks;

  float** izm1 = new float*[33];

  for (int i = 0; i < 33; i++) izm1[i] = new float[17];

  CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");

  Ptr<Facemark> facemark = FacemarkLBF::create();

  facemark->loadModel("lbfmodel.yaml.txt");

  bf::path s = (__FILE__);

  bf::path dir = s.parent_path();

  bf::path file("Fase");

  bf::path full_path = dir / file;

  bf::recursive_directory_iterator itr(full_path);

  while (itr != bf::recursive_directory_iterator())

  {
    if (boost::algorithm::ends_with(itr->path().string(), ".png") ||
        boost::algorithm::ends_with(itr->path().string(), ".jpg")) {
      bf::path path = itr->path().string();

      bf::path label = path.parent_path().stem();

      cout << path << " " << label << endl;

      p = bf::canonical(path).string();

      pil_image = imread(p);  //читаем фото

      cvtColor(pil_image, pil_image1, COLOR_BGR2GRAY);

      faceDetector.detectMultiScale(pil_image1, faces);

      bool success1 = facemark->fit(pil_image, faces, landmarks);

      if (success1)

      {
        for (int i = 0; i < landmarks.size(); i++)

        {
          float XLI = (landmarks[i][45].x + landmarks[i][42].x) /
                      2;  //считаем координады центра глаз

          float YLI = (landmarks[i][45].y + landmarks[i][42].y) / 2;

          float XRI = (landmarks[i][39].x + landmarks[i][36].x) / 2;

          float YRI = (landmarks[i][39].y + landmarks[i][36].y) / 2;

          float DX = XRI - XLI;  //расстояния между средними точками глаз вдоль
                                 //осей Х и Y (катеты)

          float DY = YRI - YLI;

          float LI =
              sqrt(DX * DX + DY * DY);  //расстояние от центра глаз (гипотенуза)

          izm1[nomer][14] = LI;

          float DX1 = landmarks[i][36].x - landmarks[i][39].x;

          float DY1 = landmarks[i][36].y - landmarks[i][39].y;

          float L1 = sqrt(DX1 * DX1 +
                          DY1 * DY1);  //расстояние от переносицы до подбородка

          izm1[nomer][0] = L1;

          float DX2 = landmarks[i][42].x - landmarks[i][45].x;

          float DY2 = landmarks[i][42].y - landmarks[i][45].y;

          float L2 = sqrt(DX2 * DX2 + DY2 * DY2);  //расстояние от крайнего
                                                   //уголка левого глаза до
                                                   //крайней левой точки носа

          izm1[nomer][1] = L2;

          float DX3 = landmarks[i][39].x - landmarks[i][42].x;

          float DY3 = landmarks[i][39].y - landmarks[i][42].y;

          float L3 = sqrt(DX3 * DX3 + DY3 * DY3);  //расстояние от крайнего
                                                   //уголка правого глаза до
                                                   //крайней правой точки носа

          izm1[nomer][2] = L3;

          float DX4 = landmarks[i][42].x - landmarks[i][33].x;

          float DY4 = landmarks[i][42].y - landmarks[i][33].y;

          float L4 = sqrt(
              DX4 * DX4 +
              DY4 * DY4);  //расстояние от правой точки уголка губ до подбородка

          izm1[nomer][3] = L4;

          float DX5 = landmarks[i][39].x - landmarks[i][33].x;

          float DY5 = landmarks[i][39].y - landmarks[i][33].y;

          float L5 = sqrt(
              DX5 * DX5 +
              DY5 * DY5);  //расстояние от правой точки уголка губ до подбородка

          izm1[nomer][4] = L5;

          float DX6 = landmarks[i][36].x - landmarks[i][33].x;

          float DY6 = landmarks[i][36].y - landmarks[i][33].y;

          float L6 = sqrt(DX6 * DX6 + DY6 * DY6);

          izm1[nomer][5] = L6;

          float DX7 = landmarks[i][45].x - landmarks[i][33].x;

          float DY7 = landmarks[i][45].y - landmarks[i][33].y;

          float L7 = sqrt(DX7 * DX7 + DY7 * DY7);

          izm1[nomer][6] = L7;

          float DX8 = landmarks[i][36].x - landmarks[i][48].x;

          float DY8 = landmarks[i][36].y - landmarks[i][48].y;

          float L8 = sqrt(DX8 * DX8 + DY8 * DY8);

          izm1[nomer][7] = L8;

          float DX9 = landmarks[i][45].x - landmarks[i][54].x;

          float DY9 = landmarks[i][45].y - landmarks[i][54].y;

          float L9 = sqrt(DX9 * DX9 + DY9 * DY9);

          izm1[nomer][8] = L9;

          float DX10 = landmarks[i][48].x - landmarks[i][54].x;

          float DY10 = landmarks[i][48].y - landmarks[i][54].y;

          float L10 = sqrt(DX10 * DX10 + DY10 * DY10);

          izm1[nomer][9] = L10;

          float DX11 = landmarks[i][33].x - landmarks[i][48].x;

          float DY11 = landmarks[i][33].y - landmarks[i][48].y;

          float L11 = sqrt(DX11 * DX11 + DY11 * DY11);

          izm1[nomer][10] = L11;

          float DX12 = landmarks[i][33].x - landmarks[i][54].x;

          float DY12 = landmarks[i][33].y - landmarks[i][54].y;

          float L12 = sqrt(DX12 * DX12 + DY12 * DY12);

          izm1[nomer][11] = L12;

          float DX13 = landmarks[i][42].x - landmarks[i][54].x;

          float DY13 = landmarks[i][42].y - landmarks[i][54].y;

          float L13 = sqrt(DX13 * DX13 + DY13 * DY13);

          izm1[nomer][12] = L13;

          float DX14 = landmarks[i][39].x - landmarks[i][48].x;

          float DY14 = landmarks[i][39].y - landmarks[i][48].y;

          float L14 = sqrt(DX14 * DX14 + DY14 * DY14);

          izm1[nomer][13] = L14;
        }
      }
    }

    nomer++;

    ++itr;
  }

  return (izm1);
}