#include <iostream>

#include <conio.h>

#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/objdetect/objdetect.hpp>

#include "opencv2/imgcodecs.hpp"

#include <opencv2/face.hpp>

#include "drawLandmarks.hpp"

#include "Viborka.h"

using namespace std;

using namespace cv;

using namespace cv::face;

int main(int argc, char** argv)

{

    setlocale(LC_ALL, "Russian");

    CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");

    Ptr<Facemark> facemark = FacemarkLBF::create();

    facemark->loadModel("lbfmodel.yaml.txt");

    int n = 0;

    int k = 0; //Хранит значение нужных трёх строк

    cout << "Введите идентификационный номер:" << endl;

    cin >> n;

    float** izm = main1();

    string name[11] = { "Valya", "Vasyta", "Golovanov", "Gulevich", "Lisa", "Melnik", "Nartov", "Safronov", "Sasha", "Skobkin", "Strelnikov" };

    VideoCapture cam(0);

    Mat frame, gray;

    while (cam.read(frame))

    {

        vector<Rect> faces;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        faceDetector.detectMultiScale(gray, faces);

        vector<vector<Point2f> > landmarks;

        bool success = facemark->fit(frame, faces, landmarks);

        if (success)

        {

            //for (size_t i = 0; i < faces.size(); i++)

            //{

            //cv::rectangle(frame, faces[i], Scalar(0, 255, 0), 3);

            //}

            for (int i = 0; i < landmarks.size(); i++)

            {

                float XLI = (landmarks[i][45].x + landmarks[i][42].x) / 2; //считаем координады центра глаз

                float YLI = (landmarks[i][45].y + landmarks[i][42].y) / 2;

                float XRI = (landmarks[i][39].x + landmarks[i][36].x) / 2;

                float YRI = (landmarks[i][39].y + landmarks[i][36].y) / 2;

                cv::line(frame, Point(XLI, YLI), Point(XRI, YRI), Scalar(0, 0, 255), 2);

                float DX = XRI - XLI; //расстояния между средними точками глаз вдоль осей Х и Y (катеты)

                float DY = YRI - YLI;

                float LI = sqrt(DX * DX + DY * DY); //расстояние от центра глаз (гипотенуза)

                putText(frame, std::to_string(LI), Point(landmarks[i][27].x, landmarks[i][27].y), 1, 2, Scalar(0, 0, 255), 2);

                float DX1 = landmarks[i][36].x - landmarks[i][39].x;

                float DY1 = landmarks[i][36].y - landmarks[i][39].y;

                float L1 = sqrt(DX1 * DX1 + DY1 * DY1); //расстояние от переносицы до подбородка

                float DX2 = landmarks[i][42].x - landmarks[i][45].x;

                float DY2 = landmarks[i][42].y - landmarks[i][45].y;

                float L2 = sqrt(DX2 * DX2 + DY2 * DY2); //расстояние от крайнего уголка левого глаза до крайней левой точки носа

                float DX3 = landmarks[i][39].x - landmarks[i][42].x;

                float DY3 = landmarks[i][39].y - landmarks[i][42].y;

                float L3 = sqrt(DX3 * DX3 + DY3 * DY3); //расстояние от крайнего уголка правого глаза до крайней правой точки носа

                float DX4 = landmarks[i][42].x - landmarks[i][33].x;

                float DY4 = landmarks[i][42].y - landmarks[i][33].y;

                float L4 = sqrt(DX4 * DX4 + DY4 * DY4); //расстояние от правой точки уголка губ до подбородка

                float DX5 = landmarks[i][39].x - landmarks[i][33].x;

                float DY5 = landmarks[i][39].y - landmarks[i][33].y;

                float L5 = sqrt(DX5 * DX5 + DY5 * DY5); //расстояние от правой точки уголка губ до подбородка

                float DX6 = landmarks[i][36].x - landmarks[i][33].x;

                float DY6 = landmarks[i][36].y - landmarks[i][33].y;

                float L6 = sqrt(DX6 * DX6 + DY6 * DY6);

                float DX7 = landmarks[i][45].x - landmarks[i][33].x;

                float DY7 = landmarks[i][45].y - landmarks[i][33].y;

                float L7 = sqrt(DX7 * DX7 + DY7 * DY7);

                float DX8 = landmarks[i][36].x - landmarks[i][48].x;

                float DY8 = landmarks[i][36].y - landmarks[i][48].y;

                float L8 = sqrt(DX8 * DX8 + DY8 * DY8);

                float DX9 = landmarks[i][45].x - landmarks[i][54].x;

                float DY9 = landmarks[i][45].y - landmarks[i][54].y;

                float L9 = sqrt(DX9 * DX9 + DY9 * DY9);

                float DX10 = landmarks[i][48].x - landmarks[i][54].x;

                float DY10 = landmarks[i][48].y - landmarks[i][54].y;

                float L10 = sqrt(DX10 * DX10 + DY10 * DY10);

                float DX11 = landmarks[i][33].x - landmarks[i][48].x;

                float DY11 = landmarks[i][33].y - landmarks[i][48].y;

                float L11 = sqrt(DX11 * DX11 + DY11 * DY11);

                float DX12 = landmarks[i][33].x - landmarks[i][54].x;

                float DY12 = landmarks[i][33].y - landmarks[i][54].y;

                float L12 = sqrt(DX12 * DX12 + DY12 * DY12);

                float DX13 = landmarks[i][42].x - landmarks[i][54].x;

                float DY13 = landmarks[i][42].y - landmarks[i][54].y;

                float L13 = sqrt(DX13 * DX13 + DY13 * DY13);

                float DX14 = landmarks[i][39].x - landmarks[i][48].x;

                float DY14 = landmarks[i][39].y - landmarks[i][48].y;

                float L14 = sqrt(DX14 * DX14 + DY14 * DY14);

                for (int i = 0; i < 33; i++) {

                    if (i / 3 == n - 1)

                        k = i;
                }

                float n1 = izm[k - 2][14] / LI; //Коэффицент подобия для ракурса анфас

                float n2 = izm[k - 1][14] / LI; //Коэффицент подобия для ракурса полоборота влево

                float n3 = izm[k][14] / LI; //Коэффицент подобия для ракурса полоборота вправо

                float ln1, ln2, ln3, ln4, ln5, ln6, ln7, ln8, ln9, ln10, ln11, ln12, ln13, ln14; //Значения масштабированных измерений

                for (int i = 0; i < 33; i++) { //Отклонения для ракурса анфаз (в столбце izm[i][8])

                    ln1 = L1 * n1;

                    ln2 = L2 * n1;

                    ln3 = L3 * n1;

                    ln4 = L4 * n1;

                    ln5 = L5 * n1;

                    ln6 = L6 * n1;

                    ln7 = L7 * n1;

                    ln8 = L8 * n1;

                    ln9 = L9 * n1;

                    ln10 = L10 * n1;

                    ln11 = L11 * n1;

                    ln12 = L12 * n1;

                    ln13 = L13 * n1;

                    ln14 = L14 * n1;

                    izm[i][15] = (izm[i][0] - ln1) * (izm[i][0] - ln1) + (izm[i][1] - ln2) * (izm[i][1] - ln2) + (izm[i][2] - ln3) * (izm[i][2] - ln3) + (izm[i][3] - ln4) * (izm[i][3] - ln4) + (izm[i][4] - ln5) * (izm[i][4] - ln5) + (izm[i][5] - ln6) * (izm[i][5] - ln6) + (izm[i][6] - ln7) * (izm[i][6] - ln7) + (izm[i][7] - ln8) * (izm[i][7] - ln8) + (izm[i][8] - ln9) * (izm[i][8] - ln9) + (izm[i][9] - ln10) * (izm[i][9] - ln10) + (izm[i][10] - ln11) * (izm[i][10] - ln11) + (izm[i][11] - ln12) * (izm[i][11] - ln12) + (izm[i][12] - ln13) * (izm[i][12] - ln13) + (izm[i][13] - ln14) * (izm[i][13] - ln14);
                }

                for (int i = 0; i < 33; i++) { //Отклонения для ракурса полоборота влево (в столбце izm[i][9])

                    ln1 = L1 * n2;

                    ln2 = L2 * n2;

                    ln3 = L3 * n2;

                    ln4 = L4 * n2;

                    ln5 = L5 * n2;

                    ln6 = L6 * n2;

                    ln7 = L7 * n2;

                    ln8 = L8 * n2;

                    ln9 = L9 * n2;

                    ln10 = L10 * n2;

                    ln11 = L11 * n2;

                    ln12 = L12 * n2;

                    ln13 = L13 * n2;

                    ln14 = L14 * n2;

                    izm[i][16] = (izm[i][0] - ln1) * (izm[i][0] - ln1) + (izm[i][1] - ln2) * (izm[i][1] - ln2) + (izm[i][2] - ln3) * (izm[i][2] - ln3) + (izm[i][3] - ln4) * (izm[i][3] - ln4) + (izm[i][4] - ln5) * (izm[i][4] - ln5) + (izm[i][5] - ln6) * (izm[i][5] - ln6) + (izm[i][6] - ln7) * (izm[i][6] - ln7) + (izm[i][7] - ln8) * (izm[i][7] - ln8) + (izm[i][8] - ln9) * (izm[i][8] - ln9) + (izm[i][9] - ln10) * (izm[i][9] - ln10) + (izm[i][10] - ln11) * (izm[i][10] - ln11) + (izm[i][11] - ln12) * (izm[i][11] - ln12) + (izm[i][12] - ln13) * (izm[i][12] - ln13) + (izm[i][13] - ln14) * (izm[i][13] - ln14);
                }

                for (int i = 0; i < 33; i++) { //Отклонения для ракурса полоборота влево (в столбце izm[i][10])

                    ln1 = L1 * n3;

                    ln2 = L2 * n3;

                    ln3 = L3 * n3;

                    ln4 = L4 * n3;

                    ln5 = L5 * n3;

                    ln6 = L6 * n3;

                    ln7 = L7 * n3;

                    ln8 = L8 * n3;

                    ln9 = L9 * n3;

                    ln10 = L10 * n3;

                    ln11 = L11 * n3;

                    ln12 = L12 * n3;

                    ln13 = L13 * n3;

                    ln14 = L14 * n3;

                    izm[i][17] = (izm[i][0] - ln1) * (izm[i][0] - ln1) + (izm[i][1] - ln2) * (izm[i][1] - ln2) + (izm[i][2] - ln3) * (izm[i][2] - ln3) + (izm[i][3] - ln4) * (izm[i][3] - ln4) + (izm[i][4] - ln5) * (izm[i][4] - ln5) + (izm[i][5] - ln6) * (izm[i][5] - ln6) + (izm[i][6] - ln7) * (izm[i][6] - ln7) + (izm[i][7] - ln8) * (izm[i][7] - ln8) + (izm[i][8] - ln9) * (izm[i][8] - ln9) + (izm[i][9] - ln10) * (izm[i][9] - ln10) + (izm[i][10] - ln11) * (izm[i][10] - ln11) + (izm[i][11] - ln12) * (izm[i][11] - ln12) + (izm[i][12] - ln13) * (izm[i][12] - ln13) + (izm[i][13] - ln14) * (izm[i][13] - ln14);
                }

                int min = izm[0][15];

                for (int i = 0; i < 33; i++) {

                    for (int j = 15; j < 17; j++) {

                        if (min > izm[i][j]) {

                            min = izm[i][j];

                            k = i;
                        }
                    }
                }

                for (int j = 0; j < 33; j++) {

                    if (j == k) {

                        k = j / 3;

                        //putText(frame, name[k], Point(landmarks[i][27].x, landmarks[i][27].y), 1, 2, Scalar(0, 0, 255), 2);
                    }
                }
            }
        }

        imshow("Facial Landmark Detection", frame);

        cam.set(CAP_PROP_FPS, 1);

        if (waitKey(1) == 27)
            break;
    }

    return 0;
}