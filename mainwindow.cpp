#include <Python.h>
#include <torch/torch.h>
#include <torch/script.h>
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<iostream>
#include "ui_mainwindow.h"
#include <qfiledialog.h>
#include <QFileDialog>
#include <QDebug>
#include <QFile>
#include <QMimeData>
#include <string>
#include <sys/time.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <QSettings>
#include <QCompleter>
#include <QStringList>
#include <QMessageBox>
#include <QTime>
#include <time.h>
#include <QThread>
#include <mythread.h>
#include "trilinear_cuda.h"
#include "ostream"

using namespace std;
using namespace cv;
Mat image;
string fileName1 = "/home/ubuntu/Desktop/DJFiles/pt/learn_LUT0.txt";
string fileName2 = "/home/ubuntu/Desktop/DJFiles/pt/learn_LUT1.txt";
string fileName3 = "/home/ubuntu/Desktop/DJFiles/pt/learn_LUT2.txt";
torch::jit::script::Module module_Class =torch::jit::load("/home/ubuntu/Desktop/DJFiles/pt/dict_classifier.pt");

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    int a=580,b=800;
    this->move(400,300);
    this->setFixedSize(a, b);
    ui->label->setVisible(0);

}

MainWindow::~MainWindow()
{
    delete ui;
}
void MainWindow::on_pushButton_clicked()
{
    QString filename;
    filename = QFileDialog::getOpenFileName(this,
            tr("选择图像"),
            "",
            tr("Images (*.png *.bmp *.jpg *.tif *.GIF)"));
    if(filename.isEmpty() == false)
    {
        ui->lineEdit->setText(filename);
        int w1 = filename.lastIndexOf(".");
        int w2 = filename.lastIndexOf("/");
        str3 =filename.mid(w2,w1-w2).toStdString();
    }
    image = imread(filename.toLocal8Bit().data());
}

void MainWindow::on_pushButton_3_clicked()
{
        module_Class.to(at::kCUDA);
        torch::Tensor img_tensor = torch::from_blob(image.data, { 1, image.rows, image.cols, 3 }, torch::kByte);
        img_tensor = img_tensor.to(at::kCUDA);
        img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
        img_tensor = img_tensor.toType(torch::kFloat);
        img_tensor = img_tensor.div(255.0);
        torch::Tensor output = module_Class.forward({ img_tensor }).toTensor();
        output = output.squeeze(0).contiguous();
        // LUT
        torch::Tensor initLut = torch::ones({3,33,33,33});
        initLut = initLut.toType(torch::kFloat);
        initLut = initLut.to(at::kCUDA);
        // open luts file
        ifstream inFile1(fileName1.c_str(), ios_base::in);
        ifstream inFile2(fileName2.c_str(), ios_base::in);
        ifstream inFile3(fileName3.c_str(), ios_base::in);
        // load lut data
        istream_iterator<float>begin(inFile1);
        istream_iterator<float>end;
        vector<float> inData(begin, end);
        cv::Mat tmpMat = cv::Mat(inData);
        torch::Tensor tlut = torch::from_blob(
                    tmpMat.data, {3,33,33,33},
                    torch::kFloat32);
        istream_iterator<float>begin2(inFile2);
        istream_iterator<float>end2;
        vector<float> inData2(begin2, end2);
        cv::Mat tmpMat2 = cv::Mat(inData2);
        torch::Tensor tlut2 = torch::from_blob(
                    tmpMat2.data, {3,33,33,33},
                    torch::kFloat32);
        istream_iterator<float>begin3(inFile3);
        istream_iterator<float>end3;
        vector<float> inData3(begin3, end3);
        cv::Mat tmpMat3 = cv::Mat(inData3);
        torch::Tensor tlut3 = torch::from_blob(
                    tmpMat3.data, {3,33,33,33},
                    torch::kFloat32);
        tlut = tlut.to(at::kCUDA);
        tlut2 = tlut2.to(at::kCUDA);
        tlut3 = tlut3.to(at::kCUDA);
        initLut = tlut * output[0] + tlut2 * output[1] + tlut3 * output[2];
        img_tensor = img_tensor.contiguous();
        torch::Tensor res_output = img_tensor.new_zeros(img_tensor.sizes());
        res_output = res_output.to(at::kCUDA);
        int x = trilinear_forward_cuda(initLut,
                               img_tensor,
                               res_output,
                               33,
                               35937,
                               1.000001/32,
                               img_tensor.size(2),
                               img_tensor.size(3),
                               1);
        if(x == 1)
        {
            module_2.to(at::kCUDA);
            res_output = res_output.squeeze(0).detach().permute({1,2,0}).contiguous();
            res_output = res_output.mul_(255).add(0.5).clamp(0, 255).to(torch::kU8);
            res_output = res_output.to(torch::kCPU);
            Mat result(image.rows, image.cols, CV_8UC3, (uchar*)res_output.data_ptr());
        }
}


