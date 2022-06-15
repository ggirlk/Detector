#ifndef GLOBALS_H
#define GLOBALS_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/face.hpp>
#include <gtkmm.h>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace cv::face;

extern vector<string> classes; // will contain Model classes

extern vector<cv::Scalar> colors; // will contain some different colors for each class

extern float confThreshold; // Confidence threshold
extern float nmsThreshold;  // Non-maximum suppression threshold
extern float maskThreshold; // Mask threshold

extern int inpWidth;  // Width of network's input image
extern int inpHeight; // Height of network's input image

/* buttons switchers */ 
extern bool detect;
extern bool DetectVisionAxis;
extern bool detectEyes;
extern bool DetectLandMarks;
extern bool trackFaces;

extern Net myNet; // Artificial Neural Network for object detection
extern string modelConfiguration; // the model architecture file
extern string modelWeights; // Model pretrained weights file
extern string classesFile; // this where the classes names stored

extern string device; // device configuration CPU or GPU

//extern CascadeClassifier face_cascade; // Cascade classifier for face detection
//extern CascadeClassifier eyes_cascade; // Cascade classifier for eye detection

#endif /* GLOBALS_H */