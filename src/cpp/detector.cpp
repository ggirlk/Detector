#include "detector.hpp"

#include "globals.hpp"

#include "drawLandmarks.hpp"

using namespace std;
using namespace cv;
using namespace dnn;

/*************/
vector<string> classes; // will contain Model classes

vector<cv::Scalar> colors; // will contain some different colors for each class

float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
float maskThreshold = 0.3; // Mask threshold

int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image

bool detect = false;
bool DetectVisionAxis = false;
bool trackFaces = false;
bool detectEyes = false;
bool DetectLandMarks = false;

string device = "CPU"; // or GPU
string modelConfiguration = "src/yolo/yolov3.cfg";
string modelWeights = "src/yolo/yolov3.weights";
string classesFile = "src/yolo/coco.names";

Net myNet = readNetFromDarknet(modelConfiguration, modelWeights);

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
// Create an instance of Facemark
Ptr<Facemark> facemark = FacemarkLBF::create();

String lbfmodelFile = "src/cascade/lbfmodel.yaml";
String face_cascade_name = "src/cascade/haarcascade_frontalface_alt2.xml";
String eyes_cascade_name = "src/cascade/haarcascade_eye.xml";

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));

            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences

    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame, colors[classIds[idx]]);
    }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top,
	int right, int bottom, Mat& frame, Scalar color)
{
    //Draw a rectangle displaying the bounding box
    
    rectangle(frame, Point(left, top), Point(right, bottom), color, 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);

    
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

/*************/

CameraDrawingArea::CameraDrawingArea():
videoCapture(0) {
	// Lets refresh drawing area very now and then.
	everyNowAndThenConnection = Glib::signal_timeout().connect(sigc::mem_fun(*this, &CameraDrawingArea::everyNowAndThen), 100);
}

CameraDrawingArea::~CameraDrawingArea() {
	everyNowAndThenConnection.disconnect();
}

/**
 * Every now and then, we invalidate the whole Widget rectangle,
 * forcing a complete refresh.
 */
bool CameraDrawingArea::everyNowAndThen() {
	auto win = get_window();
	if (win) {
		Gdk::Rectangle r(0, 0, width, height);
		win->invalidate_rect(r, false);
	}
	
	// Don't stop calling me:
	return true;
}

/**
 * on_size_allocate - Called every time the widget has its allocation changed.
 * @allocation: new allocation
 * 
 * Return: nothing
 */
void CameraDrawingArea::on_size_allocate (Gtk::Allocation& allocation) {
	// Call the parent to do whatever needs to be done:
	DrawingArea::on_size_allocate(allocation);
	
	// Remember the new allocated size for resizing operation:
	width = allocation.get_width();
	height = allocation.get_height();

	// Get the classes names, and generate colors for each class
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line))
	{
		classes.push_back(line);
		cv::Scalar color(
		  (double)std::rand() / RAND_MAX * 255,
		  (double)std::rand() / RAND_MAX * 255,
		  (double)std::rand() / RAND_MAX * 255
		);
		colors.push_back(color);
	}
	// Set Backend
	if (device == "CPU")
    {
        cout << "Using CPU device" << endl;
		//myNet.setPreferableBackend(DNN_BACKEND_OPENCV);
        myNet.setPreferableBackend(DNN_TARGET_CPU);
        myNet.setPreferableTarget(0);
    }
    else if (device == "GPU")
    {
        cout << "Using GPU device" << endl;
        myNet.setPreferableBackend(DNN_BACKEND_CUDA);
        myNet.setPreferableTarget(DNN_TARGET_CUDA);
    }

    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) )
    {
        cout << "--(!)Error loading face cascade\n";
        exit(1);
    };
    if( !eyes_cascade.load( eyes_cascade_name ) )
    {
        cout << "--(!)Error loading eyes cascade\n";
        exit(1);
    };
    // Load landmark detector
   facemark->loadModel(lbfmodelFile);
}

/**
 * on_draw - Called every time the widget needs to be redrawn.
 * 			 This happens when the Widget got resized, or obscured by
 * 			 another object, or every now and then.
 * @cr: Cairo context
 * Return: true or false
 */
cv::Mat frame, eye_tpl;
cv::Rect eye_bb;

bool CameraDrawingArea::on_draw(const Cairo::RefPtr<Cairo::Context>& cr) {
	
	// Prevent the drawing if size is 0:
	if (width == 0 || height == 0) {
		return true;
	}

	// Capture one image from camera:
	videoCapture.read(webcam);
	// Fix Colors
	cvtColor(webcam, webcam, COLOR_BGR2RGB);
	// Resize it to the allocated size of the Widget.
	resize(webcam, output, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
/*
	if (GBlure)
	{
		GaussianBlur(output, output, Size(5,5), 3, 3);
	}
    if (CannyF)
    {
    	cvtColor(output, output, COLOR_BGR2GRAY);
    	Canny(output, output, 10, 100, 3, true);
    }*/
    vector<Rect> faces;

	if (detectEyes || DetectLandMarks || DetectVisionAxis)
	{
	  // Variable to store a video frame and its grayscale 
	  Mat gray;

      // Convert frame to grayscale because
      // face_cascade.load requires grayscale image.
      cvtColor(output, gray, COLOR_BGR2GRAY);

      // Detect faces
      face_cascade.detectMultiScale(gray, faces);
      if (detectEyes)
      {
	    equalizeHist( gray, gray );

	    for ( size_t i = 0; i < faces.size(); i++ )
	    {
	        std::vector<cv::Point2d> image_points;

	        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
	        //ellipse( output, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 1 );
	        Mat faceROI = gray( faces[i] );
	        //-- In each face, detect eyes
	        vector<Rect> eyes;
	        eyes_cascade.detectMultiScale( faceROI, eyes );
	        for ( size_t j = 0; j < eyes.size(); j++ )
	        {
	            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
	            int radius = 8;// cvRound( (eyes[j].width + eyes[j].height)*0.10 );
	            circle( output, eye_center, radius, Scalar( 255, 0, 0 ), 1 );
	        }

	    }
	  }

      // Variable for landmarks. 
      // Landmarks for one face is a vector of points
      // There can be more than one face in the image. Hence, we 
      // use a vector of vector of points. 
      vector< vector<Point2f> > landmarks;
      
      // Run landmark detector
      bool success = facemark->fit(output,faces,landmarks);
      
      if(success)
      {
        // If successful, render the landmarks on the face
      	if (DetectLandMarks)
        {
        	for(size_t i = 0; i < landmarks.size(); i++)
                {
                  drawLandmarks(output, landmarks[i]);
                }
        }
        if (DetectVisionAxis)
        {
        	for(size_t i = 0; i < landmarks.size(); i++)
            {

			    // 2D image points. If you change the image, you need to change vector

			    std::vector<cv::Point2d> image_points;

			    image_points.push_back( cv::Point2d(landmarks[i][30]) );    // Nose tip

			    image_points.push_back( cv::Point2d(landmarks[i][8]) );    // Chin

			    image_points.push_back( cv::Point2d(landmarks[i][36]) );     // Left eye left corner

			    image_points.push_back( cv::Point2d(landmarks[i][45]) );    // Right eye right corner

			    image_points.push_back( cv::Point2d(landmarks[i][48]) );    // Left Mouth corner

			    image_points.push_back( cv::Point2d(landmarks[i][54]) );    // Right mouth corner
			 

			    // 3D model points.

			    std::vector<cv::Point3d> model_points;

			    model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip

			    model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin

			    model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner

			    model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner

			    model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // Left Mouth corner

			    model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));       // Right mouth corner

		        // Camera internals

			    double focal_length = output.cols; // Approximate focal length.

			    Point2d center = cv::Point2d(output.cols/2,output.rows/2);

			    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);

			    cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion

			    //cout << "Camera Matrix " << endl << camera_matrix << endl ;

			    // Output rotation and translation

			    cv::Mat rotation_vector; // Rotation in axis-angle form

			    cv::Mat translation_vector;

			    // Solve for pose

			    cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

			    // Project a 3D point (0, 0, 1000.0) onto the image plane.

			    // We use this to draw a line sticking out of the nose

			    vector<Point3d> nose_end_point3D;

			    vector<Point2d> nose_end_point2D;

			    nose_end_point3D.push_back(Point3d(0,0,1000.0));

			    projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);
			    for(size_t i=0; i < image_points.size(); i++)
			    	circle(output, image_points[i], 3, Scalar(0,0,255), -1);

			    //cout << image_points[0] << endl;
			    cv::line(output,image_points[0], nose_end_point2D[0], cv::Scalar(255,0,0), 2);
			    circle(output, nose_end_point2D[0], 3, Scalar(0,255,0), -1);
			}

        }
   	 }

    }
	if (detect) {
		resize(webcam, output, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

		Mat frame, blob;
		VideoWriter video;
		string outputFile = "yolo_out_cpp.avi";

		blobFromImage(output, blob, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);

		//cout << "booom\n";
	    //Sets the input to the network
	    myNet.setInput(blob);

	    // Runs the forward pass to get output of the output layers
	    vector<Mat> outs;
	    myNet.forward(outs, getOutputsNames(myNet));
	    
	    // Remove the bounding boxes with low confidence
	    postprocess(output, outs);

	    // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	    vector<double> layersTimes;
	    double freq = getTickFrequency() / 1000;
	    double t = myNet.getPerfProfile(layersTimes) / freq;
	    string label = format("Inference time for a frame : %.2f ms", t);
	    //rectangle(output, (startX, startY), (endX, endY),
		//		COLORS[idx], 2)

	    putText(output, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
	    //imshow("Gray", output);

	    // Write the frame with the detection boxes
	    //Mat detectedFrame;
	    //output.convertTo(detectedFrame, CV_8U);
		//video.write(detectedFrame);
	}
	// Initializes a pixbuf sharing the same data as the mat:
	Glib::RefPtr<Gdk::Pixbuf> pixbuf =
		Gdk::Pixbuf::create_from_data((guint8*)output.data,
									  Gdk::COLORSPACE_RGB,
									  false,
									  8,
									  output.cols,
									  output.rows,
									  (int) output.step);

	// Display
	Gdk::Cairo::set_source_pixbuf(cr, pixbuf);
	cr->paint();

	// Don't stop calling me.
	return true;
}


