#include <opencv2/opencv.hpp>

#include "detector.hpp"
#include "window.hpp"

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
string device = "CPU"; // or GPU
string modelConfiguration = "src/yolov3.cfg";
string modelWeights = "src/yolov3.weights";
//Net myNet = readNetFromDarknet(modelConfiguration, modelWeights);
Net myNet = readNetFromDarknet(modelConfiguration, modelWeights);
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
    
    Mat outDetections = outs[0];
    Mat outMasks = outs[0];
 
    // Output size of masks is NxCxHxW where
    // N - number of detected boxes
    // C - number of classes (excluding background)

    // HxW - segmentation shape

    //const int numDetections = outDetections.size[2];

    //const int numClasses = outMasks.size[1];

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
        //cout << colors[idx] << " idx: " << classIds[idx] << endl;
		/********
        // Extract the mask for the object
        Mat objectMask(box.width, box.height,CV_32F, outMasks.ptr<float>(i,idx));
        //drawBox(frame, classId, score, box, objectMask);   
        // Resize the mask, threshold, color and apply it on the image
	    //resize(objectMask, objectMask, Size(box.width, box.height), 0, 0, cv::INTER_LINEAR);
        cout << "heeeere" << endl;

	    Mat mask = (objectMask > maskThreshold);
	    Mat coloredRoi = (0.3 * colors[idx] + 0.7 * frame(box));
	    coloredRoi.convertTo(coloredRoi, CV_8UC3);

	    // Draw the contours on the image
	    vector<Mat> contours;
	    Mat hierarchy;
	    mask.convertTo(mask, CV_8U);
	    findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	    drawContours(coloredRoi, contours, -1, colors[idx], 5, LINE_8, hierarchy, 100);
	    coloredRoi.copyTo(frame(box), mask); */    
        
    }
}
void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask)
{
	/*// Resize the mask, threshold, color and apply it on the image
    resize(objectMask, objectMask, Size(box.width, box.height));
    Mat mask = (objectMask > maskThreshold);
    Mat coloredRoi = (0.3 * colors[idx] + 0.7 * frame(box));
    coloredRoi.convertTo(coloredRoi, CV_8UC3);

    // Draw the contours on the image
    vector<Mat> contours;
    Mat hierarchy;
    mask.convertTo(mask, CV_8U);
    findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    drawContours(coloredRoi, contours, -1, colors[idx], 5, LINE_8, hierarchy, 100);
    coloredRoi.copyTo(frame(box), mask);*/
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

	string classesFile = "src/coco.names";
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
	
	if (device == "CPU")
    {
        cout << "Using CPU device" << endl;
		myNet.setPreferableBackend(DNN_BACKEND_OPENCV);
        myNet.setPreferableBackend(DNN_TARGET_CPU);
    }
    else if (device == "GPU")
    {
        cout << "Using GPU device" << endl;
        myNet.setPreferableBackend(DNN_BACKEND_CUDA);
        myNet.setPreferableTarget(DNN_TARGET_CUDA);
    }
}
/**
 * startDetecting - Click button function to Start detecting.
 * 
 * Return: nothing
 */
void MainWindow::startDetecting() {
	detect = true;
}
/**
 * stopDetecting - Click button function to Stop detecting.
 * 
 * Return: nothing
 */
void MainWindow::stopDetecting() {
	detect = false;
}
/**
 * on_draw - Called every time the widget needs to be redrawn.
 * 			 This happens when the Widget got resized, or obscured by
 * 			 another object, or every now and then.
 * @cr: Cairo context
 * Return: true or false
 */
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
    
	if (detect) {
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
	    string label = format("Inference time for a output : %.2f ms", t);
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
