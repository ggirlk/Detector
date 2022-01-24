#ifndef CAMERA_DRAWING_AREA_H
#define CAMERA_DRAWING_AREA_H

#include <opencv2/opencv.hpp>
#include <gtkmm.h>


using namespace std;
using namespace cv;
using namespace dnn;


class CameraDrawingArea :
public Gtk::DrawingArea {
public:
    CameraDrawingArea();
    virtual ~CameraDrawingArea();
    
protected:
    bool on_draw(const Cairo::RefPtr<Cairo::Context>& cr) override;
	void on_size_allocate (Gtk::Allocation& allocation) override;
	
	bool everyNowAndThen();

private:
	sigc::connection everyNowAndThenConnection;
	cv::VideoCapture videoCapture;
	cv::Mat webcam;
	cv::Mat output;
	int width, height;
};

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& out);

void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top,
	int right, int bottom, Mat& frame, Scalar color);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);
#endif
