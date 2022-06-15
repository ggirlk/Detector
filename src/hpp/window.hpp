#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <gtkmm.h>

#include "detector.hpp"

/**
 * MainWindow - Class definition to create a window
 */
class MainWindow : public Gtk::Window {
public:
	MainWindow(int width, int height);
	virtual ~MainWindow() = default;

protected:
	bool on_key_press_event(GdkEventKey* event) override;
    
private:
	void objectDetection();
	void eyesDetection();
	void landmarksDetection();
	void directionsDetection();

	bool probablyInFullScreen;
	Gtk::Button m_button_objects;
	Gtk::Button m_button_eyes;
	Gtk::Button m_button_landmarks;
	Gtk::Button m_button_directions;
	Gtk::ButtonBox m_box_btns;
	Gtk::Box m_box;
	CameraDrawingArea cameraDrawingArea;
};

#endif
