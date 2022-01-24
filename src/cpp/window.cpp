#include "window.hpp"
#include <syslog.h>

using namespace std;
using namespace cv;
using namespace dnn; 

/**
 * MainWindow - start window object
 * @width: window width
 * @height: window height
 * 
 */
MainWindow::MainWindow(int width, int height):
probablyInFullScreen(false),
m_button_start("Start Detecting"),
m_button_stop("Stop Detection"),
m_box(Gtk::ORIENTATION_VERTICAL)
{
	// Configure this window:
	this->set_default_size(width, height);

	// Connect the 'click' signal and start detecting
	m_button_start.signal_clicked().connect(
	    sigc::mem_fun(*this, &MainWindow::startDetecting));
	m_button_start.show();

	// Connect the 'click' signal and stop detection
	m_button_stop.signal_clicked().connect(
	    sigc::mem_fun(*this, &MainWindow::stopDetecting));
	m_button_stop.show();

	// Make the second label visible:
	cameraDrawingArea.show();
	
	// Pack all elements in the box:
	m_box.pack_start(m_button_start, Gtk::PACK_SHRINK);
	m_box.pack_start(m_button_stop, Gtk::PACK_SHRINK);
	m_box.pack_start(cameraDrawingArea, Gtk::PACK_EXPAND_WIDGET);

	// Add the box in this window:
	add(m_box);
	
	// Make the box visible:
	m_box.show();
	
	// Activate Key-Press events
	add_events(Gdk::KEY_PRESS_MASK);
}
/**
 * on_key_press_event - key press event handler
 * @event: Gdk key event
 * 
 * Return: true or false
 */
bool MainWindow::on_key_press_event(GdkEventKey* event)
{
	switch (event->keyval) {
		// Ctrl + C: Ends the app:
		case GDK_KEY_C:
		case GDK_KEY_c:
			if ((event->state & GDK_CONTROL_MASK) == GDK_CONTROL_MASK) {
				get_application()->quit();
			}
			return true;
			
		// [F] toggles fullscreen mode:
		case GDK_KEY_F:
		case GDK_KEY_f:
			if (probablyInFullScreen) {
				unfullscreen();
				probablyInFullScreen = false;
			} else {
				fullscreen();
				probablyInFullScreen = true;
			}
			return true;
			
		// [esc] exits fullscreen mode:
		case GDK_KEY_Escape:
			unfullscreen();
			probablyInFullScreen = false;
			return true;
	}
	
	return false;
}
