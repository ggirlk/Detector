#include "globals.hpp"
#include "window.hpp"
#include <syslog.h>

/**
 * MainWindow - start window object
 * @width: window width
 * @height: window height
 * 
 */
MainWindow::MainWindow(int width, int height):
probablyInFullScreen(false),
m_button_objects("Start/Stop Object Detection"),
m_button_eyes("Start/Stop Eyes Detection"),
m_button_landmarks("Start/Stop Face Landmarks Detection"),
m_button_directions("Start/Stop Vision axis Direction"),
m_box(Gtk::ORIENTATION_VERTICAL)
{
	// Configure this window:
	this->set_default_size(width, height);

	// Load Style Sheet
	Glib::RefPtr<Gtk::CssProvider> cssProvider = Gtk::CssProvider::create();
	cssProvider->load_from_path("src/css/style.css"); 

	Glib::RefPtr<Gtk::StyleContext> styleContext = Gtk::StyleContext::create();
	Glib::RefPtr<Gdk::Screen> screen = Gdk::Screen::get_default();//get default screen
	styleContext->add_provider_for_screen(screen, cssProvider, GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);//add provider for screen in all application

	Glib::RefPtr<Gtk::StyleContext> context;

	context = m_button_objects.get_style_context();
	context->add_class("button");

	context = m_button_eyes.get_style_context();
	context->add_class("button");

	context = m_button_landmarks.get_style_context();
	context->add_class("button");

	context = m_button_directions.get_style_context();
	context->add_class("button");
	
	// Connect the 'click' signal and start detecting
	m_button_objects.signal_clicked().connect(
	    sigc::mem_fun(*this, &MainWindow::objectDetection));
	m_button_objects.show();

	// Connect the 'click' signal and stop detection
	m_button_eyes.signal_clicked().connect(
	    sigc::mem_fun(*this, &MainWindow::eyesDetection));
	m_button_eyes.show();

	// Connect the 'click' signal and apply landmarks detection
	m_button_landmarks.signal_clicked().connect(
	    sigc::mem_fun(*this, &MainWindow::landmarksDetection));
	m_button_landmarks.show();

	// Connect the 'click' signal and apply vision axis detection
	m_button_directions.signal_clicked().connect(
	    sigc::mem_fun(*this, &MainWindow::directionsDetection));
	m_button_directions.show();

	// Connect the 'click' signal and apply edge detection
	//m_button_canny.signal_clicked().connect(
	//    sigc::mem_fun(*this, &MainWindow::toggleCanny));
	//m_button_canny.show();

	// Make the second label visible:
	cameraDrawingArea.show();
	
	// Pack all elements in the box:
	m_box.pack_start(m_button_objects, Gtk::PACK_SHRINK);
	m_box.pack_start(m_button_eyes, Gtk::PACK_SHRINK);
	m_box.pack_start(m_button_landmarks, Gtk::PACK_SHRINK);
	m_box.pack_start(m_button_directions, Gtk::PACK_SHRINK);

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

/**
 * objectDetection - Click button function to Start/Stop detecting objects.
 * 
 * Return: nothing
 */
void MainWindow::objectDetection() {
	if (detect)
		detect = false;
	else
		detect = true;
}

/**
 * eyesDetection - Click button function to Start/Stop detecting eyes.
 * 
 * Return: nothing
 */
void MainWindow::eyesDetection() {
	if (detectEyes)
		detectEyes = false;
	else
		detectEyes = true;
}
/**
 * landmarksDetection - Click button function to Start/Stop detecting face landmarks.
 * 
 * Return: nothing
 */
void MainWindow::landmarksDetection() {
	if (DetectLandMarks)
		DetectLandMarks = false;
	else
		DetectLandMarks = true;
}

/**
 * directionsDetection - Click button function to apply Eyes detection.
 * 
 * Return: nothing
 */
void MainWindow::directionsDetection() {
	
	if (DetectVisionAxis)
		DetectVisionAxis = false;
	else
		DetectVisionAxis = true;
}

