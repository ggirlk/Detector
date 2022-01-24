#include "window.hpp"

/**
 * main - main function to run the app
 * @argc: number of args
 * @argv: arguments vector
 *
 * Return: 
 */
int main (int argc, char *argv[]) {

	auto app = Gtk::Application::create(argc, argv, "Detector");
	MainWindow mainWindow(500, 500);
	app->run(mainWindow);
	return 0;// app->run(mainWindow);	
}
