#include "stdafx.h"
#include "Camera.h"

Camera::Camera(double x, double y, double z)
{
	anglex = acos(x / z);
	angley = acos((x*x + z*z) / y);
	dist = sqrt(x*x + y*y + z*z);
	this->eye.x = x;
	this->eye.y = y;
	this->eye.z = z;
	center.x = 0;
	center.y = 0;
	center.z = 0;
	up.x = 0;
	up.y = 1;
	up.z = 0;

	eye_s = eye;
}
