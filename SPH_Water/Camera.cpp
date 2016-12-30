#include "stdafx.h"
#include "Camera.h"

Camera::Camera(double x, double y, double z)
{
	anglex = 0;
	angley = 0;
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

void Camera::rotate(float ax, float ay)
{
//	ax /= 1000;
//	ay /= 1000;
	this->angley += ay/80;

	if (this->angley < 1E-2)
		this->angley = 1E-2;
	if (this->angley > 3.14159)
		this->angley = 3.14159;
	this->anglex += ax/80;
	this->eye.z = this->dist * sin(this->angley) * cos(this->anglex);
	this->eye.x = this->dist * sin(this->angley) * sin(this->anglex);
	this->eye.y = this->dist * cos(this->angley);
}
