#pragma once

#include "math/vec3.h"

#define _USE_MATH_DEFINES
#include <cmath>

class Camera
{
public:
	Camera(double x, double y, double z);

	void rotate(float ax, float ay);
	void scale_dist(float scale);

	vec3 eye;     // ������� ������

	vec3 center;  // ����� ������� ������ (target)

	vec3 up;      // ������, ������ ��������� "�����" - ����� ��� ����������� �������� � �� ������

	vec3 eye_s;   // ��������� ���������� ������
	double anglex;  // ������� � �������������� ���������
	double angley;  // ������� � ������������ ���������
	double dist;    // ���������� �� ������ ��������� ������� ��

private:
	void set_pos();
};

