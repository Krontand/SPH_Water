#pragma once

#include "math/vec3.h"
#include <math.h>

class Camera
{
public:
	Camera(double x, double y, double z);

	vec3 eye;     // ������� ������

	vec3 center;  // ����� ������� ������ (target)

	vec3 up;      // ������, ������ ��������� "�����" - ����� ��� ����������� �������� � �� ������

	vec3 eye_s;   // ��������� ���������� ������
	double anglex;  // ������� � �������������� ���������
	double angley;  // ������� � ������������ ���������
	double dist;    // ���������� �� ������ ��������� ������� ��
};

