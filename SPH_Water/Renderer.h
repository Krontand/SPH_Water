#pragma once

#include "math/mathgl.h"
#include "Camera.h"

class Renderer
{
public:
	Renderer(int w, int h);
	~Renderer();

	void getMatrix(float *matr, float *vmatr, float *ipmatr, Camera *cam, float part);

	/*
	* ��������� ������� �������� � �� ������
	*/
	void setviewmatr(Camera *cam);

	/*
	* ��������� ������� ������������ �������������
	*/
	void setprojmatr(Camera *cam);

	/*
	* ��������� ������� �������� � �������� �����������
	*/
	void setViewPort(int x, int y, int w, int h, float part);

	mat4 projMatr;            // ������� �������������� ���������
	mat4 viewMatr;            // ������� �������� � �� ������
	mat4 viewPort;            // ������� �������� � �������� �����������

	int w;
	int h;
};
