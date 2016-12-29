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
	* Вычислить матрицу перехода к СК камеры
	*/
	void setviewmatr(Camera *cam);

	/*
	* Вычислить матрицу центрального проецирования
	*/
	void setprojmatr(Camera *cam);

	/*
	* Вычислить матрицу перехода к экранным координатам
	*/
	void setViewPort(int x, int y, int w, int h, float part);

	mat4 projMatr;            // Матрица перспективного искажения
	mat4 viewMatr;            // Матрица перехода к СК камеры
	mat4 viewPort;            // Матрица перехода к экранным координатам

	int w;
	int h;
};
