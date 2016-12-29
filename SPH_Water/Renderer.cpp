#include "stdafx.h"
#include "Renderer.h"
#include "renderer.h"

Renderer::Renderer(int w, int h)
{
	this->w = w;
	this->h = h;
}

Renderer::~Renderer()
{
}

void Renderer::setViewPort(int x, int y, int w, int h, float part)
{
	viewPort.m[3] = x + w / 2.f;
	viewPort.m[7] = y + h / 2.f;

	viewPort.m[11] = 1000 / 2.f;

	viewPort.m[0] = w / 2.f / part;
	viewPort.m[5] = h / 2.f / part;
	viewPort.m[10] = 1000 / 2.f;
	viewPort.m[15] = 0;
}

void Renderer::setviewmatr(Camera *cam)
{
	viewMatr = GLLookAt(cam->eye, cam->center, cam->up);
}

void Renderer::setprojmatr(Camera *cam)
{
	projMatr = GLPerspective(90, 1.33, 1, 200);
}


void Renderer::getMatrix(float *matr, float *vmatr, float *ipmatr, Camera *cam, float part)
{
	this->setprojmatr(cam);
	this->setviewmatr(cam);

	mat4 m = projMatr * viewMatr;
	mat4 ip = inverse(projMatr);

	for (int i = 0; i < 16; i++)
	{
		matr[i] = m.m[i];
		vmatr[i] = viewMatr.m[i];
		ipmatr[i] = ip.m[i];
	}
}
