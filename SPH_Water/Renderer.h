#pragma once

#include "math/mathgl.h"
#include "Camera.h"
#include "OpenGL.h"
#include "Particles.h"

class Renderer
{
public:
	Renderer(int w, int h, Particles *p, Camera *cam);
	~Renderer();

	void render(Particles *particles, Camera *cam);

	void clear();

	void setMatrices(float *matr, float *vmatr, float *ipmatr, Camera *cam, float part);

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

private:
	mat4 projMatr;            // ������� �������������� ���������
	mat4 viewMatr;            // ������� �������� � �� ������
	mat4 viewPort;            // ������� �������� � �������� �����������

	int w;
	int h;

	// ��������� ��� �������� ��������������� ��������� ��������� � ��������
	GLuint shaderProgram = 0, vertexShader = 0, fragmentShader = 0;

	GLint projectionMatrixInverseLocation, projectionMatrixLocation, viewMatrixLocation, positionLocation, colorLocation;

	float *projectionMatrix = new float[16];
	float *viewMatrix = new float[16];
	float *projectionMatrixInverse = new float[16];

	// ������ ����� ������� ���� � ������ - 6 float �� ������� � �� ���� �������
	int VERTEX_SIZE = 6 * sizeof(float);

	// �������� ������ ������ �������
	int VERTEX_POSITION_OFFSET = 0;
	int VERTEX_COLOR_OFFSET = 3 * sizeof(float);

	// ���������� ��� �������� ��������������� VAO � VBO
	GLuint meshVAO = 0, meshVBO = 0;

};
