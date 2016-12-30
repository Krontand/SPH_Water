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

private:
	mat4 projMatr;            // Матрица перспективного искажения
	mat4 viewMatr;            // Матрица перехода к СК камеры
	mat4 viewPort;            // Матрица перехода к экранным координатам

	int w;
	int h;

	// пременные для хранения идентификаторов шейдерной программы и шейдеров
	GLuint shaderProgram = 0, vertexShader = 0, fragmentShader = 0;

	GLint projectionMatrixInverseLocation, projectionMatrixLocation, viewMatrixLocation, positionLocation, colorLocation;

	float *projectionMatrix = new float[16];
	float *viewMatrix = new float[16];
	float *projectionMatrixInverse = new float[16];

	// размер одной вершины меша в байтах - 6 float на позицию и на цвет вершины
	int VERTEX_SIZE = 6 * sizeof(float);

	// смещения данных внутри вершины
	int VERTEX_POSITION_OFFSET = 0;
	int VERTEX_COLOR_OFFSET = 3 * sizeof(float);

	// переменные для хранения идентификаторов VAO и VBO
	GLuint meshVAO = 0, meshVBO = 0;

};
