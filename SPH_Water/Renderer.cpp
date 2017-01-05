#include "stdafx.h"
#include "Renderer.h"

Renderer::Renderer(int w, int h, Particles *p, Camera *cam)
{
	this->w = w;
	this->h = h;

	vec4 lightDir = vec4(1, 1, 1, 0);

	uint8_t  *shaderSource;
	uint32_t sourceLength;


	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_DEPTH_TEST);

	// ������������� ������� �� ��� ����
	glViewport(0, 0, w, h);

	// ��������� OpenGL
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(1.0f);

	// �������� ��������� ��������� � ������� ��� ���
	shaderProgram = glCreateProgram();
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	// �������� ��������� ������
	if (!LoadFile("shaders/lesson.vs", true, &shaderSource, &sourceLength))
		LOG_ERROR("��������� ������ �� ������");

	// ������� ������� �������� ��� � ������������ ���
	glShaderSource(vertexShader, 1, (const GLchar**)&shaderSource, (const GLint*)&sourceLength);
	glCompileShader(vertexShader);

	delete[] shaderSource;

	// �������� ������ �������
	if (ShaderStatus(vertexShader, GL_COMPILE_STATUS) != GL_TRUE)
		LOG_ERROR("��������� ������ �� ���������������");

	// �������� ����������� ������
	if (!LoadFile("shaders/lesson.fs", true, &shaderSource, &sourceLength))
		LOG_ERROR("����������� ������ �� ������");

	// ������� ������� �������� ��� � ������������ ���
	glShaderSource(fragmentShader, 1, (const GLchar**)&shaderSource, (const GLint*)&sourceLength);
	glCompileShader(fragmentShader);

	delete[] shaderSource;

	// �������� ������ �������
	if (ShaderStatus(fragmentShader, GL_COMPILE_STATUS) != GL_TRUE)
		LOG_ERROR("(ShaderStatus(fragmentShader, GL_COMPILE_STATUS) != GL_TRUE)");

	// ����������� ����������� ������� � ��������� ���������
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// �������� �� ���� �� ������
	OPENGL_CHECK_FOR_ERRORS();

	// �������� ��������� ��������� � �������� ������� ��������
	glLinkProgram(shaderProgram);
	if (ShaderProgramStatus(shaderProgram, GL_LINK_STATUS) != GL_TRUE)
		LOG_ERROR("(ShaderProgramStatus(shaderProgram, GL_LINK_STATUS) != GL_TRUE)");

	// ������� ������ ��������
	glUseProgram(shaderProgram);

	// �������� ������������� �������
	setMatrices(projectionMatrix, viewMatrix, projectionMatrixInverse, cam, 1.0);

	projectionMatrixLocation = glGetUniformLocation(shaderProgram, "modelViewProjectionMatrix");
	viewMatrixLocation = glGetUniformLocation(shaderProgram, "modelViewMatrix");
	projectionMatrixInverseLocation = glGetUniformLocation(shaderProgram, "projectionMatrixInverse");

	//	lightDir = inverse(transpose(mat4(viewMatrix))) * lightDir;
	if (projectionMatrixLocation != -1)
		glUniformMatrix4fv(projectionMatrixLocation, 1, GL_TRUE, projectionMatrix);

	if (viewMatrixLocation != -1)
		glUniformMatrix4fv(viewMatrixLocation, 1, GL_TRUE, viewMatrix);

	if (projectionMatrixInverseLocation != -1)
		glUniformMatrix4fv(projectionMatrixInverseLocation, 1, GL_TRUE, projectionMatrixInverse);

	// �������� �� ������������ ��������� ���������
	glValidateProgram(shaderProgram);
	if (ShaderProgramStatus(shaderProgram, GL_VALIDATE_STATUS) != GL_TRUE)
		LOG_ERROR("ShaderProgramStatus(shaderProgram, GL_VALIDATE_STATUS) != GL_TRUE");

	glUniform4fv(glGetUniformLocation(shaderProgram, "lightDir"), 1, lightDir.v);

	// �������� �� ���� �� ������
	OPENGL_CHECK_FOR_ERRORS();

	// �������� � ���������� Vertex Array Object (VAO)
	glGenVertexArrays(1, &meshVAO);
	glBindVertexArray(meshVAO);

	// �������� � ���������� Vertex Buffer Object (VBO)
	glGenBuffers(1, &meshVBO);
	glBindBuffer(ARR_TARGET, meshVBO);

	// �������� VBO ������� ������������
	glBufferData(ARR_TARGET, p->MESH_VERTEX_COUNT * VERTEX_SIZE,
		p->triangleMesh, GL_DYNAMIC_DRAW_ARB);

	HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&cudaParticles,
		meshVBO, cudaGraphicsMapFlagsNone));
	HANDLE_ERROR(cudaGraphicsMapResources(1, &cudaParticles, NULL));
	HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaParticles));
	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cudaParticles, NULL));

	// ������� ������� �������� 'position' �� �������
	positionLocation = glGetAttribLocation(shaderProgram, "position");
	if (positionLocation != -1)
	{
		// �������� �� ������� ��������� ������� � VBO
		glVertexAttribPointer(positionLocation, 3, GL_FLOAT, GL_FALSE,
			VERTEX_SIZE, (const GLvoid*)VERTEX_POSITION_OFFSET);
		// �������� ������������� ��������
		glEnableVertexAttribArray(positionLocation);
	}

	// ������� ������� �������� 'color' �� �������
	colorLocation = glGetAttribLocation(shaderProgram, "color");
	if (colorLocation != -1)
	{
		// �������� �� ������� ��������� ������� � VBO
		glVertexAttribPointer(colorLocation, 3, GL_FLOAT, GL_FALSE,
			VERTEX_SIZE, (const GLvoid*)VERTEX_COLOR_OFFSET);
		// �������� ������������� ��������
		glEnableVertexAttribArray(colorLocation);
	}

	OPENGL_CHECK_FOR_ERRORS();
}

Renderer::~Renderer()
{
}

void Renderer::render(Particles *particles, Camera *cam)
{
	// �������� ������������� �������
	setMatrices(projectionMatrix, viewMatrix, projectionMatrixInverse, cam, 1.0);

	glUniformMatrix4fv(projectionMatrixLocation, 1, GL_TRUE, projectionMatrix);
	glUniformMatrix4fv(viewMatrixLocation, 1, GL_TRUE, viewMatrix);
	glUniformMatrix4fv(projectionMatrixInverseLocation, 1, GL_TRUE, projectionMatrixInverse);
	// ������� ����� ����� � �������
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// ������ ��������� ��������� ��������
	glUseProgram(shaderProgram);

	// �������� VBO ������� ������������
//	glBufferData(ARR_TARGET, particles->MESH_VERTEX_COUNT * VERTEX_SIZE,
//		particles->triangleMesh, GL_DYNAMIC_DRAW_ARB);

	// ������ ������������ �� VBO ������������ � VAO
	glDrawArrays(GL_POINTS, 0, particles->MESH_VERTEX_COUNT);


}

void Renderer::clear()
{
	// ������� VAO � VBO
	glBindBuffer(BUF_TARGET, 0);
	glDeleteBuffers(1, &meshVBO);

	glBindVertexArray(0);
	glDeleteVertexArrays(1, &meshVAO);

	glUseProgram(0);
	glDeleteProgram(shaderProgram);

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
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

float * Renderer::map_resource()
{
//	HANDLE_ERROR(cudaGraphicsMapResources(1, &cudaParticles, NULL));
//	HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaParticles));
	return devPtr;
}

void Renderer::unmap_resource()
{
//	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cudaParticles, NULL));
}

void Renderer::setviewmatr(Camera *cam)
{
	viewMatr = GLLookAt(cam->eye, cam->center, cam->up);
}

void Renderer::setprojmatr(Camera *cam)
{
	projMatr = GLPerspective(90, 1360.0/768.0, 0.1, 200);
}


void Renderer::setMatrices(float *matr, float *vmatr, float *ipmatr, Camera *cam, float part)
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
