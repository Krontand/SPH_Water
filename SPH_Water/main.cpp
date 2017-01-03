#include "stdafx.h"
#define WIN32_LEAN_AND_MEAN 1

#include <windows.h>
#include "OpenGL.h"
#include "common.h"
#include "GLWindow.h"
#include "Scene.h"

#define WIDTH 1360
#define HEIGHT 768

extern "C" {
	_declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}

// положение курсора и его смещение с последнего кадра
static int16_t cursorPos[2] = { 0, 0 }, rotateDelta[2] = { 0, 0 }, oldPos[2] = { -1, -1 }, oldwheel = 0, wheel = 0;

static Scene *scene;

// инициализаця OpenGL
bool GLWindowInit(const GLWindow *window)
{
	ASSERT(window);

	scene = new Scene(WIDTH, HEIGHT);
	return true;
}

// очистка OpenGL
void GLWindowClear(const GLWindow *window)
{
	ASSERT(window);

	scene->clear();
}

// функция рендера
void GLWindowRender(const GLWindow *window)
{
	ASSERT(window);

	scene->render();
}

// функция обновления
void GLWindowUpdate(const GLWindow *window, double deltaTime)
{
	ASSERT(window);
	ASSERT(deltaTime >= 0.0); // проверка на возможность бага

	scene->rotate_cam(rotateDelta[0], rotateDelta[1]);
	if (wheel != 0)
		scene->change_cam_dist(wheel);
	rotateDelta[0] = 0;
	rotateDelta[1] = 0;
	wheel = 0;

	scene->update_particles(deltaTime);
}

// функция обработки ввода с клавиатуры и мыши
void GLWindowInput(const GLWindow *window)
{
	ASSERT(window);
	// центр окна
	int32_t xCenter = window->width / 2, yCenter = window->height / 2;

	// выход из приложения по кнопке Esc
	if (InputIsKeyPressed(VK_ESCAPE))
		GLWindowDestroy();

	// переключение между оконным и полноэкранным режимом
	// осуществляется по нажатию комбинации Alt+Enter
	if (InputIsKeyDown(VK_MENU) && InputIsKeyPressed(VK_RETURN))
		GLWindowSetSize(window->width, window->height, !window->fullScreen);
	InputGetCursorPos(cursorPos, cursorPos + 1);
	if (InputIsButtonDown(0))
	{
		if (oldPos[0] == -1)
		{
			oldPos[0] = cursorPos[0];
			oldPos[1] = cursorPos[1];
		}
		else
		{
			rotateDelta[0] += cursorPos[0] - oldPos[0];
			rotateDelta[1] += cursorPos[1] - oldPos[1];
			oldPos[0] = cursorPos[0];
			oldPos[1] = cursorPos[1];
		}
	}
	else
	{
		oldPos[0] = cursorPos[0];
		oldPos[1] = cursorPos[1];
	}
	int16_t w;
	InputGetWheelScrollTimes(w);
	wheel += w - oldwheel;
	oldwheel = w;
	
}



int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
	int result;

	LoggerCreate("SPH_Water_log.log");
//	CUDA_init();
	if (!GLWindowCreate("SPH Water (lol, water)", WIDTH, HEIGHT, false))
		return 1;

	result = GLWindowMainLoop();

	GLWindowDestroy();
	LoggerDestroy();

	return result;
}
