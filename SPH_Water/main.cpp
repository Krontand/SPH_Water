#include "stdafx.h"
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>

#include "common.h"
#include "OpenGL.h"
#include "GLWindow.h"

#include "Scene.h"

#define WIDTH 800
#define HEIGHT 600

// положение курсора и его смещение с последнего кадра
static int16_t cursorPos[2] = {0, 0}, rotateDelta[2] = {0, 0}, oldPos[2] = {-1, -1};

static Scene *scene;

// инициализаця OpenGL
bool GLWindowInit(const GLWindow *window)
{
	ASSERT(window);
	// спрячем курсор
	ShowCursor(false);

	scene = new Scene(WIDTH, HEIGHT);
	return true;
}

// очистка OpenGL
void GLWindowClear(const GLWindow *window)
{
	ASSERT(window);

	scene->clear();
	ShowCursor(true);
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
	rotateDelta[0] = 0;
	rotateDelta[1] = 0;
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

int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
	int result;

	LoggerCreate("SPH_Water_log.log");

	if (!GLWindowCreate("SPH Water (lol, water)", WIDTH, HEIGHT, false))
		return 1;

	result = GLWindowMainLoop();

	GLWindowDestroy();
	LoggerDestroy();

	return result;
}
