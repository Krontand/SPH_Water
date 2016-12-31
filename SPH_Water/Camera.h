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

	vec3 eye;     // Позиция камеры

	vec3 center;  // Точка взгляда камеры (target)

	vec3 up;      // Вектор, всегда смотрящий "вверх" - нужен для корректного перехода к СК камеры

	vec3 eye_s;   // Начальные координаты камеры
	double anglex;  // Поворот в горизонтальной плоскости
	double angley;  // Поворот в вертикальной плоскости
	double dist;    // Расстояние от центра координат мировой СК

private:
	void set_pos();
};

