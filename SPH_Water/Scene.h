#pragma once

#include "Camera.h"
#include "Renderer.h"
#include "Particles.h"

class Scene
{
public:
	Scene(int w, int h);
	~Scene();

	void render();
	void rotate_cam(float ax, float ay);
	void change_cam_dist(int count);
	void update_particles();
	void clear();

private:
	Particles *p;
	Camera *cam;
	Renderer *renderer;
};

