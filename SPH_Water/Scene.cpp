#include "stdafx.h"
#include "Scene.h"


Scene::Scene(int w, int h)
{
	this->cam = new Camera(1.6, 0, 1);
	this->p = new Particles();
	this->renderer = new Renderer(w, h, p, cam);
}


Scene::~Scene()
{
}

void Scene::render()
{
	this->renderer->render(this->p, this->cam);
}

void Scene::rotate_cam(float ax, float ay)
{
	this->cam->rotate(ax, ay);
}

void Scene::update_particles()
{
	this->p->update_particles();
}

void Scene::clear()
{
	this->renderer->clear();
}
