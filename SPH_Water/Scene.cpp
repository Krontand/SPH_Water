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

void Scene::change_cam_dist(int count)
{
	this->cam->scale_dist(pow(0.99, count));
}

void Scene::update_particles(float dt)
{
	cudaEvent_t start, stop;
	//float elapsedTime;

	//cudaEventCreate(&start);
	//cudaEventRecord(start, 0);

	float *map = this->renderer->map_resource();
	this->p->update_particles(map, dt);
	this->renderer->unmap_resource();

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	//cudaEventElapsedTime(&elapsedTime, start, stop);
	//LoggerWrite("dt = %f, Elapsed time : %f ms\n", dt, elapsedTime);


}

void Scene::clear()
{
	this->renderer->clear();
}
