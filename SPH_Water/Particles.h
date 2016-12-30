#pragma once
class Particles
{
public:
	Particles();
	~Particles();

	void update_particles();

	// ���������� ������ � ����� ���������, � ��� ������� �����������
	int MESH_VERTEX_COUNT = 125000;

	// ���������� ������ ��� ������ ������������, ����� 3 �������
	float *triangleMesh;

private:
	int xcount;
	int ycount;
	int zcount;
	float scale;
};

