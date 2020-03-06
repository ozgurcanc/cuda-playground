#pragma once

#include "Vector3.h"
#include "Ray.h"

#include <cstring>


typedef struct ImagePlane
{
	real left;     // "u" coordinate of the left edge
	real right;    // "u" coordinate of the right edge
	real bottom;   // "v" coordinate of the bottom edge
	real top;      // "v" coordinate of the top edge
	real distance; // distance to the camera (always positive)
	int nx;         // number of pixel columns
	int ny;         // number of pixel rows
} ImagePlane;


class Camera
{
public:
	char imageName[32];

	Camera(const char* imageName);
	void LookAt(const Vector3& pos, const Vector3& target, const Vector3& up); //input file da direk alýrsan g+p = t yapýp vermeyi unutma
	void Perspective(const real left, const real right, const real bottom, const real top, const real distance, const real nx, const real ny);
	void Perspective(const real fovy, const real aspectRatio, const real near, const real nx);
	__host__ __device__ void DeriveInternals();
	__device__ Ray GetPrimaryRay(int row, int col) const;
	__host__ __device__ Vector3 GetResolution();

private:
	ImagePlane imgPlane;
	
	Vector3 position;
	Vector3 gaze;
	Vector3 up;
	Vector3 right;

	Vector3 imgPlaneTopLeft;
	Vector3 horizantalStep;
	Vector3 verticalStep;
};


Camera::Camera(const char* imageName)
{
	std::strcpy(this->imageName, imageName);
}

void Camera::Perspective(const real left, const real right, const real bottom, 
		const real top, const real distance, const real nx, const real ny)
{
	imgPlane.left = left;
	imgPlane.right = right;
	imgPlane.bottom = bottom;
	imgPlane.top = top;
	imgPlane.distance = distance;
	imgPlane.nx = nx;
	imgPlane.ny = ny;
}


void Camera::Perspective(const real fovy, const real aspectRatio, const real distance, const real nx)
{
	real h = real_tan(fovy / real(2)) * distance;
	real w = h * aspectRatio;

	imgPlane.left = -w;
	imgPlane.right = w;
	imgPlane.bottom = -h;
	imgPlane.top = h;
	imgPlane.distance = distance;
	imgPlane.nx = nx;
	imgPlane.ny = nx / aspectRatio;
}


void Camera::LookAt(const Vector3& p, const Vector3& t, const Vector3& u)
{
	position = p;
	gaze = (t - p).Unit();
	right = (gaze % u).Unit();
	up = (right % gaze).Unit();
}

__host__ __device__
void Camera::DeriveInternals()
{
	imgPlaneTopLeft = position + gaze * imgPlane.distance + right * imgPlane.left + up * imgPlane.top;
	horizantalStep = right * ((imgPlane.right - imgPlane.left) / imgPlane.nx);
	verticalStep = up * ((imgPlane.top - imgPlane.bottom) / imgPlane.ny);
}

__device__
Ray Camera::GetPrimaryRay(int row, int col) const
{
	Vector3 pointOnPlane = imgPlaneTopLeft + horizantalStep * (row + 0.5) - verticalStep * (col + 0.5);
	Vector3 direction = (pointOnPlane - position).Unit();
	return Ray(position, direction);
}

__host__ __device__
Vector3 Camera::GetResolution()
{
	return Vector3(imgPlane.nx, imgPlane.ny, 0);
}
