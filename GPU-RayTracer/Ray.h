#pragma once

#include "Vector3.h"

class Ray
{
public:
	Vector3 origin;
	Vector3 direction;

	__device__ Ray();
	__device__ Ray(const Vector3 & origin, const Vector3 &direction);
	__device__ Vector3 GetPoint(float t) const;
};

__device__
Ray::Ray()
{
}

__device__
Ray::Ray(const Vector3 & origin, const Vector3 &direction)
:origin(origin), direction(direction)
{
}

__device__
Vector3 Ray::GetPoint(float t) const
{
	return origin + direction * t;
}

