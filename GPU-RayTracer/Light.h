#pragma once

#include "Material.h"

template <typename T>
__device__ T max(T a, T b)
{
	if (a > b) return a;
	
	return b;
}

class Light
{
public:
	Vector3 position;
	Vector3 intensity;

	__device__ virtual Vector3 ComputeLightContribution(Material material,Vector3 normal, Vector3 pos, Vector3 toEye) = 0;
};


class PointLight 
{
public:
	__host__ __device__ PointLight(const Vector3 & p, const Vector3& i);
	__device__ Vector3 ComputeLightContribution(Material material, Vector3 normal, Vector3 pos, Vector3 toEye);

	Vector3 position;
	Vector3 intensity;
};




__host__ __device__
PointLight::PointLight(const Vector3 & p, const Vector3& i)
{
	position = p;
	intensity = i;
}

__device__
Vector3 PointLight::ComputeLightContribution(Material material, Vector3 normal, Vector3 pos, Vector3 toEye)
{
	Vector3 contribution;

	Vector3 toLight = position - pos;
	real squareDistance = toLight.SquareMagnitude();
	toLight.Normalise();

	real diffuseFactor = normal * toLight;

	if (diffuseFactor > 0)
	{
		Vector3 half = (toEye + toLight).Unit();
		real specFactor = real_pow(max(real(0), normal * half), material.phongExp);

		contribution += material.diffuseRef * diffuseFactor;
		contribution += material.specularRef * specFactor;

		contribution *= intensity / squareDistance;
	}

	return contribution;
}