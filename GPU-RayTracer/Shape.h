#pragma once


#include "Vector3.h"
#include "Material.h"
#include "Ray.h"

struct RayHit
{
	real t = REAL_MAX;
	
	Vector3 pos;
	Vector3	normal;
	Vector3 toEye;

	int matIndex;
};

class Shape
{
public:

	Shape(int matIndex);
	__device__ virtual void Intersect(const Ray& ray, const Vector3 *vertices, RayHit * rayHit)  = 0 ;

public:
	int matIndex;
};

class Sphere
{
public:
	Sphere(int matIndex, int centerIndex,real radius);
	__device__ void Intersect(const Ray& ray, const Vector3 *vertices, RayHit * rayHit);

public:
	int matIndex;
	int centerIndex;
	real radius;
};


class Triangle 
{
public:
	Triangle(int matIndex, int index1, int index2, int index3);
	__device__ void Intersect(const Ray& ray, const Vector3 *vertices, RayHit * rayHit);

public:
	int matIndex;
	int v1Index;
	int v2Index;
	int v3Index;
};



Shape::Shape(int matIndex)
:matIndex(matIndex)
{
}

Sphere::Sphere(int matIndex, int centerIndex, real radius)
	: matIndex(matIndex),radius(radius), centerIndex(centerIndex)
{
}

Triangle::Triangle(int matIndex, int v1Index, int v2Index, int v3Index)
	: matIndex(matIndex),v1Index(v1Index),v2Index(v2Index), v3Index(v3Index)
{
}


__device__
void Sphere::Intersect(const Ray& ray, const Vector3 *vertices, RayHit * rayHit)
{
	Vector3 center = vertices[centerIndex];
	Vector3 centerToOrigin = ray.origin - center;

	real a = ray.direction * ray.direction;
	real b = ray.direction * centerToOrigin;
	real c = centerToOrigin * centerToOrigin - radius * radius;

	real delta = b * b - a * c;

	if (delta > 0)
	{
		real sqrtDelta = real_sqrt(delta);
		real t1 = (-b - sqrtDelta) / a;
		real t2 = (-b + sqrtDelta) / a;

		real tmin, tmax;

		if (t1 < t2)
		{
			tmin = t1;
			tmax = t2;
		}
		else
		{
			tmin = t2;
			tmax = t1;
		}

		if (tmin > 0)
		{
			if (tmin < rayHit->t)
			{
				rayHit->t = tmin;
				rayHit->matIndex = matIndex;
				rayHit->pos = ray.GetPoint(tmin);
				rayHit->normal = (rayHit->pos - center).Unit();
				rayHit->toEye = ray.direction * -1; 
			}
			return;
		}

		if (tmax > 0)
		{
			if (tmax < rayHit->t)
			{
				rayHit->t = tmax;
				rayHit->matIndex = matIndex;
				rayHit->pos = ray.GetPoint(tmax);
				rayHit->normal = (rayHit->pos - center).Unit();
				rayHit->toEye = ray.direction * -1; 
			}
			return;
		}

		return;
	}
	else if (delta == 0)
	{
		real t = -b / a;

		if (t > 0 && t < rayHit->t)
		{
			rayHit->t = t;
			rayHit->matIndex = matIndex;
			rayHit->pos = ray.GetPoint(t);
			rayHit->normal = (rayHit->pos - center).Unit();
			rayHit->toEye = ray.direction * -1; 
		}
	}
	else return;

}

__device__
void Triangle::Intersect(const Ray& ray, const Vector3 *vertices, RayHit * rayHit)
{
	Vector3 e1 = vertices[v2Index] - vertices[v1Index];
	Vector3 e2 = vertices[v3Index] - vertices[v1Index];
	Vector3 m = ray.origin - vertices[v1Index];

	Vector3 d_e2 = (ray.direction % e2);
	real divider = e1 * d_e2;

	if (divider == 0) return;

	Vector3 m_e1 = m % e1;

	real t = (e2 * m_e1) / divider;

	if (t <= 0 || t >= rayHit->t) return;

	real v = (ray.direction * m_e1) / divider;
	
	if (v < 0 || v>1) return;

	real u = (m * d_e2) / divider;

	if (u < 0 || u>1 || u + v > 1) return;

	rayHit->t = t;
	rayHit->matIndex = matIndex;
	rayHit->pos = ray.GetPoint(t);
	rayHit->normal = (e2 % e1).Unit() * -1; 
	rayHit->toEye = ray.direction * -1;
}



