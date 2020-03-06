#pragma once

#include "headers.h"
#include <assert.h>

class Vector3
{

public:

	real x, y, z;

	
	__host__ __device__ Vector3();
	__host__ __device__ Vector3(const real _x, const real _y, const real _z);

	__host__ __device__ Vector3 operator+(const Vector3& v) const;
	__host__ __device__ void operator+=(const Vector3& v);
	__host__ __device__ Vector3 operator-(const Vector3& v) const;
	__host__ __device__ void operator-=(const Vector3& v);
	__host__ __device__ Vector3 operator*(const real k) const;
	__host__ __device__ real operator*(const Vector3& v) const;
	__host__ __device__ void operator*=(const Vector3& v);
	__host__ __device__ void operator*=(const real k);
	__host__ __device__ Vector3 operator%(const Vector3 & v) const;
	__host__ __device__ void operator%=(const Vector3 &v);
	__host__ __device__ Vector3 operator/(const real k) const;
	__host__ __device__ void operator/=(const real k);
	__host__ __device__ real Magnitude() const;
	__host__ __device__ real SquareMagnitude() const;
	__host__ __device__ void Normalise();
	__host__ __device__ Vector3 Unit() const;
	__host__ __device__ bool operator==(const Vector3& v) const;
	__host__ __device__ bool operator!=(const Vector3& v) const;
	__host__ __device__ void Invert();
	__host__ __device__ void AddScaledVector(const Vector3& v, real k);
	__host__ __device__ real DotProduct(const Vector3& v);
	__host__ __device__ Vector3 CrossProduct(const Vector3& v);
	__host__ __device__ Vector3 ComponentProduct(const Vector3& v);
	__host__ __device__ void Clear();
	__host__ __device__ real operator[](unsigned i) const;
	__host__ __device__ real& operator[](unsigned i);

	friend std::ostream& operator<<(std::ostream& os, const Vector3& v);
};

__host__ __device__
Vector3::Vector3() : x(0), y(0), z(0) {}

__host__ __device__
Vector3::Vector3(const real _x, const real _y, const real _z) : x(_x), y(_y), z(_z) {}

__host__ __device__
Vector3 Vector3::operator+(const Vector3& v) const
{
	return Vector3(x + v.x, y + v.y, z + v.z);
}

__host__ __device__
void Vector3::operator+=(const Vector3& v)
{
	x += v.x;
	y += v.y;
	z += v.z;
}

__host__ __device__
Vector3 Vector3::operator-(const Vector3& v) const
{
	return Vector3(x - v.x, y - v.y, z - v.z);
}

__host__ __device__
void Vector3::operator-=(const Vector3& v)
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
}

__host__ __device__
Vector3 Vector3::operator*(const real k) const
{
	return Vector3(x * k, y * k, z * k);
}

__host__ __device__
real Vector3::operator*(const Vector3& v) const
{
	return x * v.x + y * v.y + z * v.z;
}

__host__ __device__
void Vector3::operator*=(const Vector3& v)
{
	x *= v.x;
	y *= v.y;
	z *= v.z;
}

__host__ __device__
void Vector3::operator*=(const real k)
{
	x *= k;
	y *= k;
	z *= k;
}

__host__ __device__
Vector3 Vector3::operator%(const Vector3 & v) const
{
	return Vector3(y*v.z - z * v.y,
		z*v.x - x * v.z,
		x*v.y - y * v.x);
}

__host__ __device__
void Vector3::operator%=(const Vector3 &v)
{
	real _x = x;
	real _y = y;
	real _z = z;

	x = _y * v.z - _z * v.y;
	y = _z * v.x - _x * v.z;
	z = _x * v.y - _y * v.x;
}

__host__ __device__
Vector3 Vector3::operator/(const real k) const
{
	assert(k != 0);

	return Vector3(x / k, y / k, z / k);
}

__host__ __device__
void Vector3::operator/=(const real k)
{
	assert(k != 0);

	x /= k;
	y /= k;
	z /= k;
}

__host__ __device__
real Vector3::Magnitude() const
{
	return real_sqrt(x*x + y * y + z * z);
}

__host__ __device__
real Vector3::SquareMagnitude() const
{
	return x * x + y * y + z * z;
}

__host__ __device__
void Vector3::Normalise()
{
	real l = this->Magnitude();

	if (l > 0)
	{
		*this /= l;
	}
}

__host__ __device__
Vector3 Vector3::Unit() const
{
	Vector3 v = *this;
	v.Normalise();
	return v;
}

__host__ __device__
bool Vector3::operator==(const Vector3& v) const
{
	return x == v.x &&
		y == v.y &&
		z == v.z;
}

__host__ __device__
bool Vector3::operator!=(const Vector3& v) const
{
	return !(*this == v);
}

__host__ __device__
void Vector3::Invert()
{
	x *= -1;
	y *= -1;
	z *= -1;
}

__host__ __device__
void Vector3::AddScaledVector(const Vector3& v, real k)
{
	*this += v * k;
}

__host__ __device__
real Vector3::DotProduct(const Vector3& v)
{
	return (*this) * v;
}

__host__ __device__
Vector3 Vector3::CrossProduct(const Vector3& v)
{
	return (*this) % v;
}

__host__ __device__
Vector3 Vector3::ComponentProduct(const Vector3& v)
{
	return Vector3(x*v.x, y*v.y, z*v.z);
}

__host__ __device__
void Vector3::Clear()
{
	x = y = z = 0;
}

__host__ __device__
real Vector3::operator[](unsigned i) const
{
	if (i == 0) return x;
	if (i == 1) return y;
	return z;
}

__host__ __device__
real& Vector3::operator[](unsigned i)
{
	if (i == 0) return x;
	if (i == 1) return y;
	return z;
}

std::ostream& operator<<(std::ostream& os, const Vector3& v)
{
	os << v.x << '/' << v.y << '/' << v.z;
	return os;
}
