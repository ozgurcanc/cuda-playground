#pragma once

#include "Vector3.h"


class Material
{
public:
	int phongExp = 0;		// Phong exponent
	Vector3 ambientRef;		// Coefficients for ambient reflection
	Vector3 diffuseRef;		// Coefficients for diffuse reflection
	Vector3 specularRef;	// Coefficients for specular reflection
	Vector3 mirrorRef;		// Coefficients for mirror reflection

private:
};
