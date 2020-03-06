#pragma once

#include "Vector3.h"
#include <cstdio>
#include <cstdlib>

#include <string>
#include "lodepng.h"

typedef union Color
{
	struct
	{
		unsigned char red;
		unsigned char grn;
		unsigned char blu;
	};

	unsigned char channel[3];
} Color;


