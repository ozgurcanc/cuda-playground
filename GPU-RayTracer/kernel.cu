
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "tinyxml2.h"
#include "Camera.h"
#include "Light.h"
#include "Material.h"
#include "Ray.h"
#include "Shape.h"
#include "Image.h"
#include <math.h>

using namespace tinyxml2;



__global__ void GPURender
(
	Vector3 * vertices,
	PointLight* pointlights,
	int pointlightsSize,
	Material * materials,
	Sphere * spheres,
	int spheresSize,
	Triangle * trianles,
	int triangleSize,
	int maxRecursionDepth,
	real shadowRayEps,
	Vector3 backgroundColor,
	Vector3 ambientLight,
	Camera cam,
	Color * imgData
);

__device__
Color ConvertToColor(const Vector3 &v);

int maxRecursionDepth;				
real intTestEps;				
real shadowRayEps;					 
Vector3 backgroundColor;		
Vector3 ambientLight;				

std::vector<Camera *> cameras;		
std::vector<PointLight *> pointLights;		
std::vector<Material *> materials;	
std::vector<Vector3> vertices;		
std::vector<Shape *> objects;		
std::vector<Sphere *> spheres;
std::vector<Triangle*> triangles;

void Scene(const char *xmlPath);
void Test();
void RenderScene();
void SaveAsPNG(const char *imageName, Color* data, int width, int height);
void seeCudaDeviceCountAndProperties();

int main(int argc, char *argv[])
{
	if (argc != 2)
    {
        std::cout << "Please run the ray tracer as:" << std::endl
             << "\t./raytracer inputs/<input_file_name>" << std::endl;
        return 1;
    }
   	
	const char *xmlPath = argv[1];
	Scene(xmlPath);
	//Test();
	//seeCudaDeviceCountAndProperties
	RenderScene();
	
}


void Scene(const char *xmlPath)
{
	const char *str;
	XMLDocument xmlDoc;
	XMLError eResult;
	XMLElement *pElement;

	maxRecursionDepth = 1;
	shadowRayEps = 0.001;

	eResult = xmlDoc.LoadFile(xmlPath);

	XMLNode *pRoot = xmlDoc.FirstChild();

	pElement = pRoot->FirstChildElement("MaxRecursionDepth");
	if (pElement != nullptr)
		pElement->QueryIntText(&maxRecursionDepth);

	pElement = pRoot->FirstChildElement("BackgroundColor");
	str = pElement->GetText();
	sscanf(str, "%f %f %f", &backgroundColor.x, &backgroundColor.y, &backgroundColor.z);

	pElement = pRoot->FirstChildElement("ShadowRayEpsilon");
	if (pElement != nullptr)
		pElement->QueryFloatText(&shadowRayEps);

	pElement = pRoot->FirstChildElement("IntersectionTestEpsilon");
	if (pElement != nullptr)
		eResult = pElement->QueryFloatText(&intTestEps);

	// Parse cameras
	pElement = pRoot->FirstChildElement("Cameras");
	XMLElement *pCamera = pElement->FirstChildElement("Camera");
	XMLElement *camElement;
	while (pCamera != nullptr)
	{
		int id;
		char imageName[64];
		Vector3 pos, gaze, up;
		ImagePlane imgPlane;

		eResult = pCamera->QueryIntAttribute("id", &id);
		camElement = pCamera->FirstChildElement("Position");
		str = camElement->GetText();
		sscanf(str, "%f %f %f", &pos.x, &pos.y, &pos.z);
		camElement = pCamera->FirstChildElement("Gaze");
		str = camElement->GetText();
		sscanf(str, "%f %f %f", &gaze.x, &gaze.y, &gaze.z);
		camElement = pCamera->FirstChildElement("Up");
		str = camElement->GetText();
		sscanf(str, "%f %f %f", &up.x, &up.y, &up.z);
		camElement = pCamera->FirstChildElement("NearPlane");
		str = camElement->GetText();
		sscanf(str, "%f %f %f %f", &imgPlane.left, &imgPlane.right, &imgPlane.bottom, &imgPlane.top);
		camElement = pCamera->FirstChildElement("NearDistance");
		eResult = camElement->QueryFloatText(&imgPlane.distance);
		camElement = pCamera->FirstChildElement("ImageResolution");
		str = camElement->GetText();
		sscanf(str, "%d %d", &imgPlane.nx, &imgPlane.ny);
		camElement = pCamera->FirstChildElement("ImageName");
		str = camElement->GetText();
		strcpy(imageName, str);

		Camera * cam = new Camera(imageName);
		cam->LookAt(pos, pos + gaze, up);
		cam->Perspective(imgPlane.left, imgPlane.right, imgPlane.bottom, imgPlane.top, imgPlane.distance, imgPlane.nx, imgPlane.ny);
		cameras.push_back(cam);

		pCamera = pCamera->NextSiblingElement("Camera");
	}

	// Parse materals
	pElement = pRoot->FirstChildElement("Materials");
	XMLElement *pMaterial = pElement->FirstChildElement("Material");
	XMLElement *materialElement;
	while (pMaterial != nullptr)
	{
		materials.push_back(new Material());

		int curr = materials.size() - 1;

		//eResult = pMaterial->QueryIntAttribute("id", &materials[curr]->id);
		materialElement = pMaterial->FirstChildElement("AmbientReflectance");
		str = materialElement->GetText();
		sscanf(str, "%f %f %f", &materials[curr]->ambientRef.x, &materials[curr]->ambientRef.y, &materials[curr]->ambientRef.z);
		materialElement = pMaterial->FirstChildElement("DiffuseReflectance");
		str = materialElement->GetText();
		sscanf(str, "%f %f %f", &materials[curr]->diffuseRef.x, &materials[curr]->diffuseRef.y, &materials[curr]->diffuseRef.z);
		materialElement = pMaterial->FirstChildElement("SpecularReflectance");
		str = materialElement->GetText();
		sscanf(str, "%f %f %f", &materials[curr]->specularRef.x, &materials[curr]->specularRef.y, &materials[curr]->specularRef.z);
		materialElement = pMaterial->FirstChildElement("MirrorReflectance");
		if (materialElement != nullptr)
		{
			str = materialElement->GetText();
			sscanf(str, "%f %f %f", &materials[curr]->mirrorRef.x, &materials[curr]->mirrorRef.y, &materials[curr]->mirrorRef.z);
		}
		else
		{
			materials[curr]->mirrorRef.x = 0.0;
			materials[curr]->mirrorRef.y = 0.0;
			materials[curr]->mirrorRef.z = 0.0;
		}
		materialElement = pMaterial->FirstChildElement("PhongExponent");
		if (materialElement != nullptr)
			materialElement->QueryIntText(&materials[curr]->phongExp);

		pMaterial = pMaterial->NextSiblingElement("Material");
	}

	// Parse vertex data
	pElement = pRoot->FirstChildElement("VertexData");
	int cursor = 0;
	Vector3 tmpPoint;
	str = pElement->GetText();
	while (str[cursor] == ' ' || str[cursor] == '\t' || str[cursor] == '\n')
		cursor++;
	while (str[cursor] != '\0')
	{
		for (int cnt = 0; cnt < 3; cnt++)
		{
			if (cnt == 0)
				tmpPoint.x = atof(str + cursor);
			else if (cnt == 1)
				tmpPoint.y = atof(str + cursor);
			else
				tmpPoint.z = atof(str + cursor);
			while (str[cursor] != ' ' && str[cursor] != '\t' && str[cursor] != '\n')
				cursor++;
			while (str[cursor] == ' ' || str[cursor] == '\t' || str[cursor] == '\n')
				cursor++;
		}
		vertices.push_back(tmpPoint);
	}

	// Parse objects
	pElement = pRoot->FirstChildElement("Objects");

	// Parse spheres
	XMLElement *pObject = pElement->FirstChildElement("Sphere");
	XMLElement *objElement;
	while (pObject != nullptr)
	{
		int id;
		int matIndex;
		int cIndex;
		float R;

		eResult = pObject->QueryIntAttribute("id", &id);
		objElement = pObject->FirstChildElement("Material");
		eResult = objElement->QueryIntText(&matIndex);
		objElement = pObject->FirstChildElement("Center");
		eResult = objElement->QueryIntText(&cIndex);
		objElement = pObject->FirstChildElement("Radius");
		eResult = objElement->QueryFloatText(&R);

		spheres.push_back(new Sphere(matIndex - 1, cIndex - 1, R));

		pObject = pObject->NextSiblingElement("Sphere");
	}

	// Parse triangles
	pObject = pElement->FirstChildElement("Triangle");
	while (pObject != nullptr)
	{
		int id;
		int matIndex;
		int p1Index;
		int p2Index;
		int p3Index;

		eResult = pObject->QueryIntAttribute("id", &id);
		objElement = pObject->FirstChildElement("Material");
		eResult = objElement->QueryIntText(&matIndex);
		objElement = pObject->FirstChildElement("Indices");
		str = objElement->GetText();
		sscanf(str, "%d %d %d", &p1Index, &p2Index, &p3Index);

		triangles.push_back(new Triangle(matIndex - 1, p1Index - 1, p2Index - 1, p3Index - 1));

		pObject = pObject->NextSiblingElement("Triangle");
	}

	// Parse meshes
	pObject = pElement->FirstChildElement("Mesh");
	while (pObject != nullptr)
	{
		int id;
		int matIndex;
		int p1Index;
		int p2Index;
		int p3Index;
		int cursor = 0;
		int vertexOffset = 0;
		std::vector<Triangle> faces;
		std::vector<int> *meshIndices = new std::vector<int>();

		eResult = pObject->QueryIntAttribute("id", &id);
		objElement = pObject->FirstChildElement("Material");
		eResult = objElement->QueryIntText(&matIndex);
		objElement = pObject->FirstChildElement("Faces");
		objElement->QueryIntAttribute("vertexOffset", &vertexOffset);
		str = objElement->GetText();
		while (str[cursor] == ' ' || str[cursor] == '\t' || str[cursor] == '\n')
			cursor++;
		while (str[cursor] != '\0')
		{
			for (int cnt = 0; cnt < 3; cnt++)
			{
				if (cnt == 0)
					p1Index = atoi(str + cursor) + vertexOffset;
				else if (cnt == 1)
					p2Index = atoi(str + cursor) + vertexOffset;
				else
					p3Index = atoi(str + cursor) + vertexOffset;
				while (str[cursor] != ' ' && str[cursor] != '\t' && str[cursor] != '\n')
					cursor++;
				while (str[cursor] == ' ' || str[cursor] == '\t' || str[cursor] == '\n')
					cursor++;
			}
			//faces.push_back(*(new Triangle(matIndex-1, p1Index-1, p2Index-1, p3Index-1)));
			triangles.push_back((new Triangle(matIndex - 1, p1Index - 1, p2Index - 1, p3Index - 1)));
			meshIndices->push_back(p1Index);
			meshIndices->push_back(p2Index);
			meshIndices->push_back(p3Index);
		}

		//objects.push_back(new Mesh(matIndex-1, faces));

		pObject = pObject->NextSiblingElement("Mesh");
	}

	// Parse lights
	int id;
	Vector3 position;
	Vector3 intensity;
	pElement = pRoot->FirstChildElement("Lights");

	XMLElement *pLight = pElement->FirstChildElement("AmbientLight");
	XMLElement *lightElement;
	str = pLight->GetText();
	sscanf(str, "%f %f %f", &ambientLight.x, &ambientLight.y, &ambientLight.z);

	pLight = pElement->FirstChildElement("PointLight");
	while (pLight != nullptr)
	{
		eResult = pLight->QueryIntAttribute("id", &id);
		lightElement = pLight->FirstChildElement("Position");
		str = lightElement->GetText();
		sscanf(str, "%f %f %f", &position.x, &position.y, &position.z);
		lightElement = pLight->FirstChildElement("Intensity");
		str = lightElement->GetText();
		sscanf(str, "%f %f %f", &intensity.x, &intensity.y, &intensity.z);

		pointLights.push_back(new PointLight(position, intensity));

		pLight = pLight->NextSiblingElement("PointLight");
	}
}

void Test()
{
	std::cout << "maxRecursionDepth" << std::endl;
	std::cout << maxRecursionDepth << std::endl;
	std::cout << "intTestEps" << std::endl;
	std::cout << intTestEps << std::endl;
	std::cout << "shadowRayEps" << std::endl;
	std::cout << shadowRayEps << std::endl;
	std::cout << "backgroundColor" << std::endl;
	std::cout << backgroundColor << std::endl;
	std::cout << "ambientLight" << std::endl;
	std::cout << ambientLight << std::endl;

	std::cout << "cameras" << std::endl;
	std::cout << cameras.size() << std::endl;

	std::cout << "lights" << std::endl;
	std::cout << pointLights.size() << std::endl;

	std::cout << "materials" << std::endl;
	std::cout << materials.size() << std::endl;

	std::cout << "vertices" << std::endl;
	std::cout << vertices.size() << std::endl;

	std::cout << "objects" << std::endl;
	std::cout << objects.size() << std::endl;

}

void RenderScene()
{
	for (Camera * cam : cameras)
	{
		cam->DeriveInternals();

		// Vertices Device Copy 
		int verticesSize = vertices.size();
		Vector3 * host_vertices = (Vector3*)malloc(verticesSize * sizeof(Vector3));
		for (int i = 0; i < verticesSize; i++) host_vertices[i] = vertices[i];

		Vector3 * device_vertices;
		cudaMalloc((void **)&device_vertices, verticesSize * sizeof(Vector3));
		cudaMemcpy(device_vertices, host_vertices, verticesSize * sizeof(Vector3), cudaMemcpyHostToDevice);
		
		free(host_vertices);

		// Point light Device Copy
		int pointLSize = pointLights.size();
		PointLight * host_pointLights = (PointLight*)malloc(pointLSize * sizeof(PointLight));
		for (int i = 0; i < pointLSize; i++) host_pointLights[i] = *pointLights[i];

		PointLight * device_pointlights;
		cudaMalloc((void **)&device_pointlights, pointLSize * sizeof(PointLight));
		cudaMemcpy(device_pointlights, host_pointLights, pointLSize * sizeof(PointLight), cudaMemcpyHostToDevice);

		free(host_pointLights);

		// Materials Device Copy 
		int materialsSize = materials.size();
		Material * host_materials = (Material*)malloc(materialsSize * sizeof(Material));
		for (int i = 0; i < materialsSize; i++) host_materials[i] = *materials[i];

		Material* device_materials;
		cudaMalloc((void **)&device_materials, materialsSize * sizeof(Material));
		cudaMemcpy(device_materials, host_materials, materialsSize * sizeof(Material), cudaMemcpyHostToDevice);

		free(host_materials);

		// spheres Device Copy
		int spheresSize = spheres.size();
		Sphere * host_spheres = (Sphere*)malloc(spheresSize * sizeof(Sphere));
		for (int i = 0; i < spheresSize; i++) host_spheres[i] = *spheres[i];

		Sphere * device_spheres;
		cudaMalloc((void **)&device_spheres, spheresSize * sizeof(Sphere));
		cudaMemcpy(device_spheres, host_spheres, spheresSize * sizeof(Sphere), cudaMemcpyHostToDevice);

		free(host_spheres);

		//Triangles device copy
		int triangleSize = triangles.size();
		Triangle * host_triangles = (Triangle *)malloc(triangleSize * sizeof(Triangle));
		for (int i = 0; i < triangleSize; i++) host_triangles[i] = *triangles[i];

		Triangle * device_triangles;
		cudaMalloc((void **)&device_triangles, triangleSize * sizeof(Triangle));
		cudaMemcpy(device_triangles, host_triangles, triangleSize * sizeof(Triangle), cudaMemcpyHostToDevice);

		free(host_triangles);

		// Create Color*
		Vector3 resolution = cam->GetResolution();
		int imgSize = resolution.x * resolution.y;

		Color * host_imgData = (Color*)calloc(imgSize , sizeof(Color));
		Color * device_imgData;
		cudaMalloc((void **)&device_imgData, imgSize * sizeof(Color));
		

		// Init GPU Render
		GPURender<<<std::ceil(imgSize / 256.0),256>>>(device_vertices,device_pointlights, pointLSize,device_materials,device_spheres,spheresSize,
			device_triangles,triangleSize,maxRecursionDepth,shadowRayEps,backgroundColor,ambientLight,*cam,device_imgData);


		cudaMemcpy(host_imgData, device_imgData, imgSize * sizeof(Color), cudaMemcpyDeviceToHost);

		cudaFree(device_vertices);
		cudaFree(device_pointlights);
		cudaFree(device_materials);
		cudaFree(device_spheres);
		cudaFree(device_triangles);
		cudaFree(device_imgData);

		SaveAsPNG(cam->imageName, host_imgData, resolution.x, resolution.y);

		free(host_imgData);
	}
}



__global__ void GPURender
(
	Vector3 * vertices,
	PointLight* pointlights,
	int pointlightsSize,
	Material * materials,
	Sphere * spheres,
	int spheresSize,
	Triangle * triangles,
	int triangleSize,
	int maxRecursionDepth,
	real shadowRayEps,
	Vector3 backgroundColor,
	Vector3 ambientLight,
	Camera cam,
	Color * imgData
)
{
	int index = __cudaGet_threadIdx().x + blockDim.x * blockIdx.x;

	int width = cam.GetResolution().x;
	int height = cam.GetResolution().y;

	if (index >= width * height) return;

	int y = index / width;
	int x = index % width;

	Ray ray = cam.GetPrimaryRay(x, y);

	Vector3 pixelColor;
	Vector3 mirrorRef = Vector3(1, 1, 1);

	for (int r = 0; r < maxRecursionDepth; r++)
	{	
		RayHit * rayHit = new RayHit();

		for (int i = 0; i < spheresSize; i++)
			spheres[i].Intersect(ray, vertices, rayHit);


		for (int i = 0; i < triangleSize; i++)
			triangles[i].Intersect(ray, vertices, rayHit);


		if (rayHit->t == REAL_MAX)
		{
			delete rayHit;
			pixelColor+=backgroundColor;
			break;
		}

		Material material = materials[rayHit->matIndex];

		pixelColor += ambientLight.ComponentProduct(material.ambientRef);

		for (int i = 0; i < pointlightsSize; i++)
		{
			PointLight light = pointlights[i];
			Vector3 shadowRayDirection = (light.position - rayHit->pos);
			Vector3 shadowRayDirNorm = shadowRayDirection.Unit();
			Ray shadowRay(rayHit->pos + shadowRayDirNorm * shadowRayEps, shadowRayDirection);

			bool isInShadow = false;
			RayHit *shadowHit = new RayHit();

			for (int i = 0; i < spheresSize; i++)
			{
				spheres[i].Intersect(shadowRay, vertices, shadowHit);
				if (shadowHit->t < real(1))
				{
					isInShadow = true;
					break;
				}
			}

			if (!isInShadow)
				for (int i = 0; i < triangleSize; i++)
				{
					triangles[i].Intersect(shadowRay, vertices, shadowHit);
					if (shadowHit->t < real(1))
					{
						isInShadow = true;
						break;
					}
				}

			delete shadowHit;

			if (isInShadow) continue;

			pixelColor += light.ComputeLightContribution(material, rayHit->normal, rayHit->pos, rayHit->toEye);
		}


		if (material.mirrorRef == Vector3())
		{
			delete rayHit;
			break;
		}

		Vector3 refDirection = ray.direction - rayHit->normal * 2 * (ray.direction * rayHit->normal);
		refDirection.Normalise();

		Ray reflectionRay(rayHit->pos + refDirection * shadowRayEps, refDirection);

		delete rayHit;

		pixelColor *= mirrorRef;

		mirrorRef = material.mirrorRef;

		ray = Ray(rayHit->pos + refDirection * shadowRayEps, refDirection);
	}
	
	imgData[y*width + x] = ConvertToColor(pixelColor);

}

__device__
Color ConvertToColor(const Vector3 &v)
{
	Color c;
	c.red = v.x > 255 ? 255 : v.x;
	c.grn = v.y > 255 ? 255 : v.y;
	c.blu = v.z > 255 ? 255 : v.z;

	return c;
}

void SaveAsPNG(const char *imageName,Color* data, int width, int height) 
{
	std::vector<unsigned char> image;

	unsigned char alpha = 255;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			image.push_back(data[y * width + x].red);
			image.push_back(data[y * width + x].grn);
			image.push_back(data[y * width + x].blu);
			image.push_back(alpha);
		}
	}

	std::vector<unsigned char> png;

	unsigned error = lodepng::encode(png, image, width, height);
	if (!error) lodepng::save_file(png, imageName);
}


void seeCudaDeviceCountAndProperties()
{
	int avaliableDev;
	cudaGetDeviceCount(&avaliableDev);
	printf("Device Count : %d \n", avaliableDev);

	printf("\n");

	cudaDeviceProp dev_prop;

	for (int i = 0; i < avaliableDev; i++)
	{
		cudaGetDeviceProperties(&dev_prop, i);

		printf("maxThreadsPerBlock : %d \n", dev_prop.maxThreadsPerBlock);
		printf("multiProcessorCount : %d \n", dev_prop.multiProcessorCount);
		printf("clockRate : %d \n", dev_prop.clockRate);
		printf("maxThreadsDim[0] : %d \n", dev_prop.maxThreadsDim[0]);
		printf("maxThreadsDim[1] : %d \n", dev_prop.maxThreadsDim[1]);
		printf("maxThreadsDim[2] : %d \n", dev_prop.maxThreadsDim[2]);
		printf("maxGridSize[0] : %d \n", dev_prop.maxGridSize[0]);
		printf("maxGridSize[1] : %d \n", dev_prop.maxGridSize[1]);
		printf("maxGridSize[2] : %d \n", dev_prop.maxGridSize[2]);
		printf("warpSize : %d \n", dev_prop.warpSize);
		printf("\n");
	}
}
