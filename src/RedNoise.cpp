#include <CanvasTriangle.h>
#include <DrawingWindow.h>
#include <Utils.h>
#include <ModelTriangle.h>
#include <Colour.h>
#include <glm/glm.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <RayTriangleIntersection.h>
#include <TextureMap.h>

//------------------------------------------------------------------------------
// CORE DEFINITIONS AND CONSTANTS
//------------------------------------------------------------------------------

// Display dimensions and pi
#define WIDTH 640
#define HEIGHT 480
#define PI 3.14159265359

// Viewing parameters
#define FOV 45.0f  // Field of View i.e the viewing angle (in degrees) defualts to human 45 deg
#define NEAR 0.1f  // The closest visible distances from the camera
#define FAR 100.0f // The furthest visible distances from the camera

//------------------------------------------------------------------------------
// RENDERING MODES
//------------------------------------------------------------------------------
enum RenderMode {
    // Drawing modes
    WIRE,      // Basic wireframe rendering
    // Rasterising modes
    FLAT,      // Solid color faces
    DEPTH,     // Handles occlusion
    DIFFUSE,   // Diffuse + ambient lighting
    SPECULAR,  // Specular + diffuse + ambient lighting
    TEXTURE,   // Texture wrapped model
    GOURAUD,   // Gouraud shading
    PHONG,     // Phong shading 
    
    // Ray tracing modes
    SHADOW,    // Hard shadows
    SOFT,      // Soft shadows 
};

//------------------------------------------------------------------------------
// DATA STRUCTURES
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// View and Transform Structures
//------------------------------------------------------------------------------

// Controls view interaction
struct ModelState {
    float rotationX = 25.21f * PI / 180.0f;                           // Initial view tilt
    float rotationY = (29.79f + 180.0f) * PI / 180.0f;                // Initial view rotation
    bool isDragging = false;                                          // Mouse drag state
    int mouseX = 0;                             
    int mouseY = 0;                             
    RenderMode renderMode = WIRE;             
};

// Defines model space
struct ModelBounds {
    glm::vec3 min = glm::vec3(std::numeric_limits<float>::max());     // Minimum corner
    glm::vec3 max = glm::vec3(std::numeric_limits<float>::lowest());  // Maximum corner
    glm::vec3 center = glm::vec3(0.0f);                               
    float diagonal = 0.0f;                                            // Bounding box diagonal
    float scale = 1.0f;                                               // HEIGHT/WIDTH
};

// Camera properties
struct Camera {
   float x = 0.0f;             
   float y = 0.0f;  
   float z = -1.0f;   
   float moveSpeed = 0.05f;        
   float rotationX = 0.0f;    
   float rotationY = 0.0f;      
   float rotationSpeed = 0.05f;     
};

//------------------------------------------------------------------------------
// Core Structures
//------------------------------------------------------------------------------

// Surface properties
struct Material {
    std::string name;
    Colour diffuse;                 // Base surface color
    float specularStrength = 0.5f; 
    float shininess = 32.0f;       
    TextureMap texture;            
    bool hasTexture = false;   
};

// Illumination properties
struct Light {
    glm::vec3 position;
    float intensity;  
    float attenuation;              // Distance falloff factor
    
    Light(const glm::vec3& pos, float i) : position(pos), intensity(i), attenuation(0.1f) {}
};

// Helper struct for full lighting
struct LightingComponents {
    static constexpr float ambient = 0.2f;
    static constexpr float diffuse = 0.6f;
    static constexpr float specular = 1.2f;
    static constexpr float shininess = 64.0f;
    
    static float calculate(const glm::vec3& pos, const glm::vec3& norm, const Light& light, const glm::vec3& viewPos) {
        // Calculates the normalized directions
        glm::vec3 lightDir = glm::normalize(light.position - pos);
        glm::vec3 viewDir = glm::normalize(viewPos - pos);
        glm::vec3 halfDir = glm::normalize(lightDir + viewDir);
        
        // Calculates the lighting components
        float diff = std::max(0.0f, glm::dot(norm, lightDir));
        float spec = std::pow(std::max(0.0f, glm::dot(norm, halfDir)), shininess/2.0f);
        
        // Calculates the attenuation
        float attenuation = 1.0f / (1.0f + 0.03f * glm::length(light.position - pos));
        
        // Combines the components with gamma correction
        return std::pow((ambient + (diff + spec * specular) * attenuation) * light.intensity, 1.0f/2.2f);
    }
};

struct BarycentricCoords {
    float u, v, w;
    bool isInside;
};

// Triangle vertex data
struct VertexData {
    CanvasPoint screenPositions[3];         
    glm::vec3 normals[3];                
    glm::vec3 worldPositions[3];            
    float depths[3];                        
    
    // Method to sort vertices by their Y coordinate
    void sortVerticesByY() {
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2-i; j++) {
                if(screenPositions[j+1].y < screenPositions[j].y) {
                    std::swap(screenPositions[j], screenPositions[j+1]);
                    std::swap(normals[j], normals[j+1]);
                    std::swap(worldPositions[j], worldPositions[j+1]);
                    std::swap(depths[j], depths[j+1]);
                }
            }
        }
    }
};

// Scanline intersections
struct ScanlineIntersection {
    float x1, x2;                // Coordinates where scanline intersects with triangle edges
    glm::vec3 normal1, normal2;  // Interpolated normals at intersections
    glm::vec3 pos1, pos2;        // Interpolated world positions at intersections
    float depth1, depth2;        // Interpolated depth values at intersections
};

// Per-vertex lighting
struct VertexLight {
    float red;
    float green;
    float blue;
    
    VertexLight() : red(0), green(0), blue(0) {}
    VertexLight(float r, float g, float b) : red(r), green(g), blue(b) {}
};

// Normal map for surface detail
struct BumpMap {
    std::vector<glm::vec3> normals;
    int width;
    int height;
};

//------------------------------------------------------------------------------
// Animation Structure
//------------------------------------------------------------------------------

// Animation controls
struct AnimationState {
    float startTime;
    float duration;
    bool isAnimating;
    bool isComplexAnim;   
    float theta;                // The angular parameter
    float verticalOffset;       // Vertical oscillation
    glm::vec3 initialCameraPos;
    glm::vec3 initialLightPos;
    RenderMode previousRenderMode;   

    // Complex animation parameters
    float cameraRadius;
    float cameraHeight;
    float spiralTightness;      // Spiral density
    float objectRotation;   
    
    AnimationState() : 
        startTime(0),
        duration(15000),
        isAnimating(false),
        isComplexAnim(false),
        theta(0),
        verticalOffset(0),
        initialCameraPos(0, 0, -2.0f),
        initialLightPos(-1.0f, 1.5f, -1.5f),
        cameraRadius(3.0f),
        cameraHeight(2.0f),
        spiralTightness(0.5f),
        objectRotation(0)
    {}
};

struct PhysicsState {
    glm::vec3 position = glm::vec3(0.0f, 5.0f, 0.0f); 
    glm::vec3 velocity = glm::vec3(0.0f);
    glm::vec3 acceleration = glm::vec3(0.0f, -9.81f, 0.0f);  // Gravity
    float dampening = 0.28f;  // Energy loss on bounce
    bool hasLanded = false;
    float landingY = 0.0f; 
    float timeSinceLanding = 0.0f;
};

//------------------------------------------------------------------------------
// GLOBAL STATE
//------------------------------------------------------------------------------

ModelState modelState;                          
Camera camera;
std::vector<Material> materials;                  // Available materials        
int currentMaterialIndex = -1;                    // Selected material              

// Geometry
std::vector<ModelTriangle> triangles;       
std::vector<ModelTriangle> originalTriangles;   
std::vector<glm::vec3> vertices;              
std::vector<glm::vec3> originalVertices;      
std::vector<glm::vec3> transformedVertices;  
std::vector<CanvasTriangle> canvasTriangles;      // Screen space triangles
std::vector<float> depthBuffer;                   // Z-buffer

// Illumination 
Light light(glm::vec3(-1.0f, 1.5f, -1.5f), 1.0f); // Main light source              

// Normal mapping
std::vector<glm::vec3> vertexNormals;             // Per-vertex normals

// Animation
AnimationState animState;
PhysicsState physicsState;

//------------------------------------------------------------------------------
// BASIC GEOMETRY
//------------------------------------------------------------------------------

glm::vec3 rotateY(const glm::vec3& v, float rotY) {
    float cosY = cos(rotY), sinY = sin(rotY);
    return glm::vec3(
        v.x * cosY - v.z * sinY,
        v.y,
        v.x * sinY + v.z * cosY
    );
}

glm::vec3 rotateX(const glm::vec3& v, float rotX) {
    float cosX = cos(rotX), sinX = sin(rotX);
    return glm::vec3(
        v.x,
        v.y * cosX + v.z * sinX,
        fmax(NEAR, -v.y * sinX + v.z * cosX) // Prevents z getting too close to camera 
    );
}

void rotateVertex(glm::vec3& v, float angleX, float angleY) {
    // Y-axis rotation
    float cosY = cos(angleY), sinY = sin(angleY);
    float newX = v.x * cosY - v.z * sinY;
    float newZ = v.x * sinY + v.z * cosY;
    
    // X-axis rotation
    float cosX = cos(angleX), sinX = sin(angleX);
    float newY = v.y * cosX + newZ * sinX;
    newZ = -v.y * sinX + newZ * cosX;
    
    v = glm::vec3(newX, newY, newZ + (v.z > 0 ? 0.0001f : -0.0001f)); // Prevents z fighting
}

//------------------------------------------------------------------------------
// PROJECTION
//------------------------------------------------------------------------------

CanvasPoint projectVertex(const glm::vec3& v) {
    // Calculates the projection scale based on FOV
    float fovRadians = FOV * PI / 180.0f;
    float aspectRatio = static_cast<float>(WIDTH) / HEIGHT;
    float projScale = 1.0f / tan(fovRadians * 0.5f);
    
    // World to camera transformation 
    glm::vec3 view(v.x - camera.x, v.y - camera.y, v.z - camera.z);
    
    // Camera to screen transformation
    glm::vec3 rotated = rotateX(rotateY(view, camera.rotationY), camera.rotationX);
    
    if (rotated.z <= NEAR || rotated.z >= FAR) {
        return CanvasPoint(0, 0, 0.0f);
    }

    float invDepth = NEAR / rotated.z;
    
    // Perspective projection with aspect ratio correction
    return CanvasPoint(
        WIDTH * 0.5f + (rotated.x * projScale / rotated.z) * WIDTH * 0.4f * (1.0f / aspectRatio),
        HEIGHT * 0.5f - (rotated.y * projScale / rotated.z) * HEIGHT * 0.4f,
        invDepth
    );
}

//------------------------------------------------------------------------------
// MODEL MANAGEMENT
//------------------------------------------------------------------------------

void updateModelRotation() {
    // Restores the original triangles
    triangles = originalTriangles;
    
    // Calculates the bounding box
    glm::vec3 minBounds(std::numeric_limits<float>::max());
    glm::vec3 maxBounds(std::numeric_limits<float>::lowest());
    
    for (const auto& tri : triangles) {
        for (const auto& v : tri.vertices) {
            minBounds = glm::min(minBounds, v);
            maxBounds = glm::max(maxBounds, v);
        }
    }
    
    // Normalises the models size 
    glm::vec3 center = (minBounds + maxBounds) * 0.5f;
    float diagonal = glm::length(maxBounds - minBounds);
    float normalizeScale = 1.0f / diagonal;
    
    // Transforms each vertex in every triangle
    for (auto& tri : triangles) {
        for (auto& v : tri.vertices) {
            // First center and scale the vertex
            v = (v - center) * normalizeScale;

            rotateVertex(v, modelState.rotationX, modelState.rotationY);
        }
    }
}

void updateCanvasTriangles() {
    canvasTriangles.clear();
    for (const auto &tri : triangles) {
        CanvasPoint p0 = projectVertex(tri.vertices[0]);
        CanvasPoint p1 = projectVertex(tri.vertices[1]);
        CanvasPoint p2 = projectVertex(tri.vertices[2]);
        canvasTriangles.emplace_back(p0, p1, p2);
    }
}

//------------------------------------------------------------------------------
// MODEL LOADING
//------------------------------------------------------------------------------

void loadMTLFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Failed to open MTL file: " << filename << std::endl;
        return;
    }

    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    
    std::string token, texturePath;
    iss >> token >> texturePath;
    
    if (token == "map_Kd") {
        Material material;
        material.name = "textured";
        material.diffuse = Colour(255, 255, 255);  // Default white
        try {
            material.texture = TextureMap(texturePath);
            material.hasTexture = true;
        } catch (const std::runtime_error& e) {
            material.hasTexture = false;
        }
        materials.push_back(material);
        currentMaterialIndex = 0;
    }
}

// Also handles fiting the model to screen space and getting materials
void loadOBJFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Failed to open OBJ file: " << filename << std::endl;
        return;
    }

    std::vector<glm::vec3> vertices;
    std::vector<TexturePoint> textureCoords;
    triangles.clear();

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "mtllib") {
            std::string mtlFile;
            iss >> mtlFile;
            loadMTLFile(mtlFile);
        }
        else if (type == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            vertices.emplace_back(x, y, z);
        }
        else if (type == "vt") {
            float u, v;
            iss >> u >> v;
            textureCoords.emplace_back(u, v); 
        }
        else if (type == "f") {
            std::array<int, 3> vertexIndices;
            std::array<int, 3> textureIndices;
            
            for (int i = 0; i < 3; i++) {
                std::string vertexData;
                iss >> vertexData;
                
                // Split the vertex data into the vertex/texture indices
                size_t slashPos = vertexData.find('/');
                if (slashPos != std::string::npos) {
                    vertexIndices[i] = std::stoi(vertexData.substr(0, slashPos)) - 1;
                    textureIndices[i] = std::stoi(vertexData.substr(slashPos + 1)) - 1;
                } else {
                    vertexIndices[i] = std::stoi(vertexData) - 1;
                    textureIndices[i] = -1;  // No texture coordinate
                }
            }

            ModelTriangle triangle(
                vertices[vertexIndices[0]],
                vertices[vertexIndices[1]],
                vertices[vertexIndices[2]],
                materials[currentMaterialIndex].diffuse
            );

            // Assigns the texture coordinates if they are available
            if (!textureCoords.empty()) {
                for (int i = 0; i < 3; i++) {
                    if (textureIndices[i] >= 0 && textureIndices[i] < textureCoords.size()) {
                        triangle.texturePoints[i] = textureCoords[textureIndices[i]];
                    }
                }
            }

            // Calculates the normal for the triangle
            glm::vec3 edge1 = triangle.vertices[1] - triangle.vertices[0];
            glm::vec3 edge2 = triangle.vertices[2] - triangle.vertices[0];
            triangle.normal = glm::normalize(glm::cross(edge1, edge2));

            triangles.push_back(triangle);
        }
    }

    // Stores the original triangles for rotation updates
    originalTriangles = triangles;

    // Updates the model state
    updateModelRotation();
    updateCanvasTriangles();
}

//------------------------------------------------------------------------------
// DEPTH BUFFER MANAGEMENT
//------------------------------------------------------------------------------

void initializeDepthBuffer(DrawingWindow &window) {
    depthBuffer.resize(window.width * window.height, 0.0f);  // Initializes with the minimum depth
}

void clearDepthBuffer() {
    std::fill(depthBuffer.begin(), depthBuffer.end(), 0.0f);  // Clears to the minimum depth
}

//------------------------------------------------------------------------------
// DRAWING UTILITIES 
//------------------------------------------------------------------------------

// Linear interpolation between two values
float interpolate(float x0, float x1, float t) {
    return x0 + (x1 - x0) * t;
}

// Linear interpolation between two 3D vectors 
glm::vec3 interpolateVec3(const glm::vec3& a, const glm::vec3& b, float t) {
    return a + (b - a) * t;
}

// Packs the RGB color (+ lighting)
uint32_t calculatePixelColor(const Colour& baseColor, float lighting) {
    return (255 << 24) + 
           (static_cast<uint8_t>(std::min(255.0f, baseColor.red * lighting)) << 16) +
           (static_cast<uint8_t>(std::min(255.0f, baseColor.green * lighting)) << 8) +
           static_cast<uint8_t>(std::min(255.0f, baseColor.blue * lighting));
}

//------------------------------------------------------------------------------
// RASTERISING UTILITIES 
//------------------------------------------------------------------------------

// Clamps a value between bounds
int clampToBounds(float value, int min, int max) {
    return std::min(max, std::max(min, static_cast<int>(value)));
}

// Linearly interpolates a depth value between two vertices of a triangle
float interpolateDepth(const float z[3], int index1, int index2, float t) {
    return z[index1] + (z[index2] - z[index1]) * t;
}

// Calculates the normalized position factor between two y coordinates 
float interpolationFactor(float y0, float y1, float y, float eps) {
    return (y1 - y0) < eps ? 0 : (y - y0) / (y1 - y0);
}

// Interpolates between three color values using barycentric coordinates, with gamma correction
uint8_t interpolateColor(float c0, float c1, float c2, const BarycentricCoords& bary) {
    // Convert to [0,1] range and apply gamma correction in one step
    float v0 = std::pow(c0/255.0f,  2.2f);
    float v1 = std::pow(c1/255.0f,  2.2f);
    float v2 = std::pow(c2/255.0f,  2.2f);
    
    // Interpolate in linear space
    float intensity = bary.u * v0 + bary.v * v1 + bary.w * v2;
    
    // Convert back to [0,255] range with inverse gamma
    return static_cast<uint8_t>(255.0f * std::pow(intensity, 1.0f / 2.2f));
}

// Calculates the intersection points where a horizontal scanline crosses a triangle's edges
void calculateEdgeIntersections(const CanvasPoint v[3], int y, float eps, 
                         float& yp, float& sp, float& x1, float& x2) {
    yp = interpolationFactor(v[0].y, v[2].y, y, eps);
    
    if (y < v[1].y) {
        sp = interpolationFactor(v[0].y, v[1].y, y, eps);
        x1 = interpolate(v[0].x, v[2].x, yp);
        x2 = interpolate(v[0].x, v[1].x, sp);
    } else {
        sp = interpolationFactor(v[1].y, v[2].y, y, eps);
        x1 = interpolate(v[0].x, v[2].x, yp);
        x2 = interpolate(v[1].x, v[2].x, sp);
    }
}

glm::vec3 calculateVertexNormal(const glm::vec3& vertex) {
    // Calculates the geometric normal by normalizing vertex position
    glm::vec3 normal = glm::normalize(vertex);
    
    // Tracks the face normals of triangles sharing this vertex
    glm::vec3 faceAveragedNormal(0.0f);
    int sharedFaces = 0;
    
     // Finds all the triangles containing this vertex
    for (const auto& tri : triangles) {
        for (const auto& v : tri.vertices) {
            if (glm::length(v - vertex) < 0.0001f) {
                // Calculates the face normal using cross product of edges
                glm::vec3 edge1 = tri.vertices[1] - tri.vertices[0];
                glm::vec3 edge2 = tri.vertices[2] - tri.vertices[0];
                glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));
                faceAveragedNormal += faceNormal;
                sharedFaces++;
                break;
            }
        }
    }
    
    if (sharedFaces > 0) {
        faceAveragedNormal = glm::normalize(faceAveragedNormal);
        // Blends between the geometric normal and the face-averaged normal
        normal = glm::normalize(normal * 0.7f + faceAveragedNormal * 0.3f);
    }
    
    return normal;
}

// Helper function to set up vertex data from the triangle inputs
VertexData setupVertexData(const CanvasTriangle& canvasTri, const ModelTriangle& modelTri) {
    VertexData data;
    
    for(int i = 0; i < 3; i++) {
        data.screenPositions[i] = canvasTri[i];
        data.normals[i] = calculateVertexNormal(modelTri.vertices[i]);
        data.worldPositions[i] = modelTri.vertices[i];
        data.depths[i] = canvasTri[i].depth;
    }
    
    data.sortVerticesByY();
    return data;
}

bool isTriangleVisible(const VertexData& data, int windowWidth, int windowHeight) {
    const auto& v = data.screenPositions;
    
    return !(v[0].y >= windowHeight || v[2].y < 0 || 
             (v[0].x < 0 && v[1].x < 0 && v[2].x < 0) || 
             (v[0].x >= windowWidth && v[1].x >= windowWidth && v[2].x >= windowWidth));
}

// Calculates the intersections for a scanline
ScanlineIntersection calculateScanlineIntersections(
    const VertexData& data, int y, bool isTopHalf, float eps = 0.0001f) 
{
    ScanlineIntersection result;
    
    // Calculates the interpolation factors
    float yp = (data.screenPositions[2].y - data.screenPositions[0].y) < eps ? 0 : 
        (y - data.screenPositions[0].y) / (data.screenPositions[2].y - data.screenPositions[0].y);
    
    int start = isTopHalf ? 0 : 1;
    int end = isTopHalf ? 1 : 2;
    float sp = (data.screenPositions[end].y - data.screenPositions[start].y) < eps ? 0 :
        (y - data.screenPositions[start].y) / (data.screenPositions[end].y - data.screenPositions[start].y);
    
    // Interpolates all of our attributes
    result.x1 = interpolate(data.screenPositions[0].x, data.screenPositions[2].x, yp);
    result.x2 = interpolate(data.screenPositions[start].x, data.screenPositions[end].x, sp);
    
    result.normal1 = glm::normalize(interpolateVec3(data.normals[0], data.normals[2], yp));
    result.normal2 = glm::normalize(interpolateVec3(data.normals[start], data.normals[end], sp));
    
    result.pos1 = interpolateVec3(data.worldPositions[0], data.worldPositions[2], yp);
    result.pos2 = interpolateVec3(data.worldPositions[start], data.worldPositions[end], sp);
    
    result.depth1 = interpolate(data.depths[0], data.depths[2], yp);
    result.depth2 = interpolate(data.depths[start], data.depths[end], sp);
    
    // Ensure x1 is to the left of x2
    if (result.x1 > result.x2) {
        std::swap(result.x1, result.x2);
        std::swap(result.normal1, result.normal2);
        std::swap(result.pos1, result.pos2);
        std::swap(result.depth1, result.depth2);
    }
    
    return result;
}

//------------------------------------------------------------------------------
// RAYTRACING UTILITIES
//------------------------------------------------------------------------------

// Finds the closest valid intersection between a ray and a triangle
RayTriangleIntersection getClosestValidIntersection(
    const glm::vec3& rayOrigin,
    const glm::vec3& rayDirection,
    const std::vector<ModelTriangle>& triangles,
    const ModelTriangle* excludeTriangle = nullptr) 
{
    RayTriangleIntersection closestIntersection;
    closestIntersection.distanceFromCamera = std::numeric_limits<float>::infinity();
    
    for (size_t i = 0; i < triangles.size(); i++) {
        const ModelTriangle& triangle = triangles[i];
        
        // Skips the triangle we're starting from
        if (excludeTriangle && excludeTriangle == &triangles[i]) continue;
        
        // Calculates the edges from vertex 0
        glm::vec3 e0 = triangle.vertices[1] - triangle.vertices[0];
        glm::vec3 e1 = triangle.vertices[2] - triangle.vertices[0];
        glm::vec3 SPVector = rayOrigin - triangle.vertices[0];

        // Creates a matrix to solve the intersection equation
        glm::mat3 DEMatrix(-rayDirection, e0, e1);
        
        // Skips if the matrix isnt invertible
        float det = glm::determinant(DEMatrix);
        if (std::abs(det) < 0.0001f) continue;
        
        // Create the matrix to solve the intersection equation
        glm::vec3 possibleSolution = glm::inverse(DEMatrix) * SPVector;
        
        // Extracts values from solution
        float t = possibleSolution.x;
        float u = possibleSolution.y;
        float v = possibleSolution.z;
        
         // Checks if the intersection is valid:
        if (t >= 0 && u >= 0 && u <= 1 && v >= 0 && v <= 1 && (u + v) <= 1) {
            if (t < closestIntersection.distanceFromCamera) {
                glm::vec3 intersectionPoint = rayOrigin + t * rayDirection;
                
                closestIntersection.intersectionPoint = intersectionPoint;
                closestIntersection.distanceFromCamera = t;
                closestIntersection.triangleIndex = i;
                closestIntersection.intersectedTriangle = triangle;
            }
        }
    }
    
    return closestIntersection;
}

bool isPointInShadow(
    const glm::vec3& point,
    const ModelTriangle* sourceTriangle,
    const std::vector<ModelTriangle>& triangles)
{
    glm::vec3 directionToLight = glm::normalize(light.position - point);
    float distanceToLight = glm::length(light.position - point);
    
    const float shadow_bias = 0.005f; 

    // Offsets the origin point slightly to prevent self-shadowing
    glm::vec3 offsetPoint = point + directionToLight * shadow_bias;
    
    // Checks for intersections between point and light
    RayTriangleIntersection shadowRay = getClosestValidIntersection(
        offsetPoint,
        directionToLight,
        triangles,
        sourceTriangle
    );
    
    // If we found an intersection that's closer than the light then the point is in shadow
    return shadowRay.distanceFromCamera < distanceToLight;
}

//------------------------------------------------------------------------------
// WIREFRAME DRAWING
//------------------------------------------------------------------------------

// Draws lines by inteprolating between points on the screen
void drawLine(DrawingWindow &window, const CanvasPoint& from, const CanvasPoint& to, const Colour& colour) {
    uint32_t color = calculatePixelColor(colour, 1.0f);
    float steps = std::max(abs(to.x - from.x), abs(to.y - from.y));
    
    for(float i = 0; i <= steps; i++) {
        float t = i/steps;
        float x = interpolate(from.x, to.x, t);
        float y = interpolate(from.y, to.y, t);
        
        int px = clampToBounds(round(x), 0, window.width - 1);
        int py = clampToBounds(round(y), 0, window.height - 1);
        window.setPixelColour(px, py, color);
    }
}

void drawTriangleWireframe(DrawingWindow& window, const CanvasTriangle& canvas, const ModelTriangle& model) {
   drawLine(window, canvas[0], canvas[1], model.colour);
   drawLine(window, canvas[1], canvas[2], model.colour);
   drawLine(window, canvas[2], canvas[0], model.colour);
}

//------------------------------------------------------------------------------
// 2D RASTER
//------------------------------------------------------------------------------

void drawBottomFlatTriangle(DrawingWindow &window, 
   const CanvasPoint &topVertex, 
   const CanvasPoint &bottomLeft, 
   const CanvasPoint &bottomRight, 
   const Colour &color, 
   float edgeOffset = 0.001f) 
{
   // Pack RGB colour
   uint32_t packedColor = (255 << 24) + (color.red << 16) + 
                         (color.green << 8) + color.blue;
   
   // Calculates gradients of edges
   float leftEdgeSlope = (bottomLeft.y - topVertex.y) > edgeOffset ? 
       (bottomLeft.x - topVertex.x) / (bottomLeft.y - topVertex.y) : 0;
   float rightEdgeSlope = (bottomRight.y - topVertex.y) > edgeOffset ? 
       (bottomRight.x - topVertex.x) / (bottomRight.y - topVertex.y) : 0;
   
   // Calculates the scanline range
   int scanlineStart = clampToBounds(std::ceil(topVertex.y), 0, window.height - 1);
   int scanlineEnd = clampToBounds(std::floor(bottomLeft.y), 0, window.height - 1);
   
   // Process each of the scanlines
   for (int currentY = scanlineStart; currentY <= scanlineEnd; currentY++) {
       // Find the edge intersections
       float lineStart = topVertex.x + (currentY - topVertex.y) * leftEdgeSlope;
       float lineEnd = topVertex.x + (currentY - topVertex.y) * rightEdgeSlope;
       
       // Ensure the lineStart is to the left of lineEnd
       if (lineStart > lineEnd) std::swap(lineStart, lineEnd);
       
       // Apply small edge offset to prevent artifacts
       lineStart += edgeOffset;
       lineEnd -= edgeOffset;
       
       // Calculates pixel range for the scanline
       int pixelStart = clampToBounds(std::ceil(lineStart), 0, window.width - 1);
       int pixelEnd = clampToBounds(std::floor(lineEnd), 0, window.width - 1);
       
       // Fills the pixels in the scanline
       for (int x = pixelStart; x <= pixelEnd; x++) {
           window.setPixelColour(x, currentY, packedColor);
       }
   }
}

// Copied from above but interpolates between the two top vertices to the bottom one. 
void drawTopFlatTriangle(DrawingWindow &window,
   const CanvasPoint &topLeft,
   const CanvasPoint &topRight, 
   const CanvasPoint &bottomVertex,
   const Colour &color,
   float edgeOffset = 0.001f) 
{
   uint32_t packedColor = (255 << 24) + (color.red << 16) + 
                         (color.green << 8) + color.blue;
   
   float leftEdgeSlope = (bottomVertex.y - topLeft.y) > edgeOffset ? 
       (bottomVertex.x - topLeft.x) / (bottomVertex.y - topLeft.y) : 0;
   float rightEdgeSlope = (bottomVertex.y - topRight.y) > edgeOffset ? 
       (bottomVertex.x - topRight.x) / (bottomVertex.y - topRight.y) : 0;
   
   int scanlineStart = clampToBounds(std::ceil(topLeft.y), 0, window.height - 1);
   int scanlineEnd = clampToBounds(std::floor(bottomVertex.y), 0, window.height - 1);
   
   for (int currentY = scanlineStart; currentY <= scanlineEnd; currentY++) {
       float lineStart = topLeft.x + (currentY - topLeft.y) * leftEdgeSlope;
       float lineEnd = topRight.x + (currentY - topRight.y) * rightEdgeSlope;
       
       if (lineStart > lineEnd) std::swap(lineStart, lineEnd);
       
       lineStart += edgeOffset;
       lineEnd -= edgeOffset;
       
       int pixelStart = clampToBounds(std::ceil(lineStart), 0, window.width - 1);
       int pixelEnd = clampToBounds(std::floor(lineEnd), 0, window.width - 1);
       
       for (int x = pixelStart; x <= pixelEnd; x++) {
           window.setPixelColour(x, currentY, packedColor);
       }
   }
}

// Rasterises triangle by splitting it into two triangles with flat bases.
void drawTriangleFlat(DrawingWindow &window, const CanvasTriangle& triangle, const Colour& colour) {
    CanvasPoint vertices[3] = {triangle[0], triangle[1], triangle[2]};
    
    // Sorts (v0 = top, v1 = middle, v2 = bottom)
    if (vertices[1].y < vertices[0].y) std::swap(vertices[0], vertices[1]);
    if (vertices[2].y < vertices[0].y) std::swap(vertices[0], vertices[2]);
    if (vertices[2].y < vertices[1].y) std::swap(vertices[1], vertices[2]);
    
    // Skips triangles with no height
    if (vertices[2].y - vertices[0].y < 1) return;
    
    // Calculates the splitting point at middle vertex of y
    float midX = vertices[0].x + ((vertices[1].y - vertices[0].y) / 
                                 (vertices[2].y - vertices[0].y)) * 
                                (vertices[2].x - vertices[0].x);
    CanvasPoint midPoint(midX, vertices[1].y);
    
    // Fills the top and bottom triangles
    drawBottomFlatTriangle(window, vertices[0], vertices[1], midPoint, colour);
    drawTopFlatTriangle(window, vertices[1], midPoint, vertices[2], colour);
}

//------------------------------------------------------------------------------
// 3D RASTER
//------------------------------------------------------------------------------

void setPixelWithDepth(DrawingWindow &window, int x, int y, float depth, const Colour &color) {
   // Dont set if off-screen
   if (x < 0 || x >= window.width || y < 0 || y >= window.height) return;
   
   int bufferIndex = y * window.width + x;

   // Only draws the pixel if its is closer than the existing pixel
   if (depth > depthBuffer[bufferIndex]) {
       depthBuffer[bufferIndex] = depth;
       uint32_t packedColor = (255 << 24) + (color.red << 16) + 
                             (color.green << 8) + color.blue;
       window.setPixelColour(x, y, packedColor);
   }
}

void drawBottomFlatTriangleWithDepth(DrawingWindow &window,
   const CanvasPoint &topVertex,
   const CanvasPoint &bottomLeft,
   const CanvasPoint &bottomRight,
   const Colour &color)
{
   float leftEdgeSlope = (bottomLeft.x - topVertex.x) / (bottomLeft.y - topVertex.y);
   float rightEdgeSlope = (bottomRight.x - topVertex.x) / (bottomRight.y - topVertex.y);
   
   // Stores the inverse depth (1/z) values for interpolation
   float topDepth = topVertex.depth;
   float leftDepth = bottomLeft.depth;
   float rightDepth = bottomRight.depth;
   
   // Calculates the depth gradients along edges
   float leftDepthSlope = (leftDepth - topDepth) / (bottomLeft.y - topVertex.y);
   float rightDepthSlope = (rightDepth - topDepth) / (bottomRight.y - topVertex.y);

   int scanlineStart = clampToBounds(std::ceil(topVertex.y), 0, window.height - 1);
   int scanlineEnd = clampToBounds(std::floor(bottomLeft.y), 0, window.height - 1);
   
   for (int currentY = scanlineStart; currentY <= scanlineEnd; currentY++) {
       float lineStart = topVertex.x + (currentY - topVertex.y) * leftEdgeSlope;
       float lineEnd = topVertex.x + (currentY - topVertex.y) * rightEdgeSlope;
       
       // Interpolates the depth at scanline intersections
       float startDepth = topDepth + (currentY - topVertex.y) * leftDepthSlope;
       float endDepth = topDepth + (currentY - topVertex.y) * rightDepthSlope;
       
       if (lineStart > lineEnd) {
           std::swap(lineStart, lineEnd);
           std::swap(startDepth, endDepth);
       }
       
       int pixelStart = clampToBounds(std::ceil(lineStart), 0, window.width - 1);
       int pixelEnd = clampToBounds(std::floor(lineEnd), 0, window.width - 1);
       
       float scanlineWidth = lineEnd - lineStart;
       
       for (int x = pixelStart; x <= pixelEnd; x++) {
           float t = (x - lineStart) / scanlineWidth;
           float interpolatedDepth = (1.0f - t) * startDepth + t * endDepth;
           setPixelWithDepth(window, x, currentY, interpolatedDepth, color);
       }
   }
}

void drawTopFlatTriangleWithDepth(DrawingWindow &window,
    const CanvasPoint &topLeft,
    const CanvasPoint &topRight,
    const CanvasPoint &bottomVertex,
    const Colour &color)
    {
    float leftEdgeSlope = (bottomVertex.x - topLeft.x) / (bottomVertex.y - topLeft.y);
    float rightEdgeSlope = (bottomVertex.x - topRight.x) / (bottomVertex.y - topRight.y);

    float leftDepth = topLeft.depth;
    float rightDepth = topRight.depth;
    float bottomDepth = bottomVertex.depth;

    float leftDepthSlope = (bottomDepth - leftDepth) / (bottomVertex.y - topLeft.y);
    float rightDepthSlope = (bottomDepth - rightDepth) / (bottomVertex.y - topRight.y);

    int scanlineStart = clampToBounds(std::ceil(topLeft.y), 0, window.height - 1);
    int scanlineEnd = clampToBounds(std::floor(bottomVertex.y), 0, window.height - 1);
   
    for (int currentY = scanlineStart; currentY <= scanlineEnd; currentY++) {
        float lineStart = topLeft.x + (currentY - topLeft.y) * leftEdgeSlope;
        float lineEnd = topRight.x + (currentY - topRight.y) * rightEdgeSlope;
        
        float startDepth = leftDepth + (currentY - topLeft.y) * leftDepthSlope;
        float endDepth = rightDepth + (currentY - topRight.y) * rightDepthSlope;
        
        if (lineStart > lineEnd) {
            std::swap(lineStart, lineEnd);
            std::swap(startDepth, endDepth);
        }
        
        int pixelStart = clampToBounds(std::ceil(lineStart), 0, window.width - 1);
        int pixelEnd = clampToBounds(std::floor(lineEnd), 0, window.width - 1);
        
        float scanlineWidth = lineEnd - lineStart;
        
        for (int x = pixelStart; x <= pixelEnd; x++) {
            float t = (x - lineStart) / scanlineWidth;
            float interpolatedDepth = (1.0f - t) * startDepth + t * endDepth;
            setPixelWithDepth(window, x, currentY, interpolatedDepth, color);
        }
    }
}

void drawTriangleDepth(DrawingWindow &window, const CanvasTriangle& triangle, const Colour& colour) {
    CanvasPoint vertices[3] = {triangle[0], triangle[1], triangle[2]};
    
    if (vertices[1].y < vertices[0].y) std::swap(vertices[0], vertices[1]);
    if (vertices[2].y < vertices[0].y) std::swap(vertices[0], vertices[2]);
    if (vertices[2].y < vertices[1].y) std::swap(vertices[1], vertices[2]);
    
    if (vertices[2].y - vertices[0].y < 1) return;
    
    float t = (vertices[1].y - vertices[0].y) / (vertices[2].y - vertices[0].y);
    float splitX = vertices[0].x + t * (vertices[2].x - vertices[0].x);
    float splitDepth = vertices[0].depth + t * (vertices[2].depth - vertices[0].depth);
    
    CanvasPoint splitVertex(splitX, vertices[1].y, splitDepth);
    
    drawBottomFlatTriangleWithDepth(window, vertices[0], vertices[1], splitVertex, colour);
    drawTopFlatTriangleWithDepth(window, vertices[1], splitVertex, vertices[2], colour);
}

//------------------------------------------------------------------------------
// AMBIENT + DIFFUSE LIGHTING RASTER
//------------------------------------------------------------------------------

float calculateDiffuseLighting(const glm::vec3& pos, const glm::vec3& norm, const Light& light) {
    const float ambient = 0.3f, diffuse = 0.8f;

    // Calculates the normalized direction from surface to light
    glm::vec3 lightDir = glm::normalize(light.position - pos);
    
    // Calculates the distance-based light attenuation 
    float attenuation = 1.0f / (1.0f + 0.05f * glm::length(light.position - pos));

    // Combines the lighting components of diffuse and ambient
    return std::pow((ambient + std::max(0.0f, glm::dot(norm, lightDir)) * 
                    diffuse * attenuation) * light.intensity, 1.0f/2.2f);
}

void drawTriangleDiffuse(DrawingWindow &window, const CanvasTriangle& triangle, 
                             const ModelTriangle& modelTri) {
    // Setup and sort vertices
    VertexData data = setupVertexData(triangle, modelTri);
    
    // Exit if triangle is not visible
    if (!isTriangleVisible(data, window.width, window.height)) return;
    
    // Processes each scanline
    for (int y = clampToBounds(std::ceil(data.screenPositions[0].y), 0, window.height - 1);
         y <= clampToBounds(std::floor(data.screenPositions[2].y), 0, window.height - 1); y++) {
        
        // Gets the intersections for current scanline
        bool isTopHalf = y < data.screenPositions[1].y;
        ScanlineIntersection intersection = calculateScanlineIntersections(data, y, isTopHalf);
        
        // Process each pixel in scanline
        for (int x = clampToBounds(std::ceil(intersection.x1), 0, window.width - 1);
             x <= clampToBounds(std::floor(intersection.x2), 0, window.width - 1); x++) {
            
            float t = interpolationFactor(intersection.x1, intersection.x2, x, 0.0001f);
            float depth = intersection.depth1 + (intersection.depth2 - intersection.depth1) * t;
            
            size_t idx = y * window.width + x;
            if (depth > depthBuffer[idx]) {
                glm::vec3 pos = interpolateVec3(intersection.pos1, intersection.pos2, t);
                glm::vec3 norm = glm::normalize(interpolateVec3(intersection.normal1, intersection.normal2, t));
                float lighting = calculateDiffuseLighting(pos, norm, light);
                
                depthBuffer[idx] = depth;
                window.setPixelColour(x, y, calculatePixelColor(modelTri.colour, lighting));
            }
        }
    }
}

//------------------------------------------------------------------------------
// AMBIENT + DIFFUSE + SPECULAR LIGHTING RASTER
//------------------------------------------------------------------------------

float calculateSpecularLighting(const glm::vec3& pos, const glm::vec3& norm, const Light& light) {
    glm::vec3 viewPos(camera.x, camera.y, camera.z);
    return LightingComponents::calculate(pos, norm, light, viewPos);
}

// Copied from drawTriangleDiffuse but with differnet lighting calculations
void drawTriangleSpecular(DrawingWindow &window, const CanvasTriangle& triangle, 
                              const ModelTriangle& modelTri) {
    VertexData data = setupVertexData(triangle, modelTri);
    
    if (!isTriangleVisible(data, window.width, window.height)) return;
    
       for (int y = clampToBounds(std::ceil(data.screenPositions[0].y), 0, window.height - 1);
            y <= clampToBounds(std::floor(data.screenPositions[2].y), 0, window.height - 1); y++) {
        
        
        bool isTopHalf = y < data.screenPositions[1].y;
        ScanlineIntersection intersection = calculateScanlineIntersections(data, y, isTopHalf);
        
        for (int x = clampToBounds(std::ceil(intersection.x1), 0, window.width - 1);
             x <= clampToBounds(std::floor(intersection.x2), 0, window.width - 1); x++) {
            
            float t = interpolationFactor(intersection.x1, intersection.x2, x, 0.0001f);
            float depth = intersection.depth1 + (intersection.depth2 - intersection.depth1) * t;
            size_t idx = y * window.width + x;
            
            if (depth > depthBuffer[idx]) {
                glm::vec3 pos = interpolateVec3(intersection.pos1, intersection.pos2, t);
                glm::vec3 normal = glm::normalize(interpolateVec3(intersection.normal1, intersection.normal2, t));
                
                float lighting = calculateSpecularLighting(pos, normal, light);
                
                depthBuffer[idx] = depth;
                window.setPixelColour(x, y, calculatePixelColor(modelTri.colour, lighting));
            }
        }
    }
}

//------------------------------------------------------------------------------
// TEXTURED RASTER
//------------------------------------------------------------------------------

BarycentricCoords calculateBarycentricCoords(const glm::vec2& point, 
                                           const CanvasPoint& v0, 
                                           const CanvasPoint& v1, 
                                           const CanvasPoint& v2) {
    // Gets the vectors for computing the barycentric coordinates
    glm::vec2 base(v0.x, v0.y);
    glm::vec2 vec0(v1.x - v0.x, v1.y - v0.y);
    glm::vec2 vec1(v2.x - v0.x, v2.y - v0.y);
    glm::vec2 vec2 = point - base;

    // Computes the dot products
    float d00 = glm::dot(vec0, vec0);
    float d01 = glm::dot(vec0, vec1);
    float d11 = glm::dot(vec1, vec1);
    float d20 = glm::dot(vec2, vec0);
    float d21 = glm::dot(vec2, vec1);

    float denom = d00 * d11 - d01 * d01;
    
    BarycentricCoords result;
    result.v = (d11 * d20 - d01 * d21) / denom;
    result.w = (d00 * d21 - d01 * d20) / denom;
    result.u = 1.0f - result.v - result.w;
    
    // Checks if the point is inside of the triangle
    result.isInside = (result.u >= 0.0f && result.v >= 0.0f && result.w >= 0.0f);
    
    return result;
}

// Samples the texture at the given coordinates and returns the color
Colour sampleTexture(const Material& material, float texX, float texY) {
    // Handles texture wrapping
    texX = fmod(texX, 1.0f);
    texY = fmod(texY, 1.0f);
    if (texX < 0) texX += 1.0f;
    if (texY < 0) texY += 1.0f;

    size_t texPixelX = static_cast<size_t>(texX * (material.texture.width - 1));
    size_t texPixelY = static_cast<size_t>(texY * (material.texture.height - 1));

    // Clamps the texture coordinates
    texPixelX = std::min(texPixelX, material.texture.width - 1);
    texPixelY = std::min(texPixelY, material.texture.height - 1);

    // Sample the texture
    size_t texIndex = texPixelY * material.texture.width + texPixelX;
    uint32_t color = material.texture.pixels[texIndex];

    return Colour(
        (color >> 16) & 0xFF,
        (color >> 8) & 0xFF,
        color & 0xFF
    );
}

void drawTriangleTextured(DrawingWindow &window, const CanvasTriangle& canvasTri, 
                                 const ModelTriangle& modelTri, const Material& material) {
    if (!material.hasTexture) return;

    // Sorts the vertices by y-coordinate with their texture coordinates
    std::array<std::pair<CanvasPoint, TexturePoint>, 3> sortedPoints = {
        std::make_pair(canvasTri[0], modelTri.texturePoints[0]),
        std::make_pair(canvasTri[1], modelTri.texturePoints[1]),
        std::make_pair(canvasTri[2], modelTri.texturePoints[2])
    };
    if (sortedPoints[1].first.y < sortedPoints[0].first.y) std::swap(sortedPoints[0], sortedPoints[1]);
    if (sortedPoints[2].first.y < sortedPoints[0].first.y) std::swap(sortedPoints[0], sortedPoints[2]);
    if (sortedPoints[2].first.y < sortedPoints[1].first.y) std::swap(sortedPoints[1], sortedPoints[2]);

    // Computes the bounds we are drawing in
    int minY = clampToBounds(std::ceil(sortedPoints[0].first.y), 0, window.height - 1);
    int maxY = clampToBounds(std::floor(sortedPoints[2].first.y), 0, window.height - 1);

    // For each scanline
    for (int y = minY; y <= maxY; y++) {
        float scanlineStart = std::numeric_limits<float>::max();
        float scanlineEnd = std::numeric_limits<float>::lowest();

        for (int i = 0; i < 3; i++) {
            int j = (i + 1) % 3;
            const CanvasPoint& p1 = sortedPoints[i].first;
            const CanvasPoint& p2 = sortedPoints[j].first;

            if (p1.y != p2.y) {  // Skip the horizontal edges
                float x = p1.x + (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y);
                if (y >= std::min(p1.y, p2.y) && y <= std::max(p1.y, p2.y)) {
                    scanlineStart = std::min(scanlineStart, x);
                    scanlineEnd = std::max(scanlineEnd, x);
                }
            }
        }

        // Computes the pixel range for this scanline
        int minX = clampToBounds(std::ceil(scanlineStart), 0, window.width - 1);
        int maxX = clampToBounds(std::floor(scanlineEnd), 0, window.width - 1);

        // For each pixel in scanline
        for (int x = minX; x <= maxX; x++) {
            // Calculate the barycentric coordinates for the current pixel
            BarycentricCoords coords = calculateBarycentricCoords(
                glm::vec2(x, y),
                sortedPoints[0].first,
                sortedPoints[1].first,
                sortedPoints[2].first
            );

            if (coords.isInside) {
                // Interpolate the texture coordinates
                float texX = coords.u * sortedPoints[0].second.x + 
                           coords.v * sortedPoints[1].second.x + 
                           coords.w * sortedPoints[2].second.x;
                float texY = coords.u * sortedPoints[0].second.y + 
                           coords.v * sortedPoints[1].second.y + 
                           coords.w * sortedPoints[2].second.y;

                // Samples the texture to get the color
                Colour pixelColor = sampleTexture(material, texX, texY);

                // Interpolates the depth
                float depth = coords.u * sortedPoints[0].first.depth + 
                            coords.v * sortedPoints[1].first.depth + 
                            coords.w * sortedPoints[2].first.depth;

                // Sets the pixel
                setPixelWithDepth(window, x, y, depth, pixelColor);
            }
        }
    }
}

void drawTexturedTriangle(DrawingWindow& window, const CanvasTriangle& canvas, 
                        const ModelTriangle& model, const Material& material) {
   if (material.hasTexture) {
       drawTriangleTextured(window, canvas, model, material);
   } else {
       drawTriangleFlat(window, canvas, model.colour);
   }
}

//------------------------------------------------------------------------------
// GOURAUD SHADING
//------------------------------------------------------------------------------

// Calculates the lighting intensity for a vertex
VertexLight calculateVertexLight(const glm::vec3& vertexPos, const glm::vec3& normal, const Colour& material) {
    glm::vec3 viewPos(camera.x, camera.y, camera.z);
    float lighting = LightingComponents::calculate(vertexPos, normal, light, viewPos);
    
    return VertexLight(
        std::min(255.0f, material.red * lighting),
        std::min(255.0f, material.green * lighting),
        std::min(255.0f, material.blue * lighting)
    );
}

// Helper struct for finding the minimum and maximum values of different vertex attributes
struct VertexRange {
    float min, max;
    
    static VertexRange calculate(const CanvasTriangle& triangle, 
                               std::function<float(const CanvasPoint&)> getter) {
        return {
            std::min({getter(triangle[0]), getter(triangle[1]), getter(triangle[2])}),
            std::max({getter(triangle[0]), getter(triangle[1]), getter(triangle[2])})
        };
    }
};

// Helper struct for the screen-space boundaries of a triangle
struct TriangleBounds {
    int xStart, xEnd, yStart, yEnd;
    
    static TriangleBounds calculate(const CanvasTriangle& triangle, 
                                  int windowWidth, int windowHeight) {
        // Get the x and y ranges
        auto xRange = VertexRange::calculate(triangle, [](const CanvasPoint& p) { return p.x; });
        auto yRange = VertexRange::calculate(triangle, [](const CanvasPoint& p) { return p.y; });
        
        // Clamps to the window boundaries
        return {
            clampToBounds(std::floor(xRange.min), 0, windowWidth - 1),
            clampToBounds(std::ceil(xRange.max), 0, windowWidth - 1),
            clampToBounds(std::floor(yRange.min), 0, windowHeight - 1),
            clampToBounds(std::ceil(yRange.max), 0, windowHeight - 1)
        };
    }
};

void drawTriangleGouraud(DrawingWindow &window, const CanvasTriangle& triangle, const ModelTriangle& modelTri) {
    // Pre-calculate lighting at each vertex 
    std::array<VertexLight, 3> vertexColors;
    for(int i = 0; i < 3; i++) {
        vertexColors[i] = calculateVertexLight(
            modelTri.vertices[i], 
            glm::normalize(modelTri.vertices[i]), 
            modelTri.colour
        );
    }

    // Gets the triangle bounds
    auto bounds = TriangleBounds::calculate(triangle, window.width, window.height);

    // Rasterisation
    for (int y = bounds.yStart; y <= bounds.yEnd; y++) {
        for (int x = bounds.xStart; x <= bounds.xEnd; x++) {
            glm::vec2 point(x, y);
            auto bary = calculateBarycentricCoords(point, triangle[0], triangle[1], triangle[2]);
            
            // Only process the pixels if they are inside the triangle
            if (bary.isInside) {
                int index = y * window.width + x;
                float depth = bary.u * triangle[0].depth + bary.v * triangle[1].depth + bary.w * triangle[2].depth;
                
                if (depth > depthBuffer[index]) {
                    depthBuffer[index] = depth;

                    // Pack an Interpolation between three different vertex colors 
                    // with a gamma correction
                    uint32_t color = (255 << 24) + 
                        (interpolateColor(
                            vertexColors[0].red, 
                            vertexColors[1].red, 
                            vertexColors[2].red, 
                            bary) << 16) +
                        (interpolateColor(
                            vertexColors[0].green, 
                            vertexColors[1].green, 
                            vertexColors[2].green, bary) << 8) +
                        interpolateColor(vertexColors[0].blue, 
                            vertexColors[1].blue, 
                            vertexColors[2].blue, bary);
                    window.setPixelColour(x, y, color);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// PHONG MODE RASTERISATION
//------------------------------------------------------------------------------

BumpMap createSphericalBumpMap(int width = 256, int height = 256) {  // Even higher resolution
    BumpMap bumpMap;
    bumpMap.width = width;
    bumpMap.height = height;
    bumpMap.normals.resize(width * height);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float u = static_cast<float>(x) / (width - 1);
            float v = static_cast<float>(y) / (height - 1);
            
            float fx = 0.0f;
            float fy = 0.0f;
            
            // Adds multiple frequencies with decreasing amplitude
            for (int i = 1; i <= 3; i++) {
                float freq = i * 10.0f;
                float amp = 0.1f / i;
                fx += std::sin(u * freq * M_PI) * std::sin(v * freq * M_PI) * amp;
                fy += std::sin(u * freq * M_PI) * std::cos(v * freq * M_PI) * amp;
            }
            
            float fz = 1.0f;
            
            bumpMap.normals[y * width + x] = glm::normalize(glm::vec3(fx, fy, fz));
        }
    }
    return bumpMap;
}

glm::vec3 getBumpMappedNormal(const glm::vec3& baseNormal, const BumpMap& bumpMap, 
                             const glm::vec3& worldPos, float bumpStrength = 0.15f) {
    glm::vec3 normalizedPos = glm::normalize(worldPos);
    
    // Generates the UV coordinates with smoother wrapping
    float u = 0.5f + std::atan2(normalizedPos.z, normalizedPos.x) / (2.0f * M_PI);
    float v = 0.5f - std::asin(normalizedPos.y) / M_PI;
    
    // Bilinear interpolation for smoother normal sampling
    float uf = u * (bumpMap.width - 1);
    float vf = v * (bumpMap.height - 1);
    
    int x0 = static_cast<int>(std::floor(uf));
    int y0 = static_cast<int>(std::floor(vf));
    int x1 = std::min(x0 + 1, bumpMap.width - 1);
    int y1 = std::min(y0 + 1, bumpMap.height - 1);
    
    float fx = uf - x0;
    float fy = vf - y0;
    
    // Gets the surrounding normals
    glm::vec3 n00 = bumpMap.normals[y0 * bumpMap.width + x0];
    glm::vec3 n10 = bumpMap.normals[y0 * bumpMap.width + x1];
    glm::vec3 n01 = bumpMap.normals[y1 * bumpMap.width + x0];
    glm::vec3 n11 = bumpMap.normals[y1 * bumpMap.width + x1];
    
    // Bilinear interpolation
    glm::vec3 bumpNormal = glm::normalize(
        n00 * ((1-fx) * (1-fy)) +
        n10 * (fx * (1-fy)) +
        n01 * ((1-fx) * fy) +
        n11 * (fx * fy)
    );
    
    // Creates a smoother TBN matrix
    glm::vec3 tangent;
    if (std::abs(baseNormal.y) < 0.999f) {
        tangent = glm::normalize(glm::cross(baseNormal, glm::vec3(0, 1, 0)));
    } else {
        tangent = glm::normalize(glm::cross(baseNormal, glm::vec3(1, 0, 0)));
    }
    glm::vec3 bitangent = glm::normalize(glm::cross(baseNormal, tangent));
    
    glm::mat3 TBN(tangent, bitangent, baseNormal);
    glm::vec3 worldBumpNormal = TBN * bumpNormal;
    
    return glm::normalize(glm::mix(baseNormal, worldBumpNormal, bumpStrength));
}

// Calculates the Phong lighting for a point
float calculatePhongLighting(const glm::vec3& fragmentPos, 
                           const glm::vec3& normal,
                           const glm::vec3& viewPos,
                           const Light& light) {
    // Phong lighting constants
    const float ambient_strength = 0.2f;
    const float diffuse_strength = 0.7f;
    const float specular_strength = 0.3f;
    const float shininess = 48.0f;
    
    // Calculates the lighting vectors
    glm::vec3 lightDir = glm::normalize(light.position - fragmentPos);
    glm::vec3 viewDir = glm::normalize(viewPos - fragmentPos);
    glm::vec3 reflectDir = glm::reflect(-lightDir, normal);
    
    float ambient = ambient_strength;
    float diffuse = diffuse_strength * std::max(0.0f, glm::dot(normal, lightDir));
    float specular = specular_strength * std::pow(std::max(0.0f, glm::dot(viewDir, reflectDir)), shininess);
    
    // Calculate attenuation
    float distance = glm::length(light.position - fragmentPos);
    float attenuation = 1.0f / (1.0f + 0.03f * distance);
    
    // Combine lighting with gamma correction
    return std::pow((ambient + (diffuse + specular) * attenuation) * light.intensity, 1.0f/2.2f);
}

// Processes a single pixel in the Phong shader
void processPhongPixel(DrawingWindow& window,
                      int x, int y,
                      float invDepth,
                      const glm::vec3& fragmentPos,
                      const glm::vec3& baseNormal,
                      const Colour& matColor,
                      const BumpMap& bumpMap,
                      std::vector<float>& depthBuffer,
                      const glm::vec3& viewPos,
                      const Light& light) {
    size_t index = y * window.width + x;
    
    if (invDepth > depthBuffer[index]) {
        // Applies the bump mapping and calculates the lighting
        glm::vec3 normal = getBumpMappedNormal(baseNormal, bumpMap, fragmentPos);
        float totalLighting = calculatePhongLighting(fragmentPos, normal, viewPos, light);
        
        // Applies the lighting to color and then sets a pixel
        uint32_t finalColor = (255 << 24) + 
            (static_cast<uint8_t>(std::min(255.0f, matColor.red * totalLighting)) << 16) +
            (static_cast<uint8_t>(std::min(255.0f, matColor.green * totalLighting)) << 8) +
            static_cast<uint8_t>(std::min(255.0f, matColor.blue * totalLighting));
        
        depthBuffer[index] = invDepth;
        window.setPixelColour(x, y, finalColor);
    }
}

// Copied code from other rasterisation functions but with phong helper functions
void drawTrianglePhong(DrawingWindow &window, const CanvasTriangle& triangle, 
                           const ModelTriangle& modelTri) {
    static BumpMap bumpMap = createSphericalBumpMap();
    glm::vec3 viewPos(camera.x, camera.y, camera.z);
    const float EPSILON = 0.0001f;

    VertexData data = setupVertexData(triangle, modelTri);
    if (!isTriangleVisible(data, window.width, window.height)) return;

    for (int y = clampToBounds(std::ceil(data.screenPositions[0].y), 0, window.height - 1);
         y <= clampToBounds(std::floor(data.screenPositions[2].y), 0, window.height - 1); y++) {
        
        // Get scanline intersections
        bool isTopHalf = y < data.screenPositions[1].y;
        ScanlineIntersection intersection = calculateScanlineIntersections(data, y, isTopHalf);
        
        int xStart = clampToBounds(std::ceil(intersection.x1), 0, window.width - 1);
        int xEnd = clampToBounds(std::floor(intersection.x2), 0, window.width - 1);
        float xLength = intersection.x2 - intersection.x1;

        for (int x = xStart; x <= xEnd; x++) {
            float xPercent = xLength < EPSILON ? 0.0f : (x - intersection.x1) / xLength;
            
            float invDepth = interpolate(intersection.depth1, intersection.depth2, xPercent);
            glm::vec3 fragmentPos = interpolateVec3(intersection.pos1, intersection.pos2, xPercent);
            glm::vec3 baseNormal = glm::normalize(interpolateVec3(intersection.normal1, intersection.normal2, xPercent));
            
            processPhongPixel(window, x, y, invDepth, fragmentPos, baseNormal,
                            modelTri.colour, bumpMap, depthBuffer, viewPos, light);
        }
    }
}

//------------------------------------------------------------------------------
// HARD SHADOW RAYTRACING
//------------------------------------------------------------------------------

void drawShadowMode(DrawingWindow &window) {
    window.clearPixels();
    clearDepthBuffer();

    const float ambient_strength = 0.2f;
    
    // Camera properties for ray tracing
    float fovRadians = FOV * PI / 180.0f;
    float aspectRatio = static_cast<float>(WIDTH) / HEIGHT;
    float scale = tan(fovRadians * 0.5f) * 1.4f; 
    
    // For each pixel on the screen
    #pragma omp parallel for collapse(2) 
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            // Convert pixel to screen space
            float pixelNdcX = (2.0f * (x + 0.5f) / WIDTH - 1.0f) * scale * aspectRatio;
            float pixelNdcY = (1.0f - 2.0f * (y + 0.5f) / HEIGHT) * scale;
            
            // Calculates the ray direction
            glm::vec3 rayDirection = glm::normalize(glm::vec3(pixelNdcX, pixelNdcY, 1.0f));
            
            // Transforms the ray direction by the camera rotation
            rayDirection = rotateX(rotateY(rayDirection, camera.rotationY), camera.rotationX);
            
            // Get the closest intersection
            glm::vec3 cameraPos(camera.x, camera.y, camera.z);
            RayTriangleIntersection intersection = getClosestValidIntersection(cameraPos, rayDirection, triangles);
            
            if (intersection.distanceFromCamera != std::numeric_limits<float>::infinity()) {
                Colour colour = intersection.intersectedTriangle.colour;
                
                // Checks if the point is in shadow
                if (isPointInShadow(intersection.intersectionPoint, &triangles[intersection.triangleIndex], triangles)) {
                    colour.red = static_cast<int>(colour.red * ambient_strength);
                    colour.green = static_cast<int>(colour.green * ambient_strength);
                    colour.blue = static_cast<int>(colour.blue * ambient_strength);
                }
                
                uint32_t pixelColor = (255 << 24) + (colour.red << 16) + (colour.green << 8) + colour.blue;
                window.setPixelColour(x, y, pixelColor);
            }
        }
    }
}

//------------------------------------------------------------------------------
// SHOFT SHADOW RAYTRACING
//------------------------------------------------------------------------------

// Generates the sample points on an area of a light
std::vector<glm::vec3> generateAreaLightSamples(const glm::vec3& lightCenter, float radius, int samples) {
    std::vector<glm::vec3> lightSamples;
    lightSamples.reserve(samples);

    // Calculates the grid dimensions for stratified sampling
    int gridSize = static_cast<int>(sqrt(samples));
    float cellSize = 2.0f / gridSize;  // Maps the grid to a [-1, 1] range

    // Generates the stratified samples across the light
    for(int i = 0; i < gridSize * gridSize; i++) {
        // Converts the linear index to 2D grid coordinates and centers them
        float u = ((i / gridSize) + 0.5f) * cellSize - 1.0f; // Maps to a [-1, 1] range
        float v = ((i % gridSize) + 0.5f) * cellSize - 1.0f; // Maps to a [-1, 1] range
        
        float angle = u * PI;
        float distance = v * radius;
        
        // Generates the sample point with slight vertical variations
        lightSamples.push_back(light.position + glm::vec3(
            cos(angle) * distance,
            radius * (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.25f,
            sin(angle) * distance
        ));
    }
    return lightSamples;
}

// Computes the soft shadow intensity at a given point by sampling multiple light positions
float computeSoftShadow(const glm::vec3& point, const ModelTriangle* sourceTriangle, 
                       const std::vector<ModelTriangle>& triangles) {
    float visibility = 0.0f;
    const float ambient_strength = 0.2f;
    const int shadow_samples = 16;        // Soft shadow quality (higher = slower render)
    const float shadow_bias = 0.005f;     // Shadow acne prevention
    const float light_radius = 0.15f;     // Size of the light
    
    // Tests the visibility from the point to each light sample
    for(const auto& lightPos : generateAreaLightSamples(light.position, light_radius, shadow_samples)) {
        glm::vec3 directionToLight = glm::normalize(lightPos - point);
        float distanceToLight = glm::length(lightPos - point);
        
         // Offsets the ray origin slightly to prevent self-intersection
        glm::vec3 offsetPoint = point + directionToLight * shadow_bias;
        RayTriangleIntersection shadowRay = getClosestValidIntersection(
            offsetPoint, directionToLight, triangles, sourceTriangle
        );
        
        // If there is no intersection found this sample is visible
        if(shadowRay.distanceFromCamera >= distanceToLight) {
            visibility += 1.0f / shadow_samples;
        }
    }
    
    // Interpolates between the ambient and full lighting based on the visibility
    return interpolate(ambient_strength, 1.0f, visibility);
}

void drawSoftShadowMode(DrawingWindow &window) {
    window.clearPixels();
    clearDepthBuffer();
    
    // Calculates the camera parameters for ray generation
    float scale = tan(FOV * PI / 360.0f) * 1.4f; 
    float aspectRatio = static_cast<float>(WIDTH) / HEIGHT;
    
    #pragma omp parallel for collapse(2)
    for(int y = 0; y < HEIGHT; y++) {
        for(int x = 0; x < WIDTH; x++) {
            glm::vec3 rayDirection = glm::normalize(glm::vec3(
                (2.0f * (x + 0.5f) / WIDTH - 1.0f) * scale * aspectRatio,
                (1.0f - 2.0f * (y + 0.5f) / HEIGHT) * scale,
                1.0f
            ));
            
            rayDirection = rotateX(rotateY(rayDirection, camera.rotationY), camera.rotationX);
            
            // Finds the closest intersection for the main ray
            RayTriangleIntersection intersection = getClosestValidIntersection(
                glm::vec3(camera.x, camera.y, camera.z), rayDirection, triangles
            );
            
             // If there is an intersection calclates the color with soft shadows
            if(intersection.distanceFromCamera != std::numeric_limits<float>::infinity()) {
                Colour colour = intersection.intersectedTriangle.colour;
                float shadowFactor = computeSoftShadow(
                    intersection.intersectionPoint,
                    &triangles[intersection.triangleIndex],
                    triangles
                );
                
                window.setPixelColour(x, y, calculatePixelColor(colour, shadowFactor));
            }
        }
    }
}

//------------------------------------------------------------------------------
// RENDERING CORE
//------------------------------------------------------------------------------

void draw(DrawingWindow& window) {
    window.clearPixels();
    clearDepthBuffer();

    // Handle raytracing modes
    if (modelState.renderMode == SHADOW) {
        drawShadowMode(window);
        return;
    }
    if (modelState.renderMode == SOFT) {
        drawSoftShadowMode(window);
        return;
    }

    // Handle rasterising modes
    for (size_t i = 0; i < canvasTriangles.size(); i++) {
        const CanvasTriangle& canvas = canvasTriangles[i];
        const ModelTriangle& model = triangles[i];

        switch (modelState.renderMode) {
            case WIRE:
                drawTriangleWireframe(window, canvas, model);
                break;
            case FLAT:
                drawTriangleFlat(window, canvas, model.colour);
                break;
            case DEPTH:
                drawTriangleDepth(window, canvas, model.colour);
                break;
            case DIFFUSE:
                drawTriangleDiffuse(window, canvas, model);
                break;
            case SPECULAR:
                drawTriangleSpecular(window, canvas, model);
                break;
            case GOURAUD:
                drawTriangleGouraud(window, canvas, model);
                break;
            case PHONG:
                drawTrianglePhong(window, canvas, model);
                break;
            case TEXTURE:
                drawTriangleTextured(window, canvas, model, materials[currentMaterialIndex]);
                break;
        }
    }
}

//------------------------------------------------------------------------------
// VIEW MANAGEMENT
//------------------------------------------------------------------------------

void resetViewToDefault() {
    // Resets the model rotation to the original values
    modelState.rotationX = 25.21f * PI / 180.0f;
    modelState.rotationY = 29.79f * PI / 180.0f;
    
    // Resets the camera to the original values
    camera.x = 0.0f;
    camera.y = 0.0f;
    camera.z = -1.0f;
    camera.rotationX = 0.0f;
    camera.rotationY = 0.0f;
    
    modelState.isDragging = false;
    
    updateModelRotation();
    updateCanvasTriangles();
}

void lookAt(const glm::vec3& eye, const glm::vec3& center) {
    // Calculates the direction vector
    glm::vec3 direction = glm::normalize(center - eye);
    
    // Calculates and sets the rotation angles
    camera.rotationY = atan2(direction.x, direction.z);
    camera.rotationX = -asin(direction.y);
}

//------------------------------------------------------------------------------
// CAMERA CONTROLS
//------------------------------------------------------------------------------

void handleCameraMovement(SDL_Keycode key, float moveSpeed, bool& needsUpdate) {
    switch(key) {
        case SDLK_w: camera.z += moveSpeed; break;
        case SDLK_s: camera.z -= moveSpeed; break;
        case SDLK_a: camera.x -= moveSpeed; break;
        case SDLK_d: camera.x += moveSpeed; break;
        case SDLK_q: camera.y -= moveSpeed; break;
        case SDLK_e: camera.y += moveSpeed; break;
        default: return;
    }
    needsUpdate = true;
}

void handleCameraRotation(SDL_Keycode key, float rotateSpeed, bool& needsUpdate) {
    switch(key) {
        case SDLK_UP:    camera.rotationX -= rotateSpeed; break;
        case SDLK_DOWN:  camera.rotationX += rotateSpeed; break;
        case SDLK_LEFT:  camera.rotationY -= rotateSpeed; break;
        case SDLK_RIGHT: camera.rotationY += rotateSpeed; break;
        default: return;
    }
    needsUpdate = true;
}

//------------------------------------------------------------------------------
// MOUSE INTERACTION
//------------------------------------------------------------------------------

void handleMouseDown(const SDL_MouseButtonEvent& button) {
    if (button.button == SDL_BUTTON_LEFT) {
        modelState.isDragging = true;
        modelState.mouseX = button.x;
        modelState.mouseY = button.y;
    }
}

void handleMouseUp(const SDL_MouseButtonEvent& button) {
    if (button.button == SDL_BUTTON_LEFT) {
        modelState.isDragging = false;
    }
}

void handleMouseMotion(const SDL_MouseMotionEvent& motion) {
   // Skip if mosue is not being dragged 
   if (!modelState.isDragging) return;
   
   // Converts the mouse movement to rotation angles
   float deltaX = (motion.x - modelState.mouseX) * 0.005f;  // Scaled to slow down speed
   float deltaY = (motion.y - modelState.mouseY) * 0.005f;
   
   modelState.rotationY -= deltaX; 
   modelState.rotationX -= deltaY;  

   modelState.mouseX = motion.x;
   modelState.mouseY = motion.y;
   
   updateModelRotation();
   updateCanvasTriangles();
}

//------------------------------------------------------------------------------
// INPUT PROCESSING
//------------------------------------------------------------------------------

void beginAnimation(bool isComplex) {
    animState.isAnimating = true;
    animState.isComplexAnim = isComplex;
    animState.startTime = SDL_GetTicks();
    animState.initialCameraPos = glm::vec3(camera.x, camera.y, camera.z);
    animState.initialLightPos = light.position;
    animState.previousRenderMode = modelState.renderMode;
    modelState.renderMode = PHONG;
    
    if (isComplex) {
        physicsState = PhysicsState();  // Resets to initial values
        physicsState.position = glm::vec3(0.0f, 3.0f, 0.0f); 
        camera.y = 0.0f;
    }
}

void endAnimation() {
    animState.isAnimating = false;
    animState.isComplexAnim = false;
}

void handleAnimationToggle(bool isComplex) {
    if (!animState.isAnimating) {
        beginAnimation(isComplex);
        std::cout << "Started " << (isComplex ? "complex" : "simple") 
                 << " orbital animation (Phong rendering)\n";
    } else {
        endAnimation();
        std::cout << "Stopped animation, restored ";
    }
}

void handleRenderModeChange(RenderMode newMode) {
    modelState.renderMode = newMode;
    
    std::string modeName;
    switch(newMode) {
        case WIRE: modeName = "WIRE"; break;
        case FLAT: modeName = "FLAT"; break;
        case DEPTH: modeName = "DEPTH"; break;
        case DIFFUSE: modeName = "DIFFUSE"; break;
        case SPECULAR: modeName = "SPECULAR"; break;
        case TEXTURE: modeName = "TEXTURE"; break;
        case GOURAUD: modeName = "GOURAUD"; break;
        case PHONG: modeName = "PHONG"; break;
        case SHADOW: modeName = "SHADOW"; break;
        case SOFT: modeName = "SOFT"; break;
    }
    
    std::cout << "Switched to " << modeName << " mode\n";
}

void handleKeyPress(const SDL_KeyboardEvent& key) {
    bool needsUpdate = false;
    const float speedMultiplier = (SDL_GetModState() & KMOD_SHIFT) ? 2.0f : 1.0f;
    
    // Handles the camera controls
    handleCameraMovement(key.keysym.sym, camera.moveSpeed * speedMultiplier, needsUpdate);
    handleCameraRotation(key.keysym.sym, camera.rotationSpeed * speedMultiplier, needsUpdate);
    
    // Handles the render mode and animation toggles
    switch(key.keysym.sym) {
        case SDLK_h:
            handleRenderModeChange(SHADOW);
            break;

        case SDLK_s:
            handleRenderModeChange(SOFT);
            break;
           
        case SDLK_SPACE: {
            // Get the next mode, skipping SHADOW and SOFT modes
            int nextMode = (static_cast<int>(modelState.renderMode) + 1) % 8;  // Only cycle throughthe first 8 modes
            handleRenderModeChange(static_cast<RenderMode>(nextMode));
            break;
        }
           
        case SDLK_p:
            handleAnimationToggle(false);
            break;
           
        case SDLK_o:
            handleAnimationToggle(true);
            break;
    }
    
    if (needsUpdate) {
        updateModelRotation();
        updateCanvasTriangles();
    }
}

void handleEvent(SDL_Event event, DrawingWindow &window) {
    switch(event.type) {
        case SDL_MOUSEBUTTONDOWN:
            handleMouseDown(event.button);
            break;
        case SDL_MOUSEBUTTONUP:
            handleMouseUp(event.button);
            break;
        case SDL_MOUSEMOTION:
            handleMouseMotion(event.motion);
            break;
        case SDL_KEYDOWN:
            handleKeyPress(event.key);
            break;
    }
}

//------------------------------------------------------------------------------
// ANIMATION SYSTEM
//------------------------------------------------------------------------------

// A simple animation of our spining around slowly
void updateSimpleAnimation(float elapsedTime) {
    // Added a PI/2 phase shift to make it face front
    float totalAngle = (elapsedTime * 3.0f * PI / (animState.duration/1000.0f)) + PI/2;
    float angle = fmod(totalAngle, 2.0f * PI);
    float radius = 1.2f;
    
    // Rotate camera around object at fixed height
    glm::vec3 cameraPos(
        radius * cos(angle),
        0.0f,
        radius * sin(angle)
    );
    
    camera.x = cameraPos.x;
    camera.y = cameraPos.y;
    camera.z = cameraPos.z;
    
    // Keep object completely straight
    modelState.rotationX = 0.0f;
    modelState.rotationY = 0.0f;
    
    // Look directly at object's position
    lookAt(cameraPos, glm::vec3(0.0f, 0.0f, 0.0f));
    
    // Light follows camera at fixed height, also adjusted by same phase shift
    light.position = glm::vec3(
        1.0f * cos(angle),
        1.5f,
        1.0f * sin(angle)
    );
    
    updateModelRotation();
    updateCanvasTriangles();
}

// A complex animation of our object falling down, bouncing, then spining around slowly
void updateComplexAnimation(float elapsedTime) {
    const float deltaTime = 1.0f/60.0f;
    const float transitionDuration = 1.0f;
    const float orbitRadius = 1.2f;
    const float rotationSpeed = 1.5f;
    
    // Keeps the model oriented straight
    modelState.rotationX = 0.0f;
    modelState.rotationY = 0.0f;
    
    if (!physicsState.hasLanded) {
        // Physics simulation
        physicsState.velocity += physicsState.acceleration * deltaTime;
        physicsState.position += physicsState.velocity * deltaTime;
        
        if (physicsState.position.y <= physicsState.landingY) {
            physicsState.position.y = physicsState.landingY;
            physicsState.velocity.y *= -physicsState.dampening;
            
            if (std::abs(physicsState.velocity.y) < 0.1f) {
                physicsState.hasLanded = true;
                physicsState.velocity = glm::vec3(0.0f);
                physicsState.timeSinceLanding = 0.0f;
            }
        }
        
        // Fixed camera position during the fall
        camera.x = 0.0f;
        camera.y = -physicsState.position.y;
        camera.z = -orbitRadius;
        lookAt(glm::vec3(camera.x, camera.y, camera.z), -physicsState.position);
        
    } else {
        physicsState.timeSinceLanding += deltaTime;
        float t = physicsState.timeSinceLanding;
        
        // Smooths the transition to rotating
        float blendFactor = std::min(t / transitionDuration, 1.0f);
        float smoothBlend = (1.0f - cos(blendFactor * PI)) * 0.5f;
        
        // Calculates the spin position with increased speed
        float cameraAngle = t * rotationSpeed * smoothBlend;
        
        // Blends from fixed position during fall to the later spinning
        camera.x = orbitRadius * sin(cameraAngle);
        camera.z = -orbitRadius * cos(cameraAngle);
        camera.y = -physicsState.landingY;
        
        lookAt(glm::vec3(camera.x, camera.y, camera.z), -physicsState.position);
    }
    
    light.position = glm::vec3(1.0f, 3.0f, -1.0f);
    
    updateModelRotation();
    updateCanvasTriangles();
}

void updateAnimation() {
    float elapsedTime = (SDL_GetTicks() - animState.startTime) / 1000.0f;
    
    if (elapsedTime <= animState.duration/1000.0f) {
        if (animState.isComplexAnim) {
            updateComplexAnimation(elapsedTime);
        } else {
            updateSimpleAnimation(elapsedTime);
        }
    } else {
        endAnimation();
    }
}

//------------------------------------------------------------------------------
// MAIN PROGRAM LOOP
//------------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    DrawingWindow window = DrawingWindow(WIDTH, HEIGHT, false);
    SDL_Event event;
    
    initializeDepthBuffer(window);
    loadOBJFile("logo.obj");
    
    // Start with complex animation immediately
    beginAnimation(true);
    bool hasShownControls = false;
    
    while (true) {
        if (window.pollForInputEvents(event)) {
            handleEvent(event, window);
        }
        
        if (animState.isAnimating) {
            updateAnimation();
        } else if (!hasShownControls) {
            // Print the controls once the animation ends
            std::cout << "\nAvailable Controls:\n";
            std::cout << "----------------\n";
            std::cout << "Mouse Controls:\n";
            std::cout << "- Left Click + Drag: Rotate model\n\n";
            
            std::cout << "Camera Movement:\n";
            std::cout << "- W/S: Move camera forward/backward\n";
            std::cout << "- A/D: Move camera left/right\n";
            std::cout << "- Q/E: Move camera up/down\n";
            std::cout << "- Arrow Keys: Rotate camera\n";
            std::cout << "- Hold Shift + Any Movement: Move/rotate faster\n\n";
            
            std::cout << "Rendering Modes:\n";
            std::cout << "- SPACE: Cycle through rendering modes (Wireframe/Flat/Depth/Diffuse/Specular/Texture/Gouraud/Phong)\n";
            std::cout << "- H: Hard shadow ray tracing mode\n";
            std::cout << "- S: Soft shadow ray tracing mode\n\n";
            
            std::cout << "Animations:\n";
            std::cout << "- O: Toggle complex animation (physics + orbit)\n";
            std::cout << "- P: Toggle simple orbit animation\n";
            
            hasShownControls = true;
        }
        
        draw(window);
        window.renderFrame();
    }
}