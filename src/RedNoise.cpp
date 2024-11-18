#include <CanvasTriangle.h>
#include <DrawingWindow.h>
#include <Utils.h>
#include <fstream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <iostream>
#include <ModelTriangle.h>
#include <Colour.h>

// ------------------------
// Core Configuration
// ------------------------
// (Config namespace - global settings and constants for the renderer)
namespace Config {
    namespace Display {
        constexpr int WIDTH = 320;
        constexpr int HEIGHT = 240;
        constexpr float FOCAL_LENGTH = 500.0f;
    }

    namespace Camera {
        constexpr float TRANSLATION_STEP = 1.0f;
        constexpr float PAN_STEP = 0.01f;
        constexpr float TILT_STEP = 0.01f;
    }

    namespace Colours {
        const Colour DEFAULT_MESH_COLOR{192, 192, 192};
    }

    struct Render {
        enum class Mode {
            WIREFRAME,
            RASTERIZED
        };
        Mode currentMode = Mode::RASTERIZED;
        std::vector<float> depthBuffer;
        
        Render() {
            depthBuffer.resize(Display::WIDTH * Display::HEIGHT, std::numeric_limits<float>::infinity());
        }

        void clearDepthBuffer() {
            std::fill(depthBuffer.begin(), depthBuffer.end(), std::numeric_limits<float>::infinity());
        }
    };
}

// ------------------------
// Math Utilities
// ------------------------
// (Math class - fundamental operations used by other components)
class Math {
    public:
        struct TriangleSlopes {
            float topToMiddleX;
            float topToBottomX;
            float middleToBottomX;
            float topToMiddleZ;
            float topToBottomZ;
            float middleToBottomZ;
        };

        static std::vector<float> interpolateSingleFloats(
            float from, float to,
            int numberOfValues
        ) {
            std::vector<float> results;
            float diff = to - from;
            float stepSize = diff / (numberOfValues - 1);
            
            for(int i = 0; i < numberOfValues; i++) {
                results.push_back(from + (stepSize * i));
            }
            
            return results;
        }

        static void sortVerticesByY(
            CanvasPoint& v0, 
            CanvasPoint& v1, 
            CanvasPoint& v2
        ) {
            if (v0.y > v1.y) std::swap(v0, v1);
            if (v0.y > v2.y) std::swap(v0, v2);
            if (v1.y > v2.y) std::swap(v1, v2);
        }

        static float calculateSlope(
            float start, float end,
            float heightDiff
        ) {
            return heightDiff > 0 ? (end - start) / heightDiff : 0;
        }

        static TriangleSlopes calculateTriangleSlopes(
            const CanvasPoint& v0, 
            const CanvasPoint& v1, 
            const CanvasPoint& v2
        ) {
            float topToMiddleHeight = v1.y - v0.y;
            float topToBottomHeight = v2.y - v0.y;
            float middleToBottomHeight = v2.y - v1.y;

            return {
                calculateSlope(v0.x, v1.x, topToMiddleHeight),
                calculateSlope(v0.x, v2.x, topToBottomHeight),
                calculateSlope(v1.x, v2.x, middleToBottomHeight),
                calculateSlope(v0.depth, v1.depth, topToMiddleHeight),
                calculateSlope(v0.depth, v2.depth, topToBottomHeight),
                calculateSlope(v1.depth, v2.depth, middleToBottomHeight)
            };
        }

        static glm::mat3 createRotationMatrixX(float angle) {
            return glm::mat3(
                1.0f, 0.0f, 0.0f,
                0.0f, cos(angle), sin(angle),   
                0.0f, -sin(angle), cos(angle)   
            );
        }

        static glm::mat3 createRotationMatrixY(float angle) {
            return glm::mat3(
                cos(angle), 0.0f, sin(angle),
                0.0f, 1.0f, 0.0f,
                -sin(angle), 0.0f, cos(angle)
            );
        }
};

// ------------------------
// Camera System
// ------------------------
// (Camera class - defines view transformation)
class Camera {
    private:
        glm::vec3 position;
        glm::mat3 orientation;
        float pitch;
        float yaw;

        void updateOrientation() {
            glm::mat3 baseOrientation(
                1.0f, 0.0f, 0.0f,
                0.0f, -1.0f, 0.0f,
                0.0f, 0.0f, 1.0f
            );
            
            orientation = Math::createRotationMatrixX(pitch) * 
                        Math::createRotationMatrixY(yaw) * 
                        baseOrientation;
        }

    public:
        Camera() : 
            position(0.0f, 0.0f, 100.0f),
            orientation(1.0f, 0.0f, 0.0f,
                    0.0f, -1.0f, 0.0f,
                    0.0f, 0.0f, 1.0f),
            pitch(0.0f),
            yaw(0.0f) {}

        void translate(const glm::vec3& direction) {
            position += orientation * (direction * Config::Camera::TRANSLATION_STEP);
        }

        void rotate(float deltaPitch, float deltaYaw) {
            pitch += deltaPitch * Config::Camera::TILT_STEP;
            yaw += deltaYaw * Config::Camera::PAN_STEP;
            updateOrientation();
        }

        // Getters
        const glm::vec3& getPosition() const { return position; }
        const glm::mat3& getOrientation() const { return orientation; }
};


// ------------------------
// Geometry Processing
// ------------------------
// (Geometry class - handles 3D transformations)
class Geometry {
    public:
        static CanvasPoint projectVertex(const glm::vec3& vertex, float focalLength) {
            int u = static_cast<int>((focalLength * vertex.x) / (-vertex.z) + Config::Display::WIDTH / 2);
            int v = static_cast<int>((focalLength * vertex.y) / (-vertex.z) + Config::Display::HEIGHT / 2);
            CanvasPoint point(u, v);
            point.depth = -vertex.z;  // Store z-depth for depth buffering
            return point;
        }

        static std::vector<glm::vec3> transformToViewSpace(
            const ModelTriangle& worldTriangle,
            const Camera& camera
        ) {
            std::vector<glm::vec3> viewSpaceVertices;
            for (const auto& worldVertex : worldTriangle.vertices) {
                // Translate vertex relative to camera position
                glm::vec3 viewSpaceVertex = worldVertex - camera.getPosition();

                // Rotate vertex based on camera orientation matrix
                viewSpaceVertex = camera.getOrientation() * viewSpaceVertex;

                viewSpaceVertices.push_back(viewSpaceVertex);
            }
            return viewSpaceVertices;
        }

        static bool isTriangleVisible(const std::vector<glm::vec3>& viewSpaceVertices) {
            return viewSpaceVertices[0].z < 0 && 
                viewSpaceVertices[1].z < 0 && 
                viewSpaceVertices[2].z < 0;
        }
};

// ------------------------
// Drawing Primitives
// ------------------------
// (Draw class - low-level rendering operations)
class Draw {
    private:
        DrawingWindow& window;
        Config::Render& config;

        void fillTriangleHalf(
            float startY, float endY,
            float startX, float& x1, float& x2,
            float startZ, float& z1, float& z2,
            float slopeX1, float slopeX2,
            float slopeZ1, float slopeZ2,
            uint32_t color
        ) {
            // Only fill if the height is positive
            if (endY - startY > 0) {
                // Loop through each row of pixels top to bottom
                // Clamp Y coordinates to screen bounds
                for (int y = std::max(0, (int)startY); 
                    y < std::min(Config::Display::HEIGHT, (int)endY); 
                    y++) {
                    // Identify the start and end x-coordinates for the current row
                    // Clamp X coordinates to screen bounds
                    int xStart = std::max(0, (int)std::min(x1, x2));
                    int xEnd = std::min(Config::Display::WIDTH - 1, (int)std::max(x1, x2));

                    // Calculate interpolated z-values for the current scanline
                    float zStart = (x1 < x2) ? z1 : z2;
                    float zEnd = (x1 < x2) ? z2 : z1;
                    float zStep = (xEnd > xStart) ? (zEnd - zStart) / (xEnd - xStart) : 0;
                    float z = zStart;

                    // Fill pixels in current row with depth checking
                    for (int x = xStart; x <= xEnd; x++) {
                        int bufferIndex = y * Config::Display::WIDTH + x;
                        // Only draw pixel if it's closer than existing depth 
                        if (z < config.depthBuffer[bufferIndex]) {// Depth buffering in order to handle occlusion
                            config.depthBuffer[bufferIndex] = z;
                            window.setPixelColour(x, y, color);
                        }
                        z += zStep;
                    }

                    // Update edge coordinates and depths using slopes
                    x1 += slopeX1;
                    x2 += slopeX2;
                    z1 += slopeZ1;
                    z2 += slopeZ2;
                }
            }
        }

        uint32_t colourToARGB(const Colour& colour) {
            return (255 << 24) + (colour.red << 16) + 
                (colour.green << 8) + colour.blue;
        }

    public:
        Draw(DrawingWindow& window, Config::Render& config) 
            : window(window), config(config) {}

        void drawLine(
            CanvasPoint from, 
            CanvasPoint to, 
            Colour colour
        ) {
            float xDiff = to.x - from.x;
            float yDiff = to.y - from.y;

            // Calculate number of steps needed i.e. length of line (Pythagorean)
            int steps = ceil(sqrt(xDiff*xDiff + yDiff*yDiff));
            
            // Interpolated x and y coordinates along our line
            auto xValues = Math::interpolateSingleFloats(from.x, to.x, steps);
            auto yValues = Math::interpolateSingleFloats(from.y, to.y, steps);
            
            // Pack colour into ARGB format
            uint32_t ARGBColour = colourToARGB(colour);
            
            // Draw each pixel our the line
            for(int i = 0; i < steps; i++) {
                // Round interpolated values
                int x = round(xValues[i]);
                int y = round(yValues[i]);
                
                 // make sure we only draw pixel if within screen bounds
                if (x >= 0 && x < Config::Display::WIDTH && 
                    y >= 0 && y < Config::Display::HEIGHT) {
                    window.setPixelColour(x, y, ARGBColour);
                }
            }
        }

        void drawTriangle(
            CanvasPoint v0, 
            CanvasPoint v1, 
            CanvasPoint v2, 
            Colour colour
        ) {
            drawLine(v0, v1, colour);
            drawLine(v1, v2, colour);
            drawLine(v2, v0, colour);
        }

       void fillTriangle(CanvasPoint v0, CanvasPoint v1, CanvasPoint v2, Colour colour) {
            uint32_t ARGBColour = colourToARGB(colour);
            
            Math::sortVerticesByY(v0, v1, v2);
            Math::TriangleSlopes slopes = Math::calculateTriangleSlopes(v0, v1, v2);
            
            float x1 = v0.x;
            float x2 = v0.x;
            float z1 = v0.depth;
            float z2 = v0.depth;
            
            // Fill top half of triangle 
            // Using the coordinates and slopes for the top half of the triangle
            fillTriangleHalf(
                v0.y, v1.y, v0.x, x1, x2,
                v0.depth, z1, z2,
                slopes.topToMiddleX, slopes.topToBottomX,
                slopes.topToMiddleZ, slopes.topToBottomZ,
                ARGBColour
            );
            
            x1 = v1.x;
            z1 = v1.depth;

            // Fill bottom half of triangle 
            fillTriangleHalf(
                v1.y, v2.y, v1.x, x1, x2,
                v1.depth, z1, z2,
                slopes.middleToBottomX, slopes.topToBottomX,
                slopes.middleToBottomZ, slopes.topToBottomZ,
                ARGBColour
            );
        }
};

// ------------------------
// Rendering Pipeline
// ------------------------
// (Renderer class - orchestrates the rendering process)
class Renderer {
    private:
        DrawingWindow& window;
        Camera camera;
        Config::Render config;
        Draw drawer;

        struct ProjectedTriangle {
            CanvasPoint v0, v1, v2;
            
            ProjectedTriangle(
                const CanvasPoint& v0, 
                const CanvasPoint& v1, 
                const CanvasPoint& v2
            ) : v0(v0), v1(v1), v2(v2) {}
        };

        void renderTriangle(
            const ModelTriangle& worldTriangle,
            const Colour& color
        ) {
            // Transform thetriangle vertices from world space to camera space
            auto viewSpaceVertices = Geometry::transformToViewSpace(worldTriangle, camera);

            // Implement our simple back-face culling
            if (Geometry::isTriangleVisible(viewSpaceVertices)) {
                // Project 3D vertices to 2D screen using perspective projection function
                auto screenTriangle = ProjectedTriangle(
                    Geometry::projectVertex(viewSpaceVertices[0], Config::Display::FOCAL_LENGTH),
                    Geometry::projectVertex(viewSpaceVertices[1], Config::Display::FOCAL_LENGTH),
                    Geometry::projectVertex(viewSpaceVertices[2], Config::Display::FOCAL_LENGTH)
                );

                // Rasterise or render wireframe based on current mode
                if (config.currentMode == Config::Render::Mode::WIREFRAME) {
                    drawer.drawTriangle(
                        screenTriangle.v0, 
                        screenTriangle.v1, 
                        screenTriangle.v2, 
                        color
                    );
                } else {
                    drawer.fillTriangle(
                        screenTriangle.v0, 
                        screenTriangle.v1, 
                        screenTriangle.v2, 
                        color
                    );
                }
            }
        }

    public:
        Renderer(DrawingWindow& window) 
            : window(window),
              config(),  // Initialize config first
              drawer(window, config) {}  // Pass config to drawer

        void render(const std::vector<ModelTriangle>& modelTriangles) {
            window.clearPixels();
            config.clearDepthBuffer();
            
            for (const ModelTriangle& triangle : modelTriangles) {
                renderTriangle(triangle, Config::Colours::DEFAULT_MESH_COLOR);
            }
        }

        void toggleRenderMode() {
            config.currentMode = (config.currentMode == Config::Render::Mode::WIREFRAME)
                ? Config::Render::Mode::RASTERIZED
                : Config::Render::Mode::WIREFRAME;
        }

        void handleInput(SDL_Keycode key) {
            switch(key) {
                case SDLK_w: camera.translate(glm::vec3(0.0f, 0.0f, -1.0f)); break;
                case SDLK_s: camera.translate(glm::vec3(0.0f, 0.0f, 1.0f)); break;
                case SDLK_a: camera.translate(glm::vec3(-1.0f, 0.0f, 0.0f)); break;
                case SDLK_d: camera.translate(glm::vec3(1.0f, 0.0f, 0.0f)); break;
                case SDLK_LEFT: camera.rotate(0.0f, 1.0f); break;
                case SDLK_RIGHT: camera.rotate(0.0f, -1.0f); break;
                case SDLK_UP: camera.rotate(1.0f, 0.0f); break;
                case SDLK_DOWN: camera.rotate(-1.0f, 0.0f); break;
                case SDLK_r: toggleRenderMode(); break;
            }
        }
};

// ------------------------
// Model Loading
// ------------------------
// (ModelLoader class - utility for loading 3D models)
class ModelLoader {
    public:
        static std::vector<ModelTriangle> loadOBJ(
            const std::string& filename, 
            float scale
        ) {
            std::vector<ModelTriangle> triangles;
            std::vector<glm::vec3> vertices;
            
            // Open the OBJ file
            std::ifstream file(filename);
            if (!file.is_open()) {
                return triangles;
            }

            std::string line;
            Colour defaultColour(200, 200, 200);

            // Process file line by line
            while (std::getline(file, line)) {
                std::vector<std::string> tokens = split(line, ' ');
                if (tokens.empty()) continue;

                if (tokens[0] == "v") {
                    // Parse vertex coordinates and scale them
                    if (tokens.size() >= 4) {
                        float x = std::stof(tokens[1]) * scale;
                        float y = std::stof(tokens[2]) * scale;
                        float z = std::stof(tokens[3]) * scale;
                        vertices.push_back(glm::vec3(x, y, z));
                    }
                }
                
                else if (tokens[0] == "f") {
                    // Parse face indices (triangles)
                    if (tokens.size() >= 4) {
                        // Split vertex data
                        std::vector<std::string> v1 = split(tokens[1], '/');
                        std::vector<std::string> v2 = split(tokens[2], '/');
                        std::vector<std::string> v3 = split(tokens[3], '/');

                        // Convert from 1-based to 0-based indexing
                        int idx1 = std::stoi(v1[0]) - 1;
                        int idx2 = std::stoi(v2[0]) - 1;
                        int idx3 = std::stoi(v3[0]) - 1;

                        // Create and store triangle
                        ModelTriangle triangle(
                            vertices[idx1],
                            vertices[idx2],
                            vertices[idx3],
                            defaultColour
                        );
                        triangles.push_back(triangle);
                    }
                }
            }

            file.close();
            return triangles;
        }
};

// ------------------------
// Main Program
// ------------------------
// (main function - program entry point)
int main(int argc, char* argv[]) {
    DrawingWindow window(Config::Display::WIDTH, Config::Display::HEIGHT, false);
    Renderer renderer(window);
    
    std::vector<ModelTriangle> model = ModelLoader::loadOBJ("ident.obj", 0.5f);
    
    SDL_Event event;
    while (true) {
        if (window.pollForInputEvents(event)) {
            if (event.type == SDL_KEYDOWN) {
                renderer.handleInput(event.key.keysym.sym);
            }
        }
        
        renderer.render(model);
        window.renderFrame();
    }
}