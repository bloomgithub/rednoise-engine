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
        
        Render() : depthBuffer(Display::WIDTH * Display::HEIGHT, 0.0f) {}
    };
}

// ------------------------
// Math Utilities
// ------------------------
// (Math class - fundamental operations used by other components)
class Math {
    public:
        struct ProjectedTriangle {
            CanvasPoint v0, v1, v2;
            
            ProjectedTriangle(
                const CanvasPoint& v0, 
                const CanvasPoint& v1, 
                const CanvasPoint& v2
            ) : v0(v0), v1(v1), v2(v2) {}
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
            return CanvasPoint(u, v);
        }

        static std::vector<glm::vec3> transformToViewSpace(
            const ModelTriangle& worldTriangle,
            const Camera& camera
        ) {
            std::vector<glm::vec3> viewSpaceVertices;
            for (const auto& worldVertex : worldTriangle.vertices) {
                glm::vec3 viewSpaceVertex = worldVertex - camera.getPosition();
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

        void sortVertices(
            CanvasPoint& v0, 
            CanvasPoint& v1, 
            CanvasPoint& v2
        ) {
            if (v0.y > v1.y) std::swap(v0, v1);
            if (v0.y > v2.y) std::swap(v0, v2);
            if (v1.y > v2.y) std::swap(v1, v2);
        }

        float calculateSlope(
            const CanvasPoint& start, 
            const CanvasPoint& end
        ) {
        if (end.y - start.y > 0) {
            return (end.x - start.x) / (end.y - start.y);
        }
        return 0;
        }

        struct TriangleSlopes {
            float topToMiddle;
            float topToBottom;
            float middleToBottom;
        };

        TriangleSlopes calculate3Slopes(
            const CanvasPoint& v0, 
            const CanvasPoint& v1, 
            const CanvasPoint& v2
        ) {
        return {
            calculateSlope(v0, v1),
            calculateSlope(v0, v2),
            calculateSlope(v1, v2)
        };
        }

        void fillTriangleHalf(
            float startY, float endY,
            float startX, float& x1, float& x2,
            float slope1, float slope2,
            uint32_t color
        ) {
            if (endY - startY > 0) {
                for (int y = std::max(0, (int)startY); 
                    y < std::min(Config::Display::HEIGHT, (int)endY); 
                    y++) {
                    int startX = std::max(0, (int)std::min(x1, x2));
                    int endX = std::min(Config::Display::WIDTH - 1, (int)std::max(x1, x2));

                    for (int x = startX; x <= endX; x++) {
                        window.setPixelColour(x, y, color);
                    }

                    x1 += slope1;
                    x2 += slope2;
                }
            }
        }

        uint32_t colourToARGB(const Colour& colour) {
            return (255 << 24) + (colour.red << 16) + 
                (colour.green << 8) + colour.blue;
        }

    public:
        Draw(DrawingWindow& window) : window(window) {}

        void drawLine(
            CanvasPoint from, 
            CanvasPoint to, 
            Colour colour
        ) {
            float xDiff = to.x - from.x;
            float yDiff = to.y - from.y;
            int steps = ceil(sqrt(xDiff*xDiff + yDiff*yDiff));
            
            auto xValues = Math::interpolateSingleFloats(from.x, to.x, steps);
            auto yValues = Math::interpolateSingleFloats(from.y, to.y, steps);
            
            uint32_t ARGBColour = colourToARGB(colour);
            
            for(int i = 0; i < steps; i++) {
                int x = round(xValues[i]);
                int y = round(yValues[i]);
                
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

        void fillTriangle(
            CanvasPoint v0, 
            CanvasPoint v1, 
            CanvasPoint v2, 
            Colour colour
        ) {
            uint32_t ARGBColour = colourToARGB(colour);
            
            sortVertices(v0, v1, v2);
            TriangleSlopes slopes = calculate3Slopes(v0, v1, v2);

            float x1 = v0.x;
            float x2 = v0.x;

            fillTriangleHalf(v0.y, v1.y, v0.x, x1, x2, 
                            slopes.topToMiddle, slopes.topToBottom, ARGBColour);
            
            x1 = v1.x;
            
            fillTriangleHalf(v1.y, v2.y, v1.x, x1, x2, 
                            slopes.middleToBottom, slopes.topToBottom, ARGBColour);
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

        void renderTriangle(
            const ModelTriangle& worldTriangle,
            const Colour& color
        ) {
            auto viewSpaceVertices = Geometry::transformToViewSpace(worldTriangle, camera);

            if (Geometry::isTriangleVisible(viewSpaceVertices)) {
                auto screenTriangle = Math::ProjectedTriangle(
                    Geometry::projectVertex(viewSpaceVertices[0], Config::Display::FOCAL_LENGTH),
                    Geometry::projectVertex(viewSpaceVertices[1], Config::Display::FOCAL_LENGTH),
                    Geometry::projectVertex(viewSpaceVertices[2], Config::Display::FOCAL_LENGTH)
                );

                // Use drawer methods directly instead of function pointer
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
            drawer(window) {}

        void render(const std::vector<ModelTriangle>& modelTriangles) {
            window.clearPixels();
            
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
        static std::vector<ModelTriangle> loadOBJ(const std::string& filename, float scale) {
            std::vector<ModelTriangle> triangles;
            std::vector<glm::vec3> vertices;
            
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cout << "Error: Could not open OBJ file: " << filename << std::endl;
                return triangles;
            }

            std::string line;
            Colour defaultColour(200, 200, 200);

            while (std::getline(file, line)) {
                std::vector<std::string> tokens = split(line, ' ');
                if (tokens.empty()) continue;

                if (tokens[0] == "v") {
                    // Parse vertex
                    if (tokens.size() >= 4) {
                        float x = std::stof(tokens[1]) * scale;
                        float y = std::stof(tokens[2]) * scale;
                        float z = std::stof(tokens[3]) * scale;
                        vertices.push_back(glm::vec3(x, y, z));
                    }
                }
                else if (tokens[0] == "f") {
                    // Parse face (triangle)
                    if (tokens.size() >= 4) {
                        std::vector<std::string> v1 = split(tokens[1], '/');
                        std::vector<std::string> v2 = split(tokens[2], '/');
                        std::vector<std::string> v3 = split(tokens[3], '/');

                        int idx1 = std::stoi(v1[0]) - 1;
                        int idx2 = std::stoi(v2[0]) - 1;
                        int idx3 = std::stoi(v3[0]) - 1;

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