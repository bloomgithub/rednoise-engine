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

// Window dimensions
#define WIDTH 320
#define HEIGHT 240

std::vector<float> depthBuffer(WIDTH * HEIGHT, 0.0f);

enum RenderMode {
    WIREFRAME,
    RASTERIZED
};

namespace Camera {
    glm::vec3 position(0.0f, 0.0f, 100.0f);
    glm::mat3 orientation(
        1.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f,  // Invert Y axis in initial orientation
        0.0f, 0.0f, 1.0f
    );

    const float TRANSLATION_STEP = 1.0f;
    const float PAN_STEP = 0.01f;
    const float TILT_STEP = 0.01f;

    // Track cumulative angles
    float pitch = 0.0f;
    float yaw = 0.0f;

    RenderMode currentRenderMode = RASTERIZED;
}

std::vector<ModelTriangle> loadOBJ(const std::string& objFilename, float scale) {
    std::vector<ModelTriangle> triangles;
    std::vector<glm::vec3> vertices;
    
    std::ifstream file(objFilename);
    if (!file.is_open()) {
        std::cout << "Error: Could not open OBJ file: " << objFilename << std::endl;
        return triangles;
    }

    std::string line;
    // Default color for all triangles - you can modify this or make it random
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

                // Convert to 0-based index
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

CanvasPoint projectVertex(const glm::vec3& vertex, float focalLength) {
    int u = static_cast<int>((focalLength * vertex.x) / (-vertex.z) + WIDTH / 2);
    int v = static_cast<int>((focalLength * vertex.y) / (-vertex.z) + HEIGHT / 2);
    return CanvasPoint(u, v);
}

void drawLine(DrawingWindow& window, CanvasPoint from, CanvasPoint to, Colour colour) {
    float xDiff = to.x - from.x;
    float yDiff = to.y - from.y;
    float steps = std::max(abs(xDiff), abs(yDiff));
    
    float xStepSize = xDiff / steps;
    float yStepSize = yDiff / steps;

    uint32_t colourValue = (255 << 24) + (colour.red << 16) + (colour.green << 8) + colour.blue;
    
    for (float i = 0; i <= steps; i++) {
        int x = round(from.x + (xStepSize * i));
        int y = round(from.y + (yStepSize * i));
        if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
            window.setPixelColour(x, y, colourValue);
        }
    }
}

void drawTriangle(DrawingWindow& window, CanvasPoint v0, CanvasPoint v1, CanvasPoint v2, Colour colour) {
    drawLine(window, v0, v1, colour);
    drawLine(window, v1, v2, colour);
    drawLine(window, v2, v0, colour);
}

void drawWireframeModel(DrawingWindow& window, const std::vector<ModelTriangle>& triangles) {
    window.clearPixels();
    float focalLength = 500.0f;

    for (const auto& triangle : triangles) {
        std::vector<glm::vec3> cameraSpace;
        for (int i = 0; i < 3; i++) {
            glm::vec3 vertexRelativeToCamera = triangle.vertices[i] - Camera::position;
            vertexRelativeToCamera = Camera::orientation * vertexRelativeToCamera;
            cameraSpace.push_back(vertexRelativeToCamera);
        }

        if (cameraSpace[0].z < 0 && cameraSpace[1].z < 0 && cameraSpace[2].z < 0) {
            CanvasPoint p0 = projectVertex(cameraSpace[0], focalLength);
            CanvasPoint p1 = projectVertex(cameraSpace[1], focalLength);
            CanvasPoint p2 = projectVertex(cameraSpace[2], focalLength);

            // Draw wireframe in white
            drawTriangle(window, p0, p1, p2, triangle.colour);
        }
    }
}

void fillTriangle(DrawingWindow& window, CanvasPoint v0, CanvasPoint v1, CanvasPoint v2, Colour colour) {
    // Convert colour once instead of per pixel
    uint32_t colourValue = (255 << 24) + (colour.red << 16) + (colour.green << 8) + colour.blue;
    
    // Sort vertices by y coordinate
    if (v0.y > v1.y) std::swap(v0, v1);
    if (v0.y > v2.y) std::swap(v0, v2);
    if (v1.y > v2.y) std::swap(v1, v2);

    // Calculate slopes
    float dxdy_1 = 0, dxdy_2 = 0, dxdy_3 = 0;
    
    if (v1.y - v0.y > 0)
        dxdy_1 = (v1.x - v0.x) / (v1.y - v0.y);
    if (v2.y - v0.y > 0)
        dxdy_2 = (v2.x - v0.x) / (v2.y - v0.y);
    if (v2.y - v1.y > 0)
        dxdy_3 = (v2.x - v1.x) / (v2.y - v1.y);

    // Get starting x coordinates
    float x1 = v0.x;
    float x2 = v0.x;

    // First half of the triangle
    if (v1.y - v0.y > 0) {
        for (int y = std::max(0, (int)v0.y); y < std::min(HEIGHT, (int)v1.y); y++) {
            int startX = std::max(0, (int)std::min(x1, x2));
            int endX = std::min(WIDTH - 1, (int)std::max(x1, x2));

            // Draw horizontal line
            for (int x = startX; x <= endX; x++) {
                window.setPixelColour(x, y, colourValue);
            }

            x1 += dxdy_1;
            x2 += dxdy_2;
        }
    }

    // Second half of the triangle
    x1 = v1.x;
    if (v2.y - v1.y > 0) {
        for (int y = std::max(0, (int)v1.y); y < std::min(HEIGHT, (int)v2.y); y++) {
            int startX = std::max(0, (int)std::min(x1, x2));
            int endX = std::min(WIDTH - 1, (int)std::max(x1, x2));

            // Draw horizontal line
            for (int x = startX; x <= endX; x++) {
                window.setPixelColour(x, y, colourValue);
            }

            x1 += dxdy_3;
            x2 += dxdy_2;
        }
    }
}

void drawRasterizedModel(DrawingWindow& window, const std::vector<ModelTriangle>& triangles) {
    window.clearPixels();
    float focalLength = 500.0f;
    
    Colour silverColour(192, 192, 192);

    for (const auto& triangle : triangles) {
        std::vector<glm::vec3> cameraSpace;
        for (int i = 0; i < 3; i++) {
            glm::vec3 vertexRelativeToCamera = triangle.vertices[i] - Camera::position;
            vertexRelativeToCamera = Camera::orientation * vertexRelativeToCamera;
            cameraSpace.push_back(vertexRelativeToCamera);
        }

        if (cameraSpace[0].z < 0 && cameraSpace[1].z < 0 && cameraSpace[2].z < 0) {
            CanvasPoint p0 = projectVertex(cameraSpace[0], focalLength);
            CanvasPoint p1 = projectVertex(cameraSpace[1], focalLength);
            CanvasPoint p2 = projectVertex(cameraSpace[2], focalLength);

            // Fill triangle with silver color
            fillTriangle(window, p0, p1, p2, silverColour);
        }
    }
}

glm::mat3 createRotationMatrixX(float angle) {
    return glm::mat3(
        1.0f, 0.0f, 0.0f,
        0.0f, cos(angle), sin(angle),   
        0.0f, -sin(angle), cos(angle)   
    );
}

glm::mat3 createRotationMatrixY(float angle) {
    return glm::mat3(
        cos(angle), 0.0f, sin(angle),
        0.0f, 1.0f, 0.0f,
        -sin(angle), 0.0f, cos(angle)
    );
}

void updateCameraOrientation() {
    // Start with the base orientation that has Y inverted
    glm::mat3 baseOrientation(
        1.0f, 0.0f, 0.0f,
        0.0f, -1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    );
    
    // Apply rotations in order: yaw then pitch
    glm::mat3 rotationY = createRotationMatrixY(Camera::yaw);
    glm::mat3 rotationX = createRotationMatrixX(Camera::pitch);
    
    // Combine all transformations
    Camera::orientation = rotationX * rotationY * baseOrientation;
}

void drawModel(DrawingWindow& window, const std::vector<ModelTriangle>& triangles) {
    if (Camera::currentRenderMode == WIREFRAME) {
        drawWireframeModel(window, triangles);
    } else {
        drawRasterizedModel(window, triangles);
    }
}

void handleEvent(SDL_Event event, DrawingWindow &window) {
    if (event.type == SDL_KEYDOWN) {
        switch(event.key.keysym.sym) {
            // Camera Rotations
            case SDLK_LEFT: // Pan left
                Camera::yaw += Camera::PAN_STEP;
                updateCameraOrientation();
                break;
                
            case SDLK_RIGHT: // Pan right
                Camera::yaw -= Camera::PAN_STEP;
                updateCameraOrientation();
                break;
                
            case SDLK_UP: // Tilt up
                Camera::pitch += Camera::TILT_STEP;
                updateCameraOrientation();
                break;
                
            case SDLK_DOWN: // Tilt down
                Camera::pitch -= Camera::TILT_STEP;
                updateCameraOrientation();
                break;

            // Camera Translations
            case SDLK_w: { // Move forward
                glm::vec3 forward(0.0f, 0.0f, -Camera::TRANSLATION_STEP);
                glm::vec3 movement = Camera::orientation * forward;
                Camera::position += movement;
                break;
            }
            case SDLK_s: { // Move backward
                glm::vec3 backward(0.0f, 0.0f, Camera::TRANSLATION_STEP);
                glm::vec3 movement = Camera::orientation * backward;
                Camera::position += movement;
                break;
            }
            case SDLK_a: { // Move left
                glm::vec3 left(-Camera::TRANSLATION_STEP, 0.0f, 0.0f);
                glm::vec3 movement = Camera::orientation * left;
                Camera::position += movement;
                break;
            }
            case SDLK_d: { // Move right
                glm::vec3 right(Camera::TRANSLATION_STEP, 0.0f, 0.0f);
                glm::vec3 movement = Camera::orientation * right;
                Camera::position += movement;
                break;
            }

            case SDLK_r:
                Camera::currentRenderMode = (Camera::currentRenderMode == WIREFRAME) ? 
                                          RASTERIZED : WIREFRAME;
                std::cout << "Switched to " 
                         << (Camera::currentRenderMode == WIREFRAME ? "wireframe" : "rasterized") 
                         << " mode" << std::endl;
                break;
        }
    }
}

int main(int argc, char* argv[]) {
    DrawingWindow window = DrawingWindow(WIDTH, HEIGHT, false);
    SDL_Event event;

    // Load the 3D model
    std::string objFile = "ident.obj";
    float scale = 0.5f; // Adjust this to make model bigger/smaller

    std::vector<ModelTriangle> triangles = loadOBJ(objFile, scale);

    while (true) {
        // We need to render the frame at each iteration
        if (window.pollForInputEvents(event)) handleEvent(event, window);

        drawModel(window, triangles);
        
        // Need to render the frame at each iteration
        window.renderFrame();
    }
}
