# RedNoise 3D Computer Graphics Engine

A C++ rendering engine for the creation of promotional ident videos. The engine supports wireframe, rasterisation, and raytracing rendering modes with added lighting, physics simulation, and material systems.

## Setup
1. Clone the repository:
```bash
git clone https://github.com/bloomgithub/rednoise-engine.git
```

2. Navigate to the RedNoise directory:
```bash
cd rednoise-engine
```

3. Compile the engine:
```bash
make
```

## Requirements
- A texture file is required for the engine to work. The texture file should be referenced in a .mtl file that is linked to your .obj model.
- The default model is set to "logo.obj" but can be changed in the code.

## Warning ⚠️
The engine will fail to launch if no texture file is present and properly linked through the MTL file. Ensure your 3D model includes texture mapping information.

## Controls

### Camera Movement
- `W/S`: Move camera forward/backward
- `A/D`: Move camera left/right
- `Q/E`: Move camera up/down
- `Arrow Keys`: Rotate camera
- `Shift + Any Movement`: Move/rotate faster

### Mouse Controls
- `Left Click + Drag`: Rotate model

### Rendering Modes
- `SPACE`: Cycle through basic rendering modes:
  - Wireframe
  - Flat shading
  - Depth buffer
  - Diffuse lighting
  - Specular lighting
  - Texture mapping
  - Gouraud shading
  - Phong shading

### Advanced Rendering Modes
- `H`: Hard shadow ray tracing
- `S`: Soft shadow ray tracing

### Animations
- `O`: Toggle complex animation (physics + orbit)
- `P`: Toggle simple orbit animation

## Initial Launch
Upon launching, the engine will automatically start with a complex animation showing physics and orbital movement. Once this animation completes, you can use any of the above controls to interact with the model.

## Performance Note
Ray tracing modes (Hard shadows, Soft shadows, and Mirror reflections) are computationally intensive and may run slower than the standard rendering modes.

## License
This project is licensed under the MIT License.
