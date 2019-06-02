#version 450 core

layout (location = 0) in vec4 inPos;

out vec4 simPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform sampler3D fluid;

void main()
{
    simPos = inPos;
    gl_Position = inPos*vec4(1,1,0,0) + vec4(0,0,0,1);
}
