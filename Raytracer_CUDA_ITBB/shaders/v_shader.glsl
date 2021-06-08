#version 420 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;

out vec2 texCoord;

void main()
{
    texCoord = aUV;
    gl_Position = vec4(aPos, 0.0f, 1.0f);
}