#version 420 core

uniform sampler2D screenTexture;

layout(location = 0) in vec2 texCoord;

out vec4 FragColor;

void main()
{
    FragColor = vec4(texture(screenTexture, texCoord).rgb, 1.0f);
}