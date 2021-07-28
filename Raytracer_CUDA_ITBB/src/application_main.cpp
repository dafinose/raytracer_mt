#include <glad/glad.h>
#include <glfw3.h>

#include <iostream>
#include "../header/cuda_render.cuh"
#include <cuda_gl_interop.h>
#include "../header/shader.h"
#include "../header/itbb_render.h"
#include <Windows.h>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
unsigned int SCR_WIDTH = 800;
unsigned int SCR_HEIGHT = 600;

int i = 0;

bool isRenderingCuda = true;
int number_of_samples = 10;

int main()
{
	SCR_WIDTH = GetPrivateProfileIntA("Settings", "SCR_WIDTH", 800, "./settings.ini");
	SCR_HEIGHT = GetPrivateProfileIntA("Settings", "SCR_HEIGHT", 600, "./settings.ini");
	isRenderingCuda = GetPrivateProfileIntA("Settings", "renderCuda", 1, "./settings.ini");
	number_of_samples = GetPrivateProfileIntA("Settings", "samples", 10, "./settings.ini");
	
	std::cout << "render settings:" << std::endl;
	std::cout << "width: " << SCR_WIDTH << std::endl;
	std::cout << "height: " << SCR_HEIGHT << std::endl;
	std::cout << "rendering with cuda: " << isRenderingCuda << std::endl;
	std::cout << "samples: " << number_of_samples << std::endl;

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Raytracing Comparison", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	shader shaderProgram = shader("./shaders/v_shader.glsl", "./shaders/f_shader.glsl");

	float2* screenSpaceQuadVertices = (float2*)malloc(sizeof(float2) * 4);
	screenSpaceQuadVertices[0] = make_float2(-1.0f, -1.0f);
	screenSpaceQuadVertices[1] = make_float2(1.0f, -1.0f);
	screenSpaceQuadVertices[2] = make_float2(-1.0f, 1.0f);
	screenSpaceQuadVertices[3] = make_float2(1.0f, 1.0f);

	float2* screenSpaceQuadUV = (float2*)malloc(sizeof(float2) * 4);
	screenSpaceQuadUV[0] = make_float2(0.0f, 0.0f);
	screenSpaceQuadUV[1] = make_float2(1.0f, 0.0f);
	screenSpaceQuadUV[2] = make_float2(0.0f, 1.0f);
	screenSpaceQuadUV[3] = make_float2(1.0f, 1.0f);


	// Bufferobjekte
	unsigned int screenQuadVBO, screenQuadVAO, screenUV_VBO, textureBuffer;
	glGenVertexArrays(1, &screenQuadVAO);

	glGenBuffers(1, &screenQuadVBO);
	glGenBuffers(1, &screenUV_VBO);
	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(screenQuadVAO);

	glBindBuffer(GL_ARRAY_BUFFER, screenQuadVBO);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float2), screenSpaceQuadVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float2), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, screenUV_VBO);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float2), screenSpaceQuadUV, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float2), (void*)0);
	glEnableVertexAttribArray(1);

	// note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	unsigned int screenTexture;

	// Textur binden
	glGenTextures(1, &screenTexture);
	glBindTexture(GL_TEXTURE_2D, screenTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	void* fb = nullptr;

	// ITBB Render

	// For CUDA: NULL for ITBB: fb

	if (isRenderingCuda)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
	}
	// ITBB Render

	glBindTexture(GL_TEXTURE_2D, 0);


	cudaGraphicsResource* cuda_Resource = nullptr;
	// CUDA
	if (isRenderingCuda)
	{
		glGenBuffers(1, &textureBuffer);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, textureBuffer);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, SCR_WIDTH * SCR_HEIGHT * sizeof(vec3), 0, GL_DYNAMIC_COPY);

		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_Resource, textureBuffer, cudaGraphicsRegisterFlagsNone));
	}

	// CUDA

	// render loop
	// -----------
	while (!glfwWindowShouldClose(window))
	{
		// input
		// -----
		processInput(window);

		// render
		// ------
		//glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		//glClear(GL_COLOR_BUFFER_BIT);


		if (i < 1) {
			i++;
			if (isRenderingCuda)
			{
				cuda_main(cuda_Resource, SCR_WIDTH, SCR_HEIGHT, number_of_samples);
			}
			else
			{
				itbb_main(&fb, SCR_WIDTH, SCR_HEIGHT, number_of_samples);
			}
		}

		// CUDA
		glBindTexture(GL_TEXTURE_2D, screenTexture);
		// CUDA
		// Daten in die Textur schreiben
		if (isRenderingCuda)
		{
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, textureBuffer);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGB, GL_FLOAT, 0);
		}
		else
		{
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, fb);
		}

		shaderProgram.useShader();

		glUniform1i(glGetUniformLocation(shaderProgram.ID, "screenTexture"), 0);


		//Drawcall
		glBindVertexArray(screenQuadVAO);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glBindVertexArray(0);

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();

	return 0;
}

void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}