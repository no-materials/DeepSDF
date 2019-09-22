#include <glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include "camera.h"

using namespace glm;

Camera::Camera(GLFWwindow* window) : window(window) {
    position = vec3(0, 0, 5);
    horizontalAngle = 3.14f;
    verticalAngle = 0.0f;
    FoV = 45.0f;
    speed = 3.0f;
    mouseSpeed = 0.001f;
    fovSpeed = 2.0f;
}

void Camera::update() {
    // glfwGetTime is called only once, the first time this function is called
    static double lastTime = glfwGetTime();

    // Compute time difference between current and last frame
    double currentTime = glfwGetTime();
    float deltaTime = float(currentTime - lastTime);

    // Get mouse position
    double xPos, yPos;
    glfwGetCursorPos(window, &xPos, &yPos);

    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // Reset mouse position for next frame
    glfwSetCursorPos(window, width / 2, height / 2);

    // Task 5.1: simple camera movement that moves in +-z and +-x axes

//    // Move forward
//    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
//        position -= vec3(0, 0, 1) * deltaTime * speed;
//    }
//    // Move backward
//    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
//        position += vec3(0, 0, 1) * deltaTime * speed;
//    }
//    // Strafe right
//    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
//        position += vec3(1, 0, 0) * deltaTime * speed;
//    }
//    // Strafe left
//    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
//        position -= vec3(1, 0, 0) * deltaTime * speed;
//    }
//
//    // Task 5.2: update view matrix so it always looks at the origin
//    projectionMatrix = perspective(radians(FoV), 4.0f / 3.0f, 0.1f, 100.0f);
//    viewMatrix = lookAt(
//        position,
//        vec3(0, 0, 0),
//        vec3(0, 1, 0)
//    );

    // Task 5.3: Compute new horizontal and vertical angles, given windows size
    //*
    // and cursor position
    horizontalAngle += mouseSpeed * float(width / 2 - xPos);
    verticalAngle += mouseSpeed * float(height / 2 - yPos);

    // Task 5.4: right and up vectors of the camera coordinate system
    // use spherical coordinates
    vec3 direction(
        cos(verticalAngle) * sin(horizontalAngle),
        sin(verticalAngle),
        cos(verticalAngle) * cos(horizontalAngle)
    );

    // Right vector
    vec3 right(
        sin(horizontalAngle - 3.14f / 2.0f),
        0,
        cos(horizontalAngle - 3.14f / 2.0f)
    );

    // Up vector
    vec3 up = cross(right, direction);

    // Task 5.5: update camera position using the direction/right vectors
    // Move forward
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        position += direction * deltaTime * speed;
    }
    // Move backward
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        position -= direction * deltaTime * speed;
    }
    // Strafe right
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        position += right * deltaTime * speed;
    }
    // Strafe left
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        position -= right * deltaTime * speed;
    }

    // Task 5.6: handle zoom in/out effects
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        FoV -= fovSpeed;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        FoV += fovSpeed;
    }

    // Task 5.7: construct projection and view matrices
    projectionMatrix = perspective(radians(FoV), 4.0f / 3.0f, 0.1f, 100.0f);
    viewMatrix = lookAt(
        position,
        position + direction,
        up
    );
    //

    // Homework XX: perform orthographic projection

    // For the next frame, the "last time" will be "now"
    lastTime = currentTime;
}
