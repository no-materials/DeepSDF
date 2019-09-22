#version 330 core

layout(location = 0) in vec3 vertex;
//layout(location = 2) in vec3 vertexNormal_model;

out vec4 position_world;
out vec4 position_camera;
out vec3 viewDirection_camera;
//out vec3 normal;

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;


void main(){
    // Projected image coordinate
    gl_Position =  P * V * M * vec4(vertex,1.0);
    // world coordinate location of the vertex
    position_world = vec4(vertex,1.0);
    position_camera = V * vec4(vertex, 1.0);
    viewDirection_camera = normalize(vec3(0.0,0.0,0.0) - position_camera.xyz);
}