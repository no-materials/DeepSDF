#version 330 core

in vec3 normal_world;
in vec3 normal_camera;
in vec4 xyz_world;

uniform mat4 V;

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 FragColor2;
layout(location = 2) out vec4 FragColor3;
layout(location = 3) out vec4 FragColor4;

void main(){
    //vec3 view_vector = normalize(vec3(0,0,1) - xyz_camera.xyz);
    vec3 view_vector = vec3(0.0, 0.0, 1.0);
    vec4 test = vec4(0.0, 0.0, 0.0, 1.0);
    // Check if we need to flip the normal.
    vec3 normal_world_cor;// = normal_world;
    float d = dot(normalize(normal_camera), normalize(view_vector));
    if (abs(d) < 0.001) {
        FragColor = vec4(0.0, 0.0, 0.0, 0.0);
        FragColor2 = vec4(0.0, 0.0, 0.0, 0.0);
        FragColor3 = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    else {
        if (d < 0) {
            test = vec4(0.0, 1.0, 0.0, 1.0);
            normal_world_cor = -normal_world;
        } else {
            normal_world_cor= normal_world;
        }
        FragColor = xyz_world;
        FragColor.w = gl_PrimitiveID + 1.0f;

        FragColor2 = vec4(normalize(normal_world_cor), 1.0);
        FragColor2.w = gl_PrimitiveID + 1.0f;
    }
}