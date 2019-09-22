#version 330

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec4 position_world[];
in vec3 viewDirection_camera[];

out vec3 normal_camera;
out vec3 normal_world;
out vec4 xyz_world;
out vec3 viewDirection_cam;
out vec4 xyz_camera;

uniform mat4 V;

void main()
{
    vec3 A = position_world[1].xyz - position_world[0].xyz;
    vec3 B = position_world[2].xyz - position_world[0].xyz;
    vec3 normal = normalize(cross(A, B));
    vec3 normal_cam = (V * vec4(normal, 0.0)).xyz;

    gl_Position = gl_in[0].gl_Position;
    normal_camera = normal_cam;
    normal_world = normal;
    xyz_world = position_world[0];
    xyz_camera = V * xyz_world;
    viewDirection_cam = viewDirection_camera[0];
    gl_PrimitiveID = gl_PrimitiveIDIn;
    EmitVertex();

    gl_Position = gl_in[1].gl_Position;
    normal_camera = normal_cam;
    normal_world = normal;
    xyz_world = position_world[1];
    xyz_camera = V * xyz_world;
    viewDirection_cam = viewDirection_camera[1];
    gl_PrimitiveID = gl_PrimitiveIDIn;
    EmitVertex();

    gl_Position = gl_in[2].gl_Position;
    normal_camera = normal_cam;
    normal_world = normal;
    xyz_world = position_world[2];
    xyz_camera = V * xyz_world;
    viewDirection_cam = viewDirection_camera[2];
    gl_PrimitiveID = gl_PrimitiveIDIn;
    EmitVertex();

    EndPrimitive();
}