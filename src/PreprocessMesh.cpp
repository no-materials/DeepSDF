// Copyright 2004-present Facebook. All Rights Reserved.

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define GLEW_STATIC

#include <GL/glew.h>

#include <glfw3.h>

#include <pangolin/geometry/geometry.h>
#include <pangolin/geometry/glgeometry.h>
#include <pangolin/gl/gl.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>
#include <cnpy.h>
#include <glm/ext/matrix_transform.hpp>

#include "Utils.h"
#include "common/util.h"
#include "common/camera.h"
#include "common/shader.h"
#include "common/model.h"
#include "common/tiny_obj_loader.h"


// Global variables
int windowWidth, windowHeight;
std::string windowTitle;
GLFWwindow *window;
Camera *camera;
GLuint shaderProgram;
GLint projectionMatrixLocation, viewMatrixLocation, modelMatrixLocation;
GLuint objVAO;
GLuint objVerticiesVBO;
GLuint framebuffer;
GLuint texVertBuffer, texNormalBuffer, rbo;

extern pangolin::GlSlProgram GetShaderProgram();

void SampleFromSurface(
        pangolin::Geometry &geom,
        std::vector<Eigen::Vector3f> &surfpts,
        int num_sample) {
    float total_area = 0.0f;

    std::vector<float> cdf_by_area;

    std::vector<Eigen::Vector3i> linearized_faces;

    for (const auto &object : geom.objects) {
        auto it_vert_indices = object.second.attributes.find("vertex_indices");
        if (it_vert_indices != object.second.attributes.end()) {
            pangolin::Image<uint32_t> ibo =
                    pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

            for (int i = 0; i < ibo.h; ++i) {
                linearized_faces.emplace_back(ibo(0, i), ibo(1, i), ibo(2, i));
            }
        }
    }

    pangolin::Image<float> vertices =
            pangolin::get<pangolin::Image<float>>(geom.buffers["geometry"].attributes["vertex"]);

    for (const Eigen::Vector3i &face : linearized_faces) {
        float area = TriangleArea(
                (Eigen::Vector3f) Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(0))),
                (Eigen::Vector3f) Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(1))),
                (Eigen::Vector3f) Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(2))));

        if (std::isnan(area)) {
            area = 0.f;
        }

        total_area += area;

        if (cdf_by_area.empty()) {
            cdf_by_area.push_back(area);

        } else {
            cdf_by_area.push_back(cdf_by_area.back() + area);
        }
    }

    std::random_device seeder;
    std::mt19937 generator(seeder());
    std::uniform_real_distribution<float> rand_dist(0.0, total_area);

    while ((int) surfpts.size() < num_sample) {
        float tri_sample = rand_dist(generator);
        std::vector<float>::iterator tri_index_iter =
                lower_bound(cdf_by_area.begin(), cdf_by_area.end(), tri_sample);
        int tri_index = tri_index_iter - cdf_by_area.begin();

        const Eigen::Vector3i &face = linearized_faces[tri_index];

        surfpts.push_back(SamplePointFromTriangle(
                Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(0))),
                Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(1))),
                Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(2)))));
    }
}

void TinySampleFromSurface(
        tinyobj::attrib_t attrib,
        std::vector<tinyobj::shape_t> shapes,
        std::vector<Eigen::Vector3f> &surfpts,
        int num_sample) {
    float total_area = 0.0f;

    std::vector<float> cdf_by_area;

    std::vector<Eigen::Vector3f> faceVerts;

    // Loop over shapes
    for (auto &shape : shapes) {

        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            int fv = shape.mesh.num_face_vertices[f];

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
                faceVerts.emplace_back(vx, vy, vz);
            }
            float area = TriangleArea(faceVerts[0], faceVerts[1], faceVerts[2]);

            if (std::isnan(area)) {
                area = 0.f;
            }

            total_area += area;

            if (cdf_by_area.empty()) {
                cdf_by_area.push_back(area);
            } else {
                cdf_by_area.push_back(cdf_by_area.back() + area);
            }

            index_offset += fv;
        }
    }


    std::random_device seeder;
    std::mt19937 generator(seeder());
    std::uniform_real_distribution<float> rand_dist(0.0, total_area);

    while ((int) surfpts.size() < num_sample) {
        float tri_sample = rand_dist(generator);
        auto tri_index_iter = lower_bound(cdf_by_area.begin(), cdf_by_area.end(), tri_sample);
        int tri_index = tri_index_iter - cdf_by_area.begin();

        const Eigen::Vector3f &face = faceVerts[tri_index];
        // TODO
        surfpts.push_back(
                SamplePointFromTriangle(faceVerts[tri_index], faceVerts[tri_index + 1], faceVerts[tri_index + 2])
        );
    }
}

void SampleSDFNearSurface(
        KdVertexListTree &kdTree,
        std::vector<Eigen::Vector3f> &vertices,
        std::vector<Eigen::Vector3f> &xyz_surf,
        std::vector<Eigen::Vector3f> &normals,
        std::vector<Eigen::Vector3f> &xyz,
        std::vector<float> &sdfs,
        int num_rand_samples,
        float variance,
        float second_variance,
        float bounding_cube_dim,
        int num_votes) {
    float stdv = sqrt(variance);

    std::random_device seeder;
    std::mt19937 generator(seeder());
    std::uniform_real_distribution<float> rand_dist(0.0, 1.0);
    std::vector<Eigen::Vector3f> xyz_used;
    std::vector<Eigen::Vector3f> second_samples;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> vert_ind(0, vertices.size() - 1);
    std::normal_distribution<float> perterb_norm(0, stdv);
    std::normal_distribution<float> perterb_second(0, sqrt(second_variance));

    // Perturbation of two surface samples per surface point
    for (unsigned int i = 0; i < xyz_surf.size(); i++) {
        Eigen::Vector3f surface_p = xyz_surf[i];
        Eigen::Vector3f samp1 = surface_p;
        Eigen::Vector3f samp2 = surface_p;

        for (int j = 0; j < 3; j++) {
            samp1[j] += perterb_norm(rng);
            samp2[j] += perterb_second(rng);
        }

        xyz.push_back(samp1);
        xyz.push_back(samp2);
    }

    // Uniform sampling within unit sphere
    for (int s = 0; s < (int) (num_rand_samples); s++) {
        xyz.push_back(Eigen::Vector3f(
                rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2,
                rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2,
                rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2));
    }

    // Now compute sdf for each xyz sample
    for (int s = 0; s < (int) xyz.size(); s++) {
        Eigen::Vector3f samp_vert = xyz[s];
        std::vector<int> cl_indices(num_votes);
        std::vector<float> cl_distances(num_votes);
        kdTree.knnSearch(samp_vert.data(), num_votes, cl_indices.data(), cl_distances.data());

        int num_pos = 0;
        float sdf;

        for (int ind = 0; ind < num_votes; ind++) {
            uint32_t cl_ind = cl_indices[ind];
            Eigen::Vector3f cl_vert = vertices[cl_ind];
            Eigen::Vector3f ray_vec = samp_vert - cl_vert;
            float ray_vec_leng = ray_vec.norm();

            if (ind == 0) {
                // if close to the surface, use point plane distance
                if (ray_vec_leng < stdv)
                    sdf = fabs(normals[cl_ind].dot(ray_vec));
                else
                    sdf = ray_vec_leng;
            }

            float d = normals[cl_ind].dot(ray_vec / ray_vec_leng);
            if (d > 0)
                num_pos++;
        }

        // all or nothing , else ignore the point
        if ((num_pos == 0) || (num_pos == num_votes)) {
            xyz_used.push_back(samp_vert);
            if (num_pos <= (num_votes / 2)) {
                sdf = -sdf;
            }
            sdfs.push_back(sdf);
        }
        std::cout << s << "  :  " << xyz[s][0] << ", " << xyz[s][1] << ", " << xyz[s][2] << " --> " << sdf << std::endl;
    }

    xyz = xyz_used;
}

void writeSDFToNPY(
        std::vector<Eigen::Vector3f> &xyz,
        std::vector<float> &sdfs,
        std::string filename) {
    unsigned int num_vert = xyz.size();
    std::vector<float> data(num_vert * 4);
    int data_i = 0;

    for (unsigned int i = 0; i < num_vert; i++) {
        Eigen::Vector3f v = xyz[i];
        float s = sdfs[i];

        for (int j = 0; j < 3; j++)
            data[data_i++] = v[j];
        data[data_i++] = s;
    }

    cnpy::npy_save(filename, &data[0], {(long unsigned int) num_vert, 4}, "w");
}

void writeSDFToNPZ(
        std::vector<Eigen::Vector3f> &xyz,
        std::vector<float> &sdfs,
        std::string filename,
        bool print_num = false) {
    unsigned int num_vert = xyz.size();
    std::vector<float> pos;
    std::vector<float> neg;

    for (unsigned int i = 0; i < num_vert; i++) {
        Eigen::Vector3f v = xyz[i];
        float s = sdfs[i];

        if (s > 0) {
            for (int j = 0; j < 3; j++)
                pos.push_back(v[j]);
            pos.push_back(s);
        } else {
            for (int j = 0; j < 3; j++)
                neg.push_back(v[j]);
            neg.push_back(s);
        }
    }

    cnpy::npz_save(filename, "pos", &pos[0], {(long unsigned int) (pos.size() / 4.0), 4}, "w");
    cnpy::npz_save(filename, "neg", &neg[0], {(long unsigned int) (neg.size() / 4.0), 4}, "a");
    if (print_num) {
        std::cout << "pos num: " << pos.size() / 4.0 << std::endl;
        std::cout << "neg num: " << neg.size() / 4.0 << std::endl;
    }
}

void writeSDFToPLY(
        std::vector<Eigen::Vector3f> &xyz,
        std::vector<float> &sdfs,
        std::string filename,
        bool neg_only = true,
        bool pos_only = false) {
    int num_verts;
    if (neg_only) {
        num_verts = 0;
        for (int i = 0; i < (int) sdfs.size(); i++) {
            float s = sdfs[i];
            if (s <= 0)
                num_verts++;
        }
    } else if (pos_only) {
        num_verts = 0;
        for (int i = 0; i < (int) sdfs.size(); i++) {
            float s = sdfs[i];
            if (s >= 0)
                num_verts++;
        }
    } else {
        num_verts = xyz.size();
    }

    std::ofstream plyFile;
    plyFile.open(filename);
    plyFile << "ply\n";
    plyFile << "format ascii 1.0\n";
    plyFile << "element vertex " << num_verts << "\n";
    plyFile << "property float x\n";
    plyFile << "property float y\n";
    plyFile << "property float z\n";
    plyFile << "property uchar red\n";
    plyFile << "property uchar green\n";
    plyFile << "property uchar blue\n";
    plyFile << "end_header\n";

    for (int i = 0; i < (int) sdfs.size(); i++) {
        Eigen::Vector3f v = xyz[i];
        float sdf = sdfs[i];
        bool neg = (sdf <= 0);
        bool pos = (sdf >= 0);
        if (neg)
            sdf = -sdf;
        int sdf_i = std::min((int) (sdf * 255), 255);
        if (!neg_only && pos)
            plyFile << v[0] << " " << v[1] << " " << v[2] << " " << 0 << " " << 0 << " " << sdf_i << "\n";
        if (!pos_only && neg)
            plyFile << v[0] << " " << v[1] << " " << v[2] << " " << sdf_i << " " << 0 << " " << 0 << "\n";
    }
    plyFile.close();
}

int oldMain(int argc, char **argv) {
    std::string meshFileName;
    bool vis = false;

    std::string npyFileName;
    std::string plyFileNameOut;
    std::string spatial_samples_npz;
    bool save_ply = true;
    bool test_flag = false;
    float variance = 0.005;
    int num_sample = 500000;
    float rejection_criteria_obs = 0.02f;
//    float rejection_criteria_obs = 0.7f;
    float rejection_criteria_tri = 0.03f;
//    float rejection_criteria_tri = 0.3f;
    float num_samp_near_surf_ratio = 47.0f / 50.0f;


    CLI::App app{"PreprocessMesh"};
    app.add_option("-m", meshFileName, "Mesh File Name for Reading")->required();
    app.add_flag("-v", vis, "enable visualization");
    app.add_option("-o", npyFileName, "Save npy pc to here")->required();
    app.add_option("--ply", plyFileNameOut, "Save ply pc to here");
    app.add_option("-s", num_sample, "Num of samples");
    app.add_option("--var", variance, "Set Variance");
    app.add_flag("--sply", save_ply, "save ply point cloud for visualization");
    app.add_flag("-t", test_flag, "test_flag");
    app.add_option("-n", spatial_samples_npz, "spatial samples from file");

    CLI11_PARSE(app, argc, argv);

    if (test_flag)
        variance = 0.05;

    float second_variance = variance / 10;
    std::cout << "Variance: " << variance << " Second variance: " << second_variance << std::endl;
    if (test_flag) {
        second_variance = variance / 100;
        num_samp_near_surf_ratio = 45.0f / 50.0f;
        num_sample = 250000;
    }

    if (spatial_samples_npz.length() != 0)
        std::cout << "Spatial samples from file: " << spatial_samples_npz << std::endl;

    pangolin::Geometry geom = pangolin::LoadGeometry(meshFileName);

    std::cout << "Mesh " << meshFileName << " has " << geom.objects.size() << " objects" << std::endl;

    // Object index linearization
    {
        int total_num_faces = 0;
        // Count num of faces in mesh
        for (const auto &object : geom.objects) {
            auto it_vert_indices = object.second.attributes.find("vertex_indices");
            // If this object has a vertex_indices (faces) attribute
            if (it_vert_indices != object.second.attributes.end()) {
                // Store vertex indices (faces) of current object to Image placeholder of dim (#faces x 3)
                pangolin::Image<uint32_t> ibo = pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);
                // Aggregate #faces
                total_num_faces += ibo.h;
            }
        }

        // Creation of buffer to store faces - reinteprets buffer of 12 width for char as 3 width for int
        pangolin::ManagedImage<uint8_t> new_buffer(3 * sizeof(uint32_t), total_num_faces);
        pangolin::Image<uint32_t> new_ibo = new_buffer.UnsafeReinterpret<uint32_t>().SubImage(0, 0, 3, total_num_faces);

        int index = 0;  // index of new_ibo row where the current ibo's row will be copied
        // Fill new_ibo with vertex_index data from each row of each object of geometry
        for (const auto &object : geom.objects) {
            auto it_vert_indices = object.second.attributes.find("vertex_indices");
            if (it_vert_indices != object.second.attributes.end()) {
                pangolin::Image<uint32_t> ibo = pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

                for (int i = 0; i < ibo.h; ++i) {
                    new_ibo.Row(index).CopyFrom(ibo.Row(i));
                    ++index;
                }
            }
        }

        // Clear geom objects map
        geom.objects.clear();
        // Insert a mesh entry to geom objects map with the new_ibo of faces constructed above as value
        auto faces = geom.objects.emplace(std::string("mesh"), pangolin::Geometry::Element());
        faces->second.Reinitialise(3 * sizeof(uint32_t), total_num_faces);
        faces->second.CopyFrom(new_buffer);
        new_ibo = faces->second.UnsafeReinterpret<uint32_t>().SubImage(0, 0, 3, total_num_faces);
        faces->second.attributes["vertex_indices"] = new_ibo;
    }

    // Remove textures
    geom.textures.clear();
    // Set linearized mesh faces to local var
    pangolin::Image<uint32_t> modelFaces =
            pangolin::get<pangolin::Image<uint32_t >>(geom.objects.begin()->second.attributes["vertex_indices"]);


    // Center mesh & fit to unit sphere
    float max_dist = BoundingCubeNormalization(geom, true);

    //RENDER MESH
    if (vis)
        pangolin::CreateWindowAndBind("Main", 640, 480);
    else
        pangolin::CreateWindowAndBind("Main", 1, 1);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_DITHER);
    glDisable(GL_POINT_SMOOTH);
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_POLYGON_SMOOTH);
    glHint(GL_POINT_SMOOTH, GL_DONT_CARE);
    glHint(GL_LINE_SMOOTH, GL_DONT_CARE);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_DONT_CARE);
    glDisable(GL_MULTISAMPLE_ARB);
    glShadeModel(GL_FLAT);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
//            pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.05, 100),
            pangolin::ProjectionMatrixOrthographic(-max_dist, max_dist, -max_dist, max_dist, 0, 2.5),
            pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY));
    pangolin::OpenGlRenderState s_cam2(
//            pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.05, 100),
            pangolin::ProjectionMatrixOrthographic(-max_dist, max_dist, max_dist, -max_dist, 0, 2.5),
            pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY));

    // Create Interactive View in window
    pangolin::Handler3D handler(s_cam);
    // Convert to GL geometry entity
    pangolin::GlGeometry gl_geom = pangolin::ToGlGeometry(geom);

    // Fetch shader program
    pangolin::GlSlProgram prog = GetShaderProgram();

    if (vis) {
        pangolin::View &d_cam = pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                .SetHandler(&handler);

        while (!pangolin::ShouldQuit()) {
            // Clear screen and activate view to render into
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//            glEnable(GL_CULL_FACE);
//            glCullFace(GL_FRONT);

            d_cam.Activate(s_cam);

            prog.Bind();
            prog.SetUniform("MVP", s_cam.GetProjectionModelViewMatrix());
            prog.SetUniform("V", s_cam.GetModelViewMatrix());

//            pangolin::glDrawColouredCube();
            pangolin::GlDraw(prog, gl_geom, nullptr);
            prog.Unbind();

            // Swap frames and Process Events
            pangolin::FinishFrame();
        }
    }

    // Create Framebuffer with attached textures
    size_t w = 400;
    size_t h = 400;
    pangolin::GlRenderBuffer zbuffer(w, h, GL_DEPTH_COMPONENT32);
    pangolin::GlTexture normals(w, h, GL_RGBA32F);
    pangolin::GlTexture vertices(w, h, GL_RGBA32F);
    pangolin::GlFramebuffer framebuffer(vertices, normals, zbuffer);

    // View points around a sphere
    std::vector<Eigen::Vector3f> views = EquiDistPointsOnSphere(100, max_dist * 1.1);

    std::vector<Eigen::Vector4f> point_normals;
    std::vector<Eigen::Vector4f> point_verts;

    // Init test variables for wrong objects
    size_t num_tri = modelFaces.h;
    std::vector<Eigen::Vector4f> tri_id_normal_test(num_tri);
    for (size_t j = 0; j < num_tri; j++)
        tri_id_normal_test[j] = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f);
    int total_obs = 0;
    int wrong_obs = 0;

    for (unsigned int v = 0; v < views.size(); v++) {
        // change cam location
        s_cam2.SetModelViewMatrix(
                pangolin::ModelViewLookAt(views[v][0], views[v][1], views[v][2], 0, 0, 0, pangolin::AxisY)
        );
        // Draw the scene to the framebuffer
        framebuffer.Bind();
        glViewport(0, 0, w, h);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        prog.Bind();
        prog.SetUniform("MVP", s_cam2.GetProjectionModelViewMatrix());
        prog.SetUniform("V", s_cam2.GetModelViewMatrix());
        prog.SetUniform("ToWorld", s_cam2.GetModelViewMatrix().Inverse());
        prog.SetUniform("slant_thr", -1.0f, 1.0f);
        prog.SetUniform("ttt", 1.0, 0, 0, 1);
        pangolin::GlDraw(prog, gl_geom, nullptr);
        prog.Unbind();

        framebuffer.Unbind();

        // Fetch point normals of valid tris&points
        pangolin::TypedImage img_normals;
        normals.Download(img_normals);
        std::vector<Eigen::Vector4f> im_norms = ValidPointsAndTrisFromIm(
                img_normals.UnsafeReinterpret<Eigen::Vector4f>(), tri_id_normal_test, total_obs, wrong_obs
        );
        point_normals.insert(point_normals.end(), im_norms.begin(), im_norms.end());

        // Fetch point verts of valid points
        pangolin::TypedImage img_verts;
        vertices.Download(img_verts);
        std::vector<Eigen::Vector4f> im_verts =
                ValidPointsFromIm(img_verts.UnsafeReinterpret<Eigen::Vector4f>());
        point_verts.insert(point_verts.end(), im_verts.begin(), im_verts.end());
    }

    // Scan array of tested triangles and count how many bad tris
    int bad_tri = 0;
    for (unsigned int t = 0; t < tri_id_normal_test.size(); t++) {
        if (tri_id_normal_test[t][3] < 0.0f)
            bad_tri++;
    }

    // Log watertightness test results
    float wrong_obj_ratio = (float) (wrong_obs) / float(total_obs);
    float bad_tri_ratio = (float) (bad_tri) / float(num_tri);

    std::cout << "Mesh: " << meshFileName << " tested for watertightness:" << std::endl;
    std::cout << "Wrong objects/Total objects ratio: " << wrong_obj_ratio << " (" << wrong_obs
              << "/" << total_obs << ")" << std::endl;
    std::cout << "Bad tris/Total tris ratio: " << bad_tri_ratio << " (" << bad_tri
              << "/" << num_tri << ")" << std::endl;

    if (wrong_obj_ratio > rejection_criteria_obs || bad_tri_ratio > rejection_criteria_tri) {
        std::cout << "Mesh " << meshFileName << " rejected" << std::endl;
        return 0;
    } else {
        std::cout << "Mesh " << meshFileName << " passed test!!!!!!!!" << std::endl;
    }


    // These vectors store the above point normals & verts coords without the 4th dim (the primitive (tri) id)
    std::vector<Eigen::Vector3f> vertices2;
    std::vector<Eigen::Vector3f> normals2;
    for (unsigned int v = 0; v < point_verts.size(); v++) {
        vertices2.push_back(point_verts[v].head<3>());
        normals2.push_back(point_normals[v].head<3>());
    }

    // Construct KD-Tree from valid surface vert points
    std::cout << "Building KD-tree for mesh " << meshFileName << std::endl;
    KdVertexList kdVerts(vertices2);
    KdVertexListTree kdTree_surf(3, kdVerts);
    kdTree_surf.buildIndex();
    std::cout << "KD-tree for mesh " << meshFileName << " built" << std::endl;


    // Sample points aggressively on surface weighted by tri area
    std::vector<Eigen::Vector3f> xyz;
    std::vector<Eigen::Vector3f> xyz_surf;
    std::vector<float> sdf;
    int num_samp_near_surface = (int) (47 * num_sample / 50);  // 470000?
    std::cout << "Num of samples near surface: " << num_samp_near_surface << std::endl;
    SampleFromSurface(geom, xyz_surf, num_samp_near_surface / 2);

    // Perturbation of surface points and rest of sampling + SDF calculation, also duration calculation and log
    std::cout << "Sampling SDFs for mesh " << meshFileName << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    SampleSDFNearSurface(
            kdTree_surf,
            vertices2,
            xyz_surf,
            normals2,
            xyz,
            sdf,
            num_sample - num_samp_near_surface,
            variance,
            second_variance,
            2,
            11);
    auto finish = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(finish - start).count();
    std::cout << "Perturbation of surface points and rest of sampling + SDF calculation for mesh " << meshFileName
              << " took " << elapsed << " seconds."
              << std::endl;

    // Create PLY file from SDFs if flag is set
    if (save_ply) {
        writeSDFToPLY(xyz, sdf, plyFileNameOut, true, false);
    }

    std::cout << "Num points sampled: " << xyz.size() << std::endl;
    std::size_t save_npz = npyFileName.find("npz");
    if (save_npz == std::string::npos)
        writeSDFToNPY(xyz, sdf, npyFileName);
    else {
        writeSDFToNPZ(xyz, sdf, npyFileName, true);
    }
    std::cout << "Preprocess done for mesh " << meshFileName << std::endl << "------------------------------"
              << std::endl;

    return 0;
}



// -------------------------------------------------------------------------------


void initializeGL(bool vis, std::string meshfilename) {
    // Initialize GLFW
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW\n");
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a window and create its OpenGL context
    if (vis) {
        windowWidth = 640;
        windowHeight = 480;
    } else {
        windowWidth = 1;
        windowHeight = 1;
    }
    window = glfwCreateWindow(windowWidth, windowHeight, meshfilename.c_str(), nullptr, nullptr);

    if (window == nullptr) {
        glfwTerminate();
        throw std::runtime_error(std::string(std::string("Failed to open GLFW window.") +
                                             " If you have an Intel GPU, they are not 3.3 compatible." +
                                             "Try the 2.1 version.\n"));
    }
    glfwMakeContextCurrent(window);
    glfwPollEvents();
    glfwSetWindowSize(window, windowWidth + 1, windowHeight + 1);

    // Start GLEW extension handler
    glewExperimental = GL_TRUE;

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        glfwTerminate();
        throw std::runtime_error("Failed to initialize GLEW\n");
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // Hide the mouse and enable unlimited movement
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Set the mouse at the center of the screen
    glfwPollEvents();
    glfwSetCursorPos(window, windowWidth / 2, windowHeight / 2);

    // Gray background color
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    // Log
    logGLParameters();

    // Create camera
    camera = new Camera(window);
}

void createContext(std::vector<glm::vec3> objVertices) {

    // Create and compile our GLSL program from the shaders
    shaderProgram = loadShaders(
            "./src/shaders/vertexShader.glsl",
            "./src/shaders/fragmentShader.glsl",
            "./src/shaders/geometryShader.glsl");

    projectionMatrixLocation = glGetUniformLocation(shaderProgram, "P");
    viewMatrixLocation = glGetUniformLocation(shaderProgram, "V");
    modelMatrixLocation = glGetUniformLocation(shaderProgram, "M");

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_DITHER);
    glDisable(GL_POINT_SMOOTH);
    glDisable(GL_LINE_SMOOTH);
    glDisable(GL_POLYGON_SMOOTH);
    glHint(GL_POINT_SMOOTH, GL_DONT_CARE);
    glHint(GL_LINE_SMOOTH, GL_DONT_CARE);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_DONT_CARE);
    glDisable(GL_MULTISAMPLE_ARB);
    glShadeModel(GL_FLAT);

    // obj
    // Task 6.1: bind object vertex positions to attribute 0, UV coordinates
    // to attribute 1 and normals to attribute 2
    //*/
    glGenVertexArrays(1, &objVAO);
    glBindVertexArray(objVAO);

    // vertex VBO
    glGenBuffers(1, &objVerticiesVBO);
    glBindBuffer(GL_ARRAY_BUFFER, objVerticiesVBO);
    glBufferData(GL_ARRAY_BUFFER, objVertices.size() * sizeof(glm::vec3),
                 &objVertices[0], GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

}


void mainLoopVis(const std::vector<glm::vec3> &objVertices) {

    do {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        // camera
        camera->update();

        glBindVertexArray(objVAO);

        // Free-fly
        glm::mat4 projectionMatrix = camera->projectionMatrix;
        glm::mat4 viewMatrix = camera->viewMatrix;
        glm::mat4 modelMatrix = glm::mat4(1.0);

        // -maxDist -> maxDist perspective
//        glm::mat4 projectionMatrix{
//                1, 0, 0, 0,
//                0, 1, 0, 0,
//                0, 0, -0.8, -1,
//                0, 0, 0, 1
//        };
//        glm::mat4 projectionMatrix{
//                1.3125, 0, 0, 0,
//                0, 1.75, 0, 0,
//                0, 0, -1.001, -0.10005,
//                0, 0, 0, 1
//        };
//        glm::mat4 viewMatrix = glm::lookAt(
//                glm::vec3(0, 0, -1),
//                glm::vec3(0, 0, 0),
//                glm::vec3(0, 1, 0)
//        );
//        glm::mat4 modelMatrix = glm::mat4(1.0);

        // Task 1.4c: transfer uniforms to GPU
        glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, &projectionMatrix[0][0]);
        glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, &viewMatrix[0][0]);
        glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, &modelMatrix[0][0]);
//        glUniform3f(viewPosLocation, camera->position.x, camera->position.y, camera->position.z);

        glDrawArrays(GL_TRIANGLES, 0, objVertices.size());

        glfwSwapBuffers(window);

        glfwPollEvents();

    } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
             glfwWindowShouldClose(window) == 0);
}

void free() {
    glDeleteBuffers(1, &objVerticiesVBO);
    glDeleteVertexArrays(1, &objVAO);
    glDeleteTextures(1, &texVertBuffer);
    glDeleteTextures(1, &texNormalBuffer);
    glDeleteRenderbuffers(1, &rbo);
    glDeleteFramebuffers(1, &framebuffer);

    glDeleteProgram(shaderProgram);
    glfwTerminate();
}

void
virtualRenderTest(float max_dist, int numOfFaces, const std::vector<glm::vec3> &objVertices,
                  const std::string &meshFileName,
                  float rejection_criteria_obs, float rejection_criteria_tri,
                  std::vector<Eigen::Vector3f> &validVertices, std::vector<Eigen::Vector3f> &validNormals) {
    // Create Framebuffer with attached textures
    size_t w = 400;
    size_t h = 400;

    // generate texture buffers for attaching to frambuffer (0 for vert pos, 1 for normals)
    glGenTextures(1, &texVertBuffer);
    glBindTexture(GL_TEXTURE_2D, texVertBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &texNormalBuffer);
    glBindTexture(GL_TEXTURE_2D, texNormalBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Gen render buffer for attaching to framebuffer
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, w, h);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    // Gen & bind custom framebuffer

    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    // attach Texture & render buffers to currently bound framebuffer object
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texVertBuffer, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, texNormalBuffer, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    auto fboStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (fboStatus != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "Framebuffer not complete: " << fboStatus << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // View points around a sphere
    std::vector<Eigen::Vector3f> views = EquiDistPointsOnSphere(100, max_dist * 1.1);

    std::vector<Eigen::Vector4f> point_normals;
    std::vector<Eigen::Vector4f> point_verts;

    // Init test variables for wrong objects
    size_t num_tri = numOfFaces;
    std::vector<Eigen::Vector4f> tri_id_normal_test(num_tri);
    for (size_t j = 0; j < num_tri; j++)
        tri_id_normal_test[j] = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f);
    int total_obs = 0;
    int wrong_obs = 0;

    for (auto &view : views) {

        // change cam location
        // -maxDist -> maxDist perspective
        glm::mat4 projectionMatrix{
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, -0.8, -1,
                0, 0, 0, 1
        };
        glm::mat4 viewMatrix = glm::lookAt(
                glm::vec3(view[0], view[1], view[2]),
                glm::vec3(0, 0, 0),
                glm::vec3(0, 1, 0)
        );
        glm::mat4 modelMatrix = glm::mat4(1.0);

        // Draw the scene to the framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

        // Set the list of draw buffers.
        GLenum DrawBuffers[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
        glDrawBuffers(2, DrawBuffers); // "2" is the size of DrawBuffers

        glViewport(0, 0, w, h);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);
        // get pointers to the uniform variables
        projectionMatrixLocation = glGetUniformLocation(shaderProgram, "P");
        viewMatrixLocation = glGetUniformLocation(shaderProgram, "V");
        modelMatrixLocation = glGetUniformLocation(shaderProgram, "M");
        // viewPosLocation = glGetUniformLocation(shaderProgram, "viewPos"); // camPos
        glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, &projectionMatrix[0][0]);
        glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, &viewMatrix[0][0]);
        glUniformMatrix4fv(modelMatrixLocation, 1, GL_FALSE, &modelMatrix[0][0]);
        glEnable(GL_DEPTH_TEST);

        glBindVertexArray(objVAO);
        glDrawArrays(GL_TRIANGLES, 0, objVertices.size());

        glBindFramebuffer(GL_FRAMEBUFFER, 0);


        // Get vert & normal buffer values from textures and store into RAM
//        auto *img_verts = new GLfloat[w * h * 4];
//        glBindTexture(GL_TEXTURE_2D, texVertBuffer);
//        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, img_verts);
//        glBindTexture(GL_TEXTURE_2D, 0);
//
//        auto *norm_pixels = new GLfloat[w * h * 4];
//        glBindTexture(GL_TEXTURE_2D, texNormalBuffer);
//        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, norm_pixels);
//        glBindTexture(GL_TEXTURE_2D, 0);
//
//        for (int i = 0; i < w * h * 4; ++i) {
//            std::cout << img_verts[i] << " ";
//            if (i % 4 == 0)
//                std::cout << std::endl;
//
//        }

        pangolin::TypedImage vert_pixels;
        vert_pixels.Reinitialise(w, h, pangolin::PixelFormatFromString("RGBA128F"));
        glBindTexture(GL_TEXTURE_2D, texVertBuffer);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, vert_pixels.ptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        pangolin::TypedImage norm_pixels;
        norm_pixels.Reinitialise(w, h, pangolin::PixelFormatFromString("RGBA128F"));
        glBindTexture(GL_TEXTURE_2D, texNormalBuffer);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, norm_pixels.ptr);
        glBindTexture(GL_TEXTURE_2D, 0);


        // Fetch point normals of valid tris&points
        std::vector<Eigen::Vector4f> im_norms = ValidPointsAndTrisFromIm(
                norm_pixels.UnsafeReinterpret<Eigen::Vector4f>(), tri_id_normal_test, total_obs, wrong_obs
        );
        point_normals.insert(point_normals.end(), im_norms.begin(), im_norms.end());

        // Fetch point verts of valid points
        std::vector<Eigen::Vector4f> im_verts = ValidPointsFromIm(
                vert_pixels.UnsafeReinterpret<Eigen::Vector4f>());
        point_verts.insert(point_verts.end(), im_verts.begin(), im_verts.end());

    }

    // Scan array of tested triangles and count how many bad tris
    int bad_tri = 0;
    for (auto &t : tri_id_normal_test) {
        if (t[3] < 0.0f)
            bad_tri++;
    }

    // Log watertightness test results
    float wrong_obj_ratio = (float) (wrong_obs) / float(total_obs);
    float bad_tri_ratio = (float) (bad_tri) / float(num_tri);

    std::cout << "Mesh: " << meshFileName << " tested for watertightness:" << std::endl;
    std::cout << "Wrong objects/Total objects ratio: " << wrong_obj_ratio << " (" << wrong_obs
              << "/" << total_obs << ")" << std::endl;
    std::cout << "Bad tris/Total tris ratio: " << bad_tri_ratio << " (" << bad_tri
              << "/" << num_tri << ")" << std::endl;

    if (wrong_obj_ratio > rejection_criteria_obs || bad_tri_ratio > rejection_criteria_tri) {
        std::cout << "Mesh " << meshFileName << " REJECTED" << std::endl;
        free();
        exit(1);
    } else {
        std::cout << "Mesh " << meshFileName << " PASSED!!!!!!!!" << std::endl;
    }

    // These vectors store the above point normals & verts coords without the 4th dim (the primitive (tri) id)
    for (unsigned int v = 0; v < point_verts.size(); v++) {
        validVertices.push_back(point_verts[v].head<3>());
        validNormals.push_back(point_normals[v].head<3>());
    }
}


int main(int argc, char **argv) {

    std::string meshFileName;
    bool vis = false;

    std::string npyFileName;
    std::string plyFileNameOut;
    std::string spatial_samples_npz;
    bool save_ply = true;
    bool test_flag = false;
    float variance = 0.005;
    int num_sample = 500000;
    float rejection_criteria_obs = 0.028f;
//    float rejection_criteria_obs = 0.7f;
    float rejection_criteria_tri = 0.03f;
//    float rejection_criteria_tri = 0.3f;
    float num_samp_near_surf_ratio = 47.0f / 50.0f;

    CLI::App app{"PreprocessMesh"};
    app.add_option("-m", meshFileName, "Mesh File Name for Reading")->required();
    app.add_flag("-v", vis, "enable visualization");
    app.add_option("-o", npyFileName, "Save npy pc to here")->required();
    app.add_option("--ply", plyFileNameOut, "Save ply pc to here");
    app.add_option("-s", num_sample, "Num of samples");
    app.add_option("--var", variance, "Set Variance");
    app.add_flag("--sply", save_ply, "save ply point cloud for visualization");
    app.add_flag("-t", test_flag, "test_flag");
    app.add_option("-n", spatial_samples_npz, "spatial samples from file");

    CLI11_PARSE(app, argc, argv);

    if (test_flag)
        variance = 0.05;

    float second_variance = variance / 10;
    std::cout << "Variance: " << variance << " Second variance: " << second_variance << std::endl;
    if (test_flag) {
        second_variance = variance / 100;
        num_samp_near_surf_ratio = 45.0f / 50.0f;
        num_sample = 250000;
    }

    if (spatial_samples_npz.length() != 0)
        std::cout << "Spatial samples from file: " << spatial_samples_npz << std::endl;

    std::vector<glm::vec3> objVertices, objNormals;
    std::vector<glm::vec2> objUVs;
    std::vector<unsigned int> indices;
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    // Load obj
    std::string err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, meshFileName.c_str())) {
        throw std::runtime_error(err);
    }
    int modelFaces = 0;
    for (const auto &shape : shapes) {
        modelFaces += shape.mesh.num_face_vertices.size();
        for (const auto &index : shape.mesh.indices) {
            glm::vec3 vertex = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
            };
            objVertices.push_back(vertex);
            indices.push_back(indices.size());
        }
    }

    std::cout << "Tiny found faces: " << modelFaces << std::endl;

    // Center mesh & fit to unit sphere
    float max_dist = TinyBoundingCubeNormalization(objVertices, true);

    std::vector<Eigen::Vector3f> validVertices;
    std::vector<Eigen::Vector3f> validNormals;

    try {
        initializeGL(vis, meshFileName);
        createContext(objVertices);
        if (vis)
            mainLoopVis(objVertices);
        virtualRenderTest(max_dist, modelFaces, objVertices, meshFileName, rejection_criteria_obs,
                          rejection_criteria_tri, validVertices, validNormals);
    }
    catch (std::exception &ex) {
        std::cout << ex.what() << std::endl;
        getchar();
        free();
        return -1;
    }

    // Construct KD-Tree from valid surface vert points
    std::cout << "Building KD-tree for mesh " << meshFileName << std::endl;
    KdVertexList kdVerts(validVertices);
    KdVertexListTree kdTree_surf(3, kdVerts);
    kdTree_surf.buildIndex();
    std::cout << "KD-tree for mesh " << meshFileName << " built" << std::endl;

    // ONLY IF TEST PASSED USED PANGOLIN FOR SURFACE SAMPLE FUNCTION
    // TODO: find a way to use tinyobj manipulation to avoid using pangolin loadGeometry func
    pangolin::Geometry geom = pangolin::LoadGeometry(meshFileName);
    // Center mesh & fit to unit sphere
    BoundingCubeNormalization(geom, true);


    // Sample points aggressively on surface weighted by tri area
    std::vector<Eigen::Vector3f> xyz;
    std::vector<Eigen::Vector3f> xyz_surf;
    std::vector<float> sdf;
    int num_samp_near_surface = (int) (47 * num_sample / 50);  // 470000?
    std::cout << "Num of samples near surface: " << num_samp_near_surface << std::endl;
    SampleFromSurface(geom, xyz_surf, num_samp_near_surface / 2);
//    TinySampleFromSurface(
//            attrib,
//            shapes,
//            xyz_surf,
//            num_samp_near_surface / 2
//    );

    // Perturbation of surface points and rest of sampling + SDF calculation, also duration calculation and log
    std::cout << "Sampling SDFs for mesh " << meshFileName << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    SampleSDFNearSurface(
            kdTree_surf,
            validVertices,
            xyz_surf,
            validNormals,
            xyz,
            sdf,
            num_sample - num_samp_near_surface,
            variance,
            second_variance,
            2,
            11);
    auto finish = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(finish - start).count();
    std::cout << "Perturbation of surface points and rest of sampling + SDF calculation for mesh " << meshFileName
              << " took " << elapsed << " seconds."
              << std::endl;

    // Create PLY file from SDFs if flag is set
    if (save_ply) {
        writeSDFToPLY(xyz, sdf, plyFileNameOut, true, false);
    }

    std::cout << "Num points sampled: " << xyz.size() << std::endl;
    std::size_t save_npz = npyFileName.find("npz");
    if (save_npz == std::string::npos)
        writeSDFToNPY(xyz, sdf, npyFileName);
    else {
        writeSDFToNPZ(xyz, sdf, npyFileName, true);
    }
    std::cout << "Preprocess done for mesh " << meshFileName << std::endl << "------------------------------"
              << std::endl;

    free();
    return 0;
}
