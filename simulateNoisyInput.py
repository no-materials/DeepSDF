import numpy as np
import trimesh.visual
import os

partial_scan_dataset_dir = os.path.join(os.sep, 'Volumes', 'warm_blue', 'datasets', 'partial_scans')
denoised_dir = 'test-images_dim32_sdf_pc'
noised_dir = 'NOISE_test-images_dim32_sdf_pc'
class_dir = '03001627'
instance_dir = 'd2c93dac1e088b092561e050fe719ba__0__'
instance_filename = 'd2c93dac1e088b092561e050fe719ba__0__.ply'

if not os.path.isdir(os.path.join(partial_scan_dataset_dir, noised_dir)):
    os.makedirs(os.path.join(partial_scan_dataset_dir, noised_dir))

if not os.path.isdir(os.path.join(partial_scan_dataset_dir, noised_dir, class_dir)):
    os.makedirs(os.path.join(partial_scan_dataset_dir, noised_dir, class_dir))

if not os.path.isdir(os.path.join(partial_scan_dataset_dir, noised_dir, class_dir, instance_dir)):
    os.makedirs(os.path.join(partial_scan_dataset_dir, noised_dir, class_dir, instance_dir))

# attach to logger so trimesh messages will be printed to console
trimesh.util.attach_to_log()

sds = [0.01, 0.02, 0.03, 0.05]
for sd in sds:
    # Load pc
    pc = trimesh.load(os.path.join(partial_scan_dataset_dir, denoised_dir, class_dir, instance_filename))

    # create zero-mean Gaussian noise
    noise = np.random.normal(0, sd, pc.vertices.shape)

    # set the colors on the random point and its nearest point to be the same
    pc.vertices += noise

    noised_pc_filename = '{0}_NOISE_{1}'.format(sd, instance_filename)
    if not os.path.isfile(os.path.join(partial_scan_dataset_dir, noised_dir, class_dir, instance_dir, noised_pc_filename)):
        trimesh.Trimesh(vertices=pc.vertices).export(
            os.path.join(partial_scan_dataset_dir, noised_dir, class_dir,instance_dir, noised_pc_filename))

    # pc.show()
