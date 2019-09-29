import numpy as np
import trimesh.visual
import os
import json

partial_scan_dataset_dir = os.path.join(os.sep, 'Volumes', 'warm_blue', 'datasets', 'partial_scans')
denoised_dir = 'test-images_dim32_sdf_pc'
noised_dir = 'NOISE_test-images_dim32_sdf_pc'

if not os.path.isdir(os.path.join(partial_scan_dataset_dir, noised_dir)):
    os.makedirs(os.path.join(partial_scan_dataset_dir, noised_dir))

with open('examples/splits/sv2_chairs_test_noise.json', "r") as f:
    split = json.load(f)

# Fetch representation test chairs for filtering positive
pcs = []
for dataset in split:
    for class_name in split[dataset]:

        if not os.path.isdir(os.path.join(partial_scan_dataset_dir, noised_dir, class_name)):
            os.makedirs(os.path.join(partial_scan_dataset_dir, noised_dir, class_name))

        for instance_name in split[dataset][class_name]:
            instance_name = '{0}__0__'.format(instance_name)
            if not os.path.isdir(os.path.join(partial_scan_dataset_dir, noised_dir, class_name, instance_name)):
                os.makedirs(os.path.join(partial_scan_dataset_dir, noised_dir, class_name, instance_name))

            # attach to logger so trimesh messages will be printed to console
            trimesh.util.attach_to_log()

            sds = [0.01, 0.02, 0.03, 0.05]
            for sd in sds:
                # Load pc
                pc = trimesh.load(
                    os.path.join(partial_scan_dataset_dir, denoised_dir, class_name, instance_name + '.ply'))

                # create zero-mean Gaussian noise
                noise = np.random.normal(0, sd, pc.vertices.shape)

                # adds Gaussian noise to pc's vertices
                pc.vertices += noise

                # export ply
                noised_pc_filename = '{0}_NOISE_{1}'.format(sd, instance_name + '.ply')
                if not os.path.isfile(os.path.join(partial_scan_dataset_dir, noised_dir, class_name, instance_name,
                                                   noised_pc_filename)):
                    trimesh.Trimesh(vertices=pc.vertices).export(
                        os.path.join(partial_scan_dataset_dir, noised_dir, class_name, instance_name,
                                     noised_pc_filename))

                # pc.show()
