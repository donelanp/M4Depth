import os
import argparse
from pyquaternion import Quaternion
import numpy as np
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--db_path", default=os.path.join(*[dir_path,"..", "datasets","UseGeo"]), help="path to folder containing the databases")
parser.add_argument("--output_dir", default=os.path.join(*[dir_path,"..", "data", "usegeo"]), help="path to folder to store csv files")
a = parser.parse_args()

def angles_to_quaternion(omega, phi, kappa):
    omega, phi, kappa = np.radians([omega, phi, kappa])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(omega), -np.sin(omega)],
                   [0, np.sin(omega),  np.cos(omega)]])

    Ry = np.array([[ np.cos(phi), 0, np.sin(phi)],
                   [0, 1, 0],
                   [-np.sin(phi), 0, np.cos(phi)]])

    Rz = np.array([[np.cos(kappa), -np.sin(kappa), 0],
                   [np.sin(kappa),  np.cos(kappa), 0],
                   [0, 0, 1]])

    R_wc = Rz @ Ry @ Rx
    return Quaternion(matrix=R_wc)

if __name__== "__main__":
    os.makedirs(a.output_dir, exist_ok=True)

    data = ["Dataset-1", "Dataset-2", "Dataset-3"]

    for iset, set in enumerate(data):
        print(f"Processing {set}")

        # load the data for the entire trajectory
        img_names = []
        positions = []
        orientations = []
        params = []

        dataset_dir = os.path.join(a.db_path, set)
        image_orientations_file = os.path.join(dataset_dir, f"Image_orientations_dataset{iset + 1}.xyz")

        with open(image_orientations_file) as f:
            next(f)
            for line in f:
                portions = line.strip().split()
                img_names.append(portions[0])
                positions.append(list(map(float, portions[1:4])))
                orientations.append(list(map(float, portions[4:7])))
                params.append(list(map(float, portions[7:10])))

        # split the data into smaller trajectories
        traj_len = 10

        for i_traj, i_start in enumerate(range(1, len(img_names) - traj_len, traj_len)):
            i_stop = i_start + traj_len

            # 1:3 test to training split
            if i_traj % 3 !=0:
                out_dir = os.path.join(*[a.output_dir, "train_data", set])
            else:
                out_dir = os.path.join(*[a.output_dir, "test_data", set])

            # create csv file
            os.makedirs(out_dir, exist_ok=True)
            file_name = os.path.join(out_dir, f"traj_{i_traj:04d}.csv")

            with open(file_name, 'w') as file:
                file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % ("id", "camera", "depth", "qw", "qx", "qy", "qz", "tx", "ty", "tz", "c", "fx", "fy"))

                for index, i_img in enumerate(range(i_start, i_start + traj_len)):
                    img_name = img_names[i_img].strip('.jpg')
                    camera = os.path.join(set, "Depth_resized", "undistorted_images", img_name + "_res.jpg")
                    depth = os.path.join(set, "Depth_resized", "depth_maps", img_name + "_depth_res.png")

                    img = Image.open(os.path.join(dataset_dir, "Depth_resized", "depth_maps", img_name + "_depth_res.tiff"))
                    if img.mode != "I":
                        img = img.convert("I")
                    img.save(os.path.join(dataset_dir, "Depth_resized", "depth_maps", img_name + "_depth_res.png"), format="PNG")

                    c = params[i_img][0] / 4.0
                    fx = params[i_img][1] / 4.0
                    fy = params[i_img][2] / 4.0

                    p_a = np.array(positions[i_img - 1])
                    p_b = np.array(positions[i_img])
                    q_r_a = angles_to_quaternion(*orientations[i_img - 1])
                    q_r_b = angles_to_quaternion(*orientations[i_img])
                    trans = q_r_a.conjugate.rotate(p_b - p_a)
                    rot = (q_r_a.conjugate * q_r_b).elements

                    rot = rot.tolist()
                    rot = [rot[0], rot[2], rot[3], rot[1]]
                    trans = [trans.tolist()[1], trans.tolist()[2], trans.tolist()[0]]

                    file.write("%i\t%s\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%s\t%s\t%s\n" % (index, camera, depth, rot[0], rot[1], rot[2], rot[3], trans[0], trans[1], trans[2], c, fx, fy))
