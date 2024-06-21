import os
import sys
import copy
sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from pythonUtils import osp, print_Extension, projectInfo

# Determine root directory based on computer name
computer_name = os.popen("hostname").read().strip()
ROOT_DIR_MAP = {
    'vivan-pc': '/home/vivandoshi/Documents/Distracted/dataset/'
}
OS_ROOT_DIR = ROOT_DIR_MAP.get(computer_name)

class Driver100DatasetPaths:
    def __init__(self, root):
        self.day_cameras = {f"Day_Cam{i}": os.path.join(OS_ROOT_DIR, root, f'Day/Cam{i}_size224') for i in range(1, 5)}
        self.day_cameras_aug = {f"Day_Cam{i}_Aug": os.path.join(OS_ROOT_DIR, root, f'Day/Cam{i}_Augment') for i in range(1, 5)}
        self.night_cameras = {f"Night_Cam{i}": os.path.join(OS_ROOT_DIR, root, f'Night/Cam{i}_size224') for i in range(1, 5)}
        self.night_cameras_aug = {f"Night_Cam{i}_Aug": os.path.join(OS_ROOT_DIR, root, f'Night/Cam{i}_Augment') for i in range(1, 5)}

class Driver100Labels:
    def __init__(self, root=osp.join(projectInfo.ROOT, 'train/groundtruth/100Driver')):
        self.cross_camera_labels = self._generate_labels(root, "Cross-camera-setting", self._cross_camera_path)
        self.cross_modality_labels = self._generate_labels(root, "Cross-modality-setting", self._cross_modality_path)
        self.cross_individual_vehicle_labels = self._generate_labels(root, "Cross-vehicle-setting/Cross-individual-vehicle", self._cross_individual_vehicle_path)
        self.cross_vehicle_type_labels = self._generate_labels(root, "Cross-vehicle-setting/Cross-vehicle-type", self._cross_vehicle_type_path)
    
    def _generate_labels(self, root, cross_type, path_func):
        labels = {}
        cam_views = ['Cam1', 'Cam2', 'Cam3', 'Cam4']
        times = ['Day', 'Night']

        for time in times:
            for view in cam_views:
                labels.update(path_func(root, cross_type, time, view))
        return labels
    
    def _cross_camera_path(self, root, cross_type, time, view):
        labels = {}
        left_cams = [cam for cam in ['Cam1', 'Cam2', 'Cam3', 'Cam4'] if cam != view]
        for subset in ['Train', 'Train_A', 'Val', 'Test']:
            add_str = '_Augment' if '_A' in subset else ''
            subset_str = subset.split('_')[0].lower()
            key = f"CCS_{time[0]}{view[3]}_{subset}"
            path = os.path.join(root + add_str[1:], cross_type, time, f'{view}_to_{"_".join(cam[3] for cam in left_cams)}', f'{view}_{subset_str}{add_str}.txt')
            labels[key] = path
        return labels

    def _cross_modality_path(self, root, cross_type, time, view):
        labels = {}
        left_time = 'Night' if time == 'Day' else 'Day'
        for subset in ['Train', 'Train_A', 'Val', 'Test']:
            if (time == 'Day' and subset == 'Test') or (time == 'Night' and subset == 'Val'):
                continue
            add_str = '_Augment' if '_A' in subset else ''
            subset_str = subset.split('_')[0].lower()
            key = f"CMS_{time[0]}{view[3]}_{subset}"
            path = os.path.join(root + add_str[1:], cross_type, f'{time[0]}{view[3]}_to_{left_time[0]}{view[3]}', f'{time[0]}{view[3]}_{subset_str}{add_str}.txt')
            labels[key] = path
        return labels

    def _cross_individual_vehicle_path(self, root, cross_type, time, view):
        labels = {}
        cam_views = ['Cam1', 'Cam2', 'Cam3', 'Cam4']
        for i in range(1, 5):
            cam = f'Cam{i}'
            labels.update({
                f"CIV_D{i}_Mazda_Train": os.path.join(root, cross_type, 'Day', cam, 'mazda_train.txt'),
                f"CIV_D{i}_Mazda_Train_A": os.path.join(root + 'Augment', cross_type, 'Day', cam, 'mazda_train_Augment.txt'),
                f"CIV_D{i}_Mazda_Test": os.path.join(root, cross_type, 'Day', cam, 'mazda_test.txt'),
                f"CIV_D{i}_Ankai_Test": os.path.join(root, cross_type, 'Day', cam, 'ankai.txt'),
                f"CIV_D{i}_Hyundai_Test": os.path.join(root, cross_type, 'Day', cam, 'hyundai.txt'),
                f"CIV_D{i}_Lynk_Test": os.path.join(root, cross_type, 'Day', cam, 'lynk.txt')
            })
        return labels

    def _cross_vehicle_type_path(self, root, cross_type, time, view):
        labels = {}
        cam_views = ['Cam1', 'Cam2', 'Cam3', 'Cam4']
        for i in range(1, 5):
            cam = f'Cam{i}'
            labels.update({
                f"CVT_D{i}_Sedan_Train": os.path.join(root, cross_type, cam, 'sedan_train.txt'),
                f"CVT_D{i}_Sedan_Train_A": os.path.join(root + 'Augment', cross_type, cam, 'sedan_train_Augment.txt'),
                f"CVT_D{i}_Sedan_Val": os.path.join(root, cross_type, cam, 'sedan_val.txt'),
                f"CVT_D{i}_SUV_Test": os.path.join(root, cross_type, cam, 'SUV.txt'),
                f"CVT_D{i}_Van_Test": os.path.join(root, cross_type, cam, 'Van.txt')
            })
        return labels

    def check(self):
        for key, path in self.__dict__.items():
            if isinstance(path, dict):
                for subkey, subpath in path.items():
                    if osp.exists(subpath):
                        print_Extension(f"Check Success: {subkey}: {subpath}", frontColor=32)
                    else:
                        print_Extension(f"Check Failed: {subkey}: {subpath}", frontColor=31)
            else:
                if osp.exists(path):
                    print_Extension(f"Check Success: {key}: {path}", frontColor=32)
                else:
                    print_Extension(f"Check Failed: {key}: {path}", frontColor=31)

if __name__ == "__main__":
    driver100_paths = Driver100DatasetPaths('100Driver')
    driver100_labels = Driver100Labels()
    driver100_labels.check()

    def generate_setting(name, modal, train_paths, train_labels, val_paths, val_labels, test_path, test_label, classes, info):
        return {
            "DataName": name,
            "modal": modal,
            "ImageTrainPath": train_paths,
            "TrainLabelPath": train_labels,
            "ImageValPath": val_paths,
            "ValLabelPath": val_labels,
            "ImageTestPath": test_path,
            "TestLabelPath": test_label,
            "classes": classes,
            "class_names": None,
            "info": info,
        }

    Driver100_Tradition_Setting = generate_setting(
        "SFD_augment",
        "rgb",
        [f"{OS_ROOT_DIR}/SFD_Augment"],
        [r"train/groundtruth/SFD/trainLabel(224)_augment.txt"],
        [f"{OS_ROOT_DIR}/SFD"],
        [r"train/groundtruth/SFD/testLabel(224).txt"],
        f"{OS_ROOT_DIR}/SFD",
        r"train/groundtruth/SFD/testLabel(224).txt",
        10,
        ["SFD augment dataset"]
    )

    camera_settings = {
        f"Driver100_Cross_Camera_Setting_D{i}": {
            "train_paths": [driver100_paths.day_cameras[f"Day_Cam{i}"], driver100_paths.day_cameras_aug[f"Day_Cam{i}_Aug"]],
            "train_labels": [driver100_labels.cross_camera_labels[f"CCS_D{i}_Train"], driver100_labels.cross_camera_labels[f"CCS_D{i}_Train_A"]],
            "val_paths": [driver100_paths.day_cameras[f"Day_Cam{i}"]],
            "val_labels": [driver100_labels.cross_camera_labels[f"CCS_D{i}_Val"]],
            "test_path": driver100_paths.day_cameras[f"Day_Cam{i}"],
            "test_label": driver100_labels.cross_camera_labels[f"CCS_D{i}_Test"],
            "info": [f"Driver100_Cross_Camera_Setting Train:D{i} Val:D{i} Test:D{i}"]
        } for i in range(1, 5)
    }

    for setting_name, paths in camera_settings.items():
        globals()[setting_name] = generate_setting(
            setting_name,
            "rgb",
            paths["train_paths"],
            paths["train_labels"],
            paths["val_paths"],
            paths["val_labels"],
            paths["test_path"],
            paths["test_label"],
            22,
            paths["info"]
        )

    individual_vehicle_settings = {
        f"Driver100_Cross_Individual_Vehicle_D{i}_Mazda": {
            "train_paths": [driver100_paths.day_cameras[f"Day_Cam{i}"], driver100_paths.day_cameras_aug[f"Day_Cam{i}_Aug"]],
            "train_labels": [driver100_labels.cross_individual_vehicle_labels[f"CIV_D{i}_Mazda_Train"], driver100_labels.cross_individual_vehicle_labels[f"CIV_D{i}_Mazda_Train_A"]],
            "val_paths": [driver100_paths.day_cameras[f"Day_Cam{i}"]],
            "val_labels": [driver100_labels.cross_individual_vehicle_labels[f"CIV_D{i}_Mazda_Test"]],
            "test_path": driver100_paths.day_cameras[f"Day_Cam{i}"],
            "test_label": driver100_labels.cross_individual_vehicle_labels[f"CIV_D{i}_Mazda_Test"],
            "info": [f"Driver100_Cross_Individual_Vehicle_D{i}_Mazda Train:Mazda_Train Val:Mazda_Val Test:Mazda_Test"]
        } for i in range(1, 5)
    }

    for setting_name, paths in individual_vehicle_settings.items():
        globals()[setting_name] = generate_setting(
            setting_name,
            "rgb",
            paths["train_paths"],
            paths["train_labels"],
            paths["val_paths"],
            paths["val_labels"],
            paths["test_path"],
            paths["test_label"],
            22,
            paths["info"]
        )

    vehicle_type_settings = {
        f"Driver100_Cross_Vehicle_Type_D{i}_Sedan": {
            "train_paths": [driver100_paths.day_cameras[f"Day_Cam{i}"], driver100_paths.day_cameras_aug[f"Day_Cam{i}_Aug"]],
            "train_labels": [driver100_labels.cross_vehicle_type_labels[f"CVT_D{i}_Sedan_Train"], driver100_labels.cross_vehicle_type_labels[f"CVT_D{i}_Sedan_Train_A"]],
            "val_paths": [driver100_paths.day_cameras[f"Day_Cam{i}"]],
            "val_labels": [driver100_labels.cross_vehicle_type_labels[f"CVT_D{i}_Sedan_Val"]],
            "test_path": driver100_paths.day_cameras[f"Day_Cam{i}"],
            "test_label": driver100_labels.cross_vehicle_type_labels[f"CVT_D{i}_Sedan_Val"],
            "info": [f"Driver100_Cross_Vehicle_Type_D{i}_Sedan Train:Sedan_Train Val:Sedan_Val Test:Sedan_Test"]
        } for i in range(1, 5)
    }

    for setting_name, paths in vehicle_type_settings.items():
        globals()[setting_name] = generate_setting(
            setting_name,
            "rgb",
            paths["train_paths"],
            paths["train_labels"],
            paths["val_paths"],
            paths["val_labels"],
            paths["test_path"],
            paths["test_label"],
            22,
            paths["info"]
        )

    # Create settings for vehicle type SUV and Van tests
    vehicle_type_tests = ["SUV", "Van"]
    for i in range(1, 5):
        for test_type in vehicle_type_tests:
            setting_name = f"Driver100_Cross_Vehicle_Type_D{i}_{test_type}_Test"
            globals()[setting_name] = generate_setting(
                setting_name,
                "rgb",
                [],
                [],
                [],
                [],
                driver100_paths.day_cameras[f"Day_Cam{i}"],
                driver100_labels.cross_vehicle_type_labels[f"CVT_D{i}_{test_type}_Test"],
                22,
                [f"Driver100_Cross_Vehicle_Type_D{i}_{test_type} Train:Sedan_Train Val:Sedan_Val Test:{test_type}_Test"]
            )
