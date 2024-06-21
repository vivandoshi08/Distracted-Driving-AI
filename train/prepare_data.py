from pythonUtils import show_progress
from settings import Settings
from data_module import ClassificationDataset, TransformPipeline
from torch.utils.data import DataLoader, ConcatDataset

class DatasetSetup:
    def __init__(self, project_info, settings: Settings):
        self.project_info = project_info
        self.settings = settings
        self.settings.debug_mode = self.project_info.is_debug
        self.augment_transform = TransformPipeline(self.settings, apply_augment=True)
        self.no_augment_transform = TransformPipeline(self.settings, apply_augment=False)
        self.worker_count = 6

        self.settings.class_count = self.settings.dataset_info["classes"]

        # Training set
        if isinstance(self.settings.dataset_info["ImageTrainPath"], str):
            self.train_dataset = ClassificationDataset(
                self.settings,
                self.settings.dataset_info,
                self.settings.dataset_info["ImageTrainPath"],
                osp.join(self.project_info.ROOT, self.settings.dataset_info["TrainLabelPath"]),
                transform=self.no_augment_transform
            )
        elif isinstance(self.settings.dataset_info["ImageTrainPath"], list):
            self.train_dataset = ConcatDataset([
                ClassificationDataset(
                    self.settings,
                    self.settings.dataset_info,
                    path,
                    osp.join(self.project_info.ROOT, self.settings.dataset_info["TrainLabelPath"][i]),
                    transform=self.no_augment_transform
                ) for i, path in enumerate(self.settings.dataset_info["ImageTrainPath"])
            ])
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.settings.batch_size,
            shuffle=True,
            num_workers=self.worker_count,
            pin_memory=True
        )
        
        # Validation set
        if isinstance(self.settings.dataset_info["ImageValPath"], str):
            self.val_dataset = ClassificationDataset(
                self.settings,
                self.settings.dataset_info,
                self.settings.dataset_info["ImageValPath"],
                osp.join(self.project_info.ROOT, self.settings.dataset_info["ValLabelPath"]),
                transform=self.no_augment_transform
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.settings.batch_size,
                shuffle=False,
                num_workers=self.worker_count,
                pin_memory=True
            )
        elif isinstance(self.settings.dataset_info["ImageValPath"], list):
            self.val_datasets = [
                ClassificationDataset(
                    self.settings,
                    self.settings.dataset_info,
                    path,
                    osp.join(self.project_info.ROOT, self.settings.dataset_info["ValLabelPath"][i]),
                    transform=self.no_augment_transform
                ) for i, path in enumerate(self.settings.dataset_info["ImageValPath"])
            ]
            self.val_loaders = [
                DataLoader(
                    dataset,
                    batch_size=self.settings.batch_size,
                    shuffle=False,
                    num_workers=self.worker_count,
                    pin_memory=True
                ) for dataset in self.val_datasets
            ]
        
        if "class_samples" in self.settings.dataset_info:
            self.settings.class_samples = self.settings.dataset_info["class_samples"]
        
        # Test set
        if self.settings.dataset_info["TestLabelPath"] is not None:
            if isinstance(self.settings.dataset_info["TestLabelPath"], str):
                self.test_dataset = ClassificationDataset(
                    self.settings,
                    self.settings.dataset_info,
                    self.settings.dataset_info["ImageTestPath"],
                    osp.join(self.project_info.ROOT, self.settings.dataset_info["TestLabelPath"]),
                    transform=self.no_augment_transform
                )
                self.test_loader = DataLoader(
                    self.test_dataset,
                    batch_size=self.settings.batch_size,
                    shuffle=False,
                    num_workers=self.worker_count,
                    pin_memory=True
                )
            elif isinstance(self.settings.dataset_info["TestLabelPath"], list):
                self.test_datasets = [
                    ClassificationDataset(
                        self.settings,
                        self.settings.dataset_info,
                        path,
                        osp.join(self.project_info.ROOT, self.settings.dataset_info["TestLabelPath"][i]),
                        transform=self.no_augment_transform
                    ) for i, path in enumerate(self.settings.dataset_info["ImageTestPath"])
                ]
                self.test_loaders = [
                    DataLoader(
                        dataset,
                        batch_size=self.settings.batch_size,
                        shuffle=False,
                        num_workers=self.worker_count,
                        pin_memory=True
                    ) for dataset in self.test_datasets
                ]

    def load_data(self):
        if isinstance(self.val_loader, list):
            for loader in self.val_loaders:
                for _ in show_progress(loader, position=0, description='Loading validation dataset'):
                    pass
        else:   
            for _ in show_progress(self.val_loader, position=0, description='Loading validation dataset'):
                pass
        
        for _ in show_progress(self.train_loader, position=0, description='Loading training dataset'):
            pass
        
        if self.settings.dataset_info["TestLabelPath"] is not None:
            if isinstance(self.test_loader, list):
                for loader in self.test_loaders:
                    for _ in show_progress(loader, position=0, description='Loading test dataset'):
                        pass
            else:
                for _ in show_progress(self.test_loader, position=0, description='Loading test dataset'):
                    pass
