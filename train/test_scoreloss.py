import torch
import sys
import os
sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
print(sys.path)
from pythonUtils import print_Extension, creat_folder
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams
config = {
    "font.family": 'Times New Roman',
    "axes.unicode_minus": False,
    "font.weight": "bold"
}
rcParams.update(config)
import itertools
from collections import OrderedDict
import pandas as pd
from settings import Settings
from data_module import ClassificationDataset, TransformPipeline
from model import *
from metrics import Metrics, calculate_roc_auc, calculate_pr_curve, perform_tsne
from loss import ScoreLoss

np.set_printoptions(linewidth=1000)

def draw_confusion_matrix(matrix, num_classes, class_labels=None, figure_size=None, value_format=int):
    if class_labels is None or type(class_labels) != list:
        class_labels = [str(i) for i in range(num_classes)]

    figure = plt.figure(figsize=figure_size)
    plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    ticks = np.arange(len(class_labels))
    plt.xticks(ticks, class_labels, horizontalalignment="right", rotation=45)
    plt.yticks(ticks, class_labels)

    threshold = matrix.max() / 2.0
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        color = "white" if matrix[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            format(matrix[i, j], ".2f") if (matrix[i, j] != 0 and value_format == float) else format(matrix[i, j], "d") if (matrix[i, j] != 0 and value_format == int) else ".",
            horizontalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure

def generate_confusion_matrix(matrix, num_classes, subset_ids=None, class_labels=None, figure_size=None, normalize=False):
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    if subset_ids is None or len(subset_ids) != 0:
        if class_labels is None:
            class_labels = [str(i) for i in range(num_classes)]
        if subset_ids is None:
            subset_ids = list(range(num_classes))

        sub_matrix = matrix[subset_ids, :][:, subset_ids]
        sub_names = [class_labels[j] for j in subset_ids]

        sub_matrix = draw_confusion_matrix(
            sub_matrix,
            num_classes=len(subset_ids),
            class_labels=sub_names,
            figure_size=figure_size,
            value_format=float if normalize else int,
        )
        return sub_matrix

class DataPreparation:
    def __init__(self, project_info, settings: Settings):
        self.project_info = project_info
        self.settings = settings
        self.settings.debug_mode = self.project_info.is_debug
        self.no_augment_transform = TransformPipeline(self.settings, apply_augment=False)
        self.worker_count = 6

        if ("class_samples" in self.settings.dataset_info.keys()):
            self.settings.class_samples = self.settings.dataset_info["class_samples"]

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

    def load_data(self):
        for _ in show_progress(self.test_loader, position=0, description='Loading test dataset'):
            pass

def evaluate(settings: Settings, model: nn.Module, data_loader: DataLoader, loss_func: ScoreLoss):
    if settings.class_count is None:
        print_Extension("Error: settings.class_count is None", frontColor=31)
        sys.exit()
    confusion_matrix = np.zeros((settings.class_count, settings.class_count), dtype=np.int32)

    model.eval()
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    test_labels = []
    test_logits = []
    error_samples = OrderedDict({'name': [], 'real': [], 'predict': []})
    tsne_features = []
    tsne_labels = []
    with torch.no_grad():
        for batch, data in enumerate(data_loader, start=0):
            images = data['images'].to(device=settings.device_type, dtype=settings.data_type)
            labels = data['labels'].to(device=settings.device_type)
            names = data['items']

            logits = model(images)
            logits = logits.view(-1, loss_func.scoreLevel, settings.class_count)
            logits = nn.Softmax(dim=1)(logits)

            prob, logits = get_topk_predictions(logits, labels, loss_func)
            top1_correct += prob[0].cpu()
            top3_correct += prob[1].cpu()
            top5_correct += prob[2].cpu()
            predicted_classes = prob[3].squeeze(0)
            actual_classes = labels

            batch_logits = [
                torch.max(logits[i], dim=0).values.argmax(0).item()
                for i in range(0, logits.size()[0])
            ]
            test_logits.extend(logits.to('cpu').numpy().tolist())
            test_labels.extend(labels.to('cpu').numpy().tolist())

            for i in range(0, actual_classes.shape[0]):
                confusion_matrix[actual_classes[i].item(), predicted_classes[i].item()] += 1

            actual_labels = list(labels.to('cpu').numpy())
            predicted_labels = list(logits.argmax(1).to('cpu').numpy())
            sample_names = list(names)
            for i in range(len(labels)):
                if actual_labels[i] != predicted_labels[i]:
                    error_samples['real'].append(actual_labels[i])
                    error_samples['predict'].append(predicted_labels[i])
                    error_samples['name'].append(sample_names[i])

            tsne_features.append(logits.to('cpu'))
            tsne_labels.extend(actual_classes.to('cpu').numpy())

    assert np.sum(confusion_matrix) == len(data_loader.dataset), "Evaluation result mismatch with dataset size"
    metrics = Metrics(confusion_matrix)

    test_logits = np.asarray(test_logits)
    test_labels = np.asarray(test_labels)

    roc_results = {'test_logits': test_logits, 'test_labels': test_labels, 'error_samples': error_samples}

    test_accuracy = [top1_correct / len(data_loader.dataset), top3_correct / len(data_loader.dataset), top5_correct / len(data_loader.dataset)]

    tsne_features = torch.cat(tsne_features, 0)
    tsne_labels = np.asarray(tsne_labels).astype(int)
    tsne_data = {"features": tsne_features, "labels": tsne_labels}

    return metrics, test_accuracy, roc_results, tsne_data

def get_topk_predictions(logits: torch.Tensor, labels: torch.Tensor, loss_func: ScoreLoss):
    logits = logits * loss_func.droptLowHighscore
    logits = torch.sum(logits, dim=1)
    tmp = torch.sum(logits, dim=1).unsqueeze(-1)
    logits = torch.div(logits, tmp)

    _, pred = logits.topk(5, 1, largest=True, sorted=True)
    labels = labels.view(labels.size(0), -1).expand_as(pred)
    correct = pred.eq(labels).float()

    correct_5 = correct[:, :5].sum()
    correct_3 = correct[:, :3].sum()
    correct_1 = correct[:, :1].sum()

    return [correct_1, correct_3, correct_5, pred[:, :1]], logits

class Config:
    info = ""

def load_config(file: str):
    config = Config()
    config_keys = ['Num_workers', 'Size', 'Resize', 'Classes', 'Model', 'Dtype', 'Inchannel', 'Batch_size']
    with open(file, 'r') as fh:
        info = fh.readline().strip()
        while info:
            info = fh.readline().strip()
            if len(info) == 0:
                continue
            key = re.match('\w+', info).group()
            if key in config_keys:
                key, value = info.split(': ')
                if key in ['Num_workers', 'Classes', 'Inchannel', 'Batch_size']:
                    setattr(config, key.lower().replace(' ', ''), int(value.replace(' ', '')))
                elif key in ['Resize']:
                    setattr(config, key.lower().replace(' ', ''), bool(value.replace(' ', '')))
                elif key in ['Size']:
                    value = value.replace(' ', '')
                    value = re.search('\d+', value)[0]
                    setattr(config, key.lower().replace(' ', ''), (int(value), int(value)))
                elif key in ['Dtype']:
                    setattr(config, key.lower().replace(' ', ''), eval(value.replace(' ', '')))
                else:
                    setattr(config, key.lower().replace(' ', ''), value.replace(' ', ''))

    return config

if __name__ == '__main__':
    import argparse
    import datasetDriver100
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', default='default', type=str, required=False)
    parser.add_argument('-dataset', default='default', type=str, required=False)
    parser.add_argument('-level', default=-1, type=int, required=False)
    args = parser.parse_args()

    if args.level == -1:
        args.level = 10
    else:
        pass

    project = ProjectInfo()
    if args.ckpt == 'default':
        ckptFolderFolder = '/home/vivan/Documents/Distracted/train/log/FineturnTrainScoreLossPlus-2024-05-19-18_44_23'
    else:
        ckptFolderFolder = args.ckpt

    ckptFolders = os.listdir(ckptFolderFolder)

    for ckptFolder in ckptFolders:
        if ckptFolder != '2024-05-11-13_41_27':
            continue
        print_Extension(f"Start: {ckptFolder}", frontColor=32)
        cfgFile = osp.join(ckptFolderFolder, ckptFolder, 'config.txt')
        opt = load_config(cfgFile)

        files = os.listdir(osp.join(ckptFolderFolder, ckptFolder))
        for file in files:
            template = f'{opt.model}\S+valBest-0.pth'
            if re.search(template, file):
                ckptFile = re.search(template, file).group()

        ckptFile = osp.join(ckptFolderFolder, ckptFolder, ckptFile)

        if args.dataset == 'default':
            testDataset = datasetDriver100.Driver100_Cross_Individual_Vehicle_D4_Lynk_Test
        else:
            testDataset = eval(args.dataset)
        opt.dataset = testDataset
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        net, level = eval(opt.model)(opt, level=args.level)
        state_dict = torch.load(ckptFile)
        print_Extension("Loaded the pretrain model!\n", 32)
        net.load_state_dict(state_dict=state_dict, strict=False)
        if opt.device == "cuda":
            net.to('cuda')
            cudnn.benchmark = True

        loss_f = ScoreLoss(opt, level)

        prepared_data = DataPreparation(project, opt)

        metrics, test_acc, roc_results, tsne_results = evaluate(opt, net, prepared_data.test_loader, loss_f)

        result_folder = osp.join(ckptFolderFolder, ckptFolder, f'Test_Result_On_{opt.dataset["DataName"]}')
        creat_folder(result_folder)
        resultFile = osp.join(result_folder, 'result.txt')

        info = 'Test | ACC: %.5f%% | Pre: %.5f%% | Recall: %.5f%% | F-1: %.5f%% | ACC-Top1: %.3f%% | ACC-Top3: %.3f%% | ACC-Top5: %.3f%% |' % (100. * metrics.accuracy, 100. * metrics.mean_precision, 100. * metrics.mean_sensitivity, 100. * metrics.Macro_F1, 100. * test_acc[0], 100. * test_acc[1], 100. * test_acc[2])
        print_Extension(info, _file=resultFile)

        print_Extension('Export Result ...', frontColor=32)
        metrics.export_result(resultFile)

        print_Extension('Export confusion matrix ...', frontColor=32)
        class_labels = opt.dataset["class_names"] if opt.dataset["class_names"] else None
        confusion_matrix_figure = generate_confusion_matrix(metrics.confusion_matrix, class_labels=class_labels, num_classes=opt.classes, figure_size=[12.8, 9.6], normalize=False)
        confusion_matrix_figure.savefig(osp.join(result_folder, 'confusion_matrix.png'))

        print_Extension('Export ROC, PR ...', frontColor=32)
        roc_macro, macro_data, micro_data = calculate_roc_auc(prob=roc_results['test_logits'], labels=roc_results['test_labels'], samplesNum=len(prepared_data.testSet), n_classes=opt.classes)
        roc_macro.savefig(osp.join(result_folder, 'macroROCandAUC.png'))
        np.savetxt(osp.join(result_folder, "ROCDataMacro.csv"), macro_data, fmt="%.3f", delimiter=',', header="x, y")
        np.savetxt(osp.join(result_folder, "ROCDataMicro.csv"), micro_data, fmt="%.3f", delimiter=',', header="x, y")

        pr_macro, pr_macro_data = calculate_pr_curve(prob=roc_results['test_logits'], labels=roc_results['test_labels'], samplesNum=len(prepared_data.testSet), n_classes=opt.classes)
        pr_macro.savefig(osp.join(result_folder, 'macroPR.png'))
        np.savetxt(osp.join(result_folder, "PRDataMacro.csv"), pr_macro_data, fmt="%.3f", delimiter=',', header="x, y")

        print_Extension('Export faultSamples ...', frontColor=32)
        fault_samples = pd.DataFrame(roc_results['error_samples'])
        fault_samples.to_csv(os.path.join(result_folder, "faultsamples.txt"), sep='\t', index=0)

        print_Extension('Export t-SNE ...', frontColor=32)
        perform_tsne(opt, tsne_results, result_folder)

        print_Extension('Finished!', frontColor=32)
    sys.exit()
