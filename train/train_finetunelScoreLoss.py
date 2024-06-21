import sys
import os
import torch
import torchvision.utils
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter
from options import Options
import train_config as cfg
from data_prep import DataPreparation
from custom_loss import CustomScoreLoss
from custom_model import *
from evaluation import MetricEvaluator, calculate_metrics
from utility_functions import *

np.set_printoptions(linewidth=1000)
sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

project_folder = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

def train_model(project: ProjectDetails, settings: Options, data_prep: DataPreparation, score_level):
    log_dir = os.path.join(project.ROOT, "logs", f"{'Debug' if project.IsDebug else 'FinetuneTraining'}-{project.StartTime}", get_time_stamp())
    os.makedirs(log_dir, exist_ok=True)

    main_log = os.path.join(log_dir, "main_log.txt")
    training_log = os.path.join(log_dir, "training_log.txt")
    val_log = os.path.join(log_dir, "val_log.txt") if not isinstance(data_prep.val_loader, list) else [os.path.join(log_dir, f"val_log_{i}.txt") for i in range(len(data_prep.val_loader))]
    test_log = os.path.join(log_dir, "test_log.txt") if not isinstance(data_prep.test_loader, list) else [os.path.join(log_dir, f"test_log_{i}.txt") for i in range(len(data_prep.test_loader))]

    config_file = os.path.join(log_dir, "config.txt")
    summary_writer = SummaryWriter(os.path.join(log_dir, 'runs'))

    settings.print_info(config_file)
    model, score_level = eval(settings.model)(settings, score_level)
    if settings.write:
        summary_writer.add_graph(model, torch.rand(1, settings.in_channel, *settings.image_size))

    if settings.pretrain and os.path.exists(settings.pretrain_path):
        model_state = torch.load(settings.pretrain_path)
        log_message("Loaded pretrained model", 32, main_log)
        model.load_state_dict(state_dict=model_state, strict=False)

    if settings.device == "cuda":
        model.to('cuda')
        cudnn.benchmark = True
    if settings.device == "cuda" and settings.multi_gpu:
        model = nn.DataParallel(model)

    loss_function = CustomScoreLoss(settings, score_level)
    loss_function.print_info(file=main_log)

    model.requires_grad_(True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.000001, betas=(0.9, 0.999), weight_decay=0.001)
    
    for name, param in model.named_parameters():
        log_message(f'{name} {param.requires_grad}', main_log)
    
    scheduler = MultiStepLR(optimizer, milestones=settings.lr_decay_steps, gamma=settings.lr_decay_rate)

    start_epoch = 1
    global_step = 0
    best_val_accuracy = [-1]*len(data_prep.val_loader) if isinstance(data_prep.val_loader, list) else -1
    best_test_accuracy = [-1]*len(data_prep.test_loader) if isinstance(data_prep.test_loader, list) else -1
    
    num_epochs = 5 if project.IsDebug else settings.epochs
    settings.test_interval = 1 if project.IsDebug else settings.test_interval
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_loss = 0.0
        epoch_score_diff = 0.0
        epoch_score_avg = 0.0
        correct_1 = 0
        correct_3 = 0
        correct_5 = 0
        total_samples = 0
        batch_count = 0
        for batch_index, data in enumerate(data_prep.train_loader, start=0):
            model.train()
            global_step += 1
            batch_count += 1
            images = data['images'].to(device=settings.device, dtype=settings.dtype)
            labels = data['labels'].to(device=settings.device)
            outputs = model(images) 
            outputs = outputs.view(-1, score_level, settings.num_classes)
            outputs = nn.Softmax(dim=1)(outputs)
            
            loss, score_diff, score_avg = loss_function(outputs, labels)

            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_score_diff = score_diff.item()
            batch_score_avg = score_avg.item()
            epoch_score_diff += batch_score_diff
            epoch_score_avg += batch_score_avg

            probs = calculate_predictions(outputs, labels, loss_function)
            correct_1 += probs[0].cpu()
            correct_3 += probs[1].cpu()
            correct_5 += probs[2].cpu()
            total_samples += labels.size(0)

            if global_step % 50 == 0:
                log_message(
                    f'Train    | Epoch: {epoch} (lr={scheduler.get_last_lr()[0]}) | Batch: {batch_index + 1} | Step: {global_step} | '
                    f'Batch Loss: {batch_loss:.5f} ({batch_score_diff:.5f}, {batch_score_avg:.5f}) | '
                    f'Epoch Loss: {epoch_loss / (batch_index + 1):.5f} ({epoch_score_diff / (batch_index + 1):.5f}, {epoch_score_avg / (batch_index + 1):.5f}) | '
                    f'Acc-Top1: {100. * float(correct_1) / total_samples:.3f}% ({correct_1}/{total_samples}) | '
                    f'Acc-Top3: {100. * float(correct_3) / total_samples:.3f}% ({correct_3}/{total_samples}) | '
                    f'Acc-Top5: {100. * float(correct_5) / total_samples:.3f}% ({correct_5}/{total_samples}) |',
                    training_log
                )

            if global_step % (len(data_prep.train_loader) // settings.test_interval) == 0:
                if isinstance(data_prep.val_loader, list):
                    for i, val_loader in enumerate(data_prep.val_loader):
                        val_result = evaluate_model(settings, model, val_loader, loss_function)
                        log_message(
                            f'Evaluate-{i} | Epoch: {epoch} (lr={scheduler.get_last_lr()[0]}) | Batch: {batch_index + 1} | Step: {global_step} | '
                            f'Val Loss: {val_result[3]:.5f} ({val_result[4]:.5f}, {val_result[5]:.5f}) | '
                            f'Acc-Top1: {100. * val_result[0]:.3f}% | Acc-Top3: {100. * val_result[1]:.3f}% | Acc-Top5: {100. * val_result[2]:.3f}% |',
                            val_log[i]
                        )
                        if val_result[0] > best_val_accuracy[i]:
                            best_val_accuracy[i] = max(val_result[0], best_val_accuracy[i])
                            ckpt_name = f'{settings.model}-{settings.dataset["name"]}-valBest-{i}.pth'
                            save_checkpoint(settings, model, os.path.join(log_dir, ckpt_name))
                        if settings.write:
                            summary_writer.add_scalars(f'Val-{i}', {'val_top-1': 100. * val_result[0], 'val_top-3': 100. * val_result[1], 'val_top-5': 100. * val_result[2]}, global_step)
                            summary_writer.add_scalars('loss', {f'val_loss-{i}': val_result[3]}, global_step)
                            summary_writer.add_scalars('score_diff', {f'score_diff-{i}': val_result[4]}, global_step)
                            summary_writer.add_scalars('score_avg', {f'score_avg-{i}': val_result[5]}, global_step)
                else:     
                    val_result = evaluate_model(settings, model, data_prep.val_loader, loss_function)
                    log_message(
                        f'Evaluate | Epoch: {epoch} (lr={scheduler.get_last_lr()[0]}) | Batch: {batch_index + 1} | Step: {global_step} | '
                        f'Val Loss: {val_result[3]:.5f} ({val_result[4]:.5f}, {val_result[5]:.5f}) | '
                        f'Acc-Top1: {100. * val_result[0]:.3f}% | Acc-Top3: {100. * val_result[1]:.3f}% | Acc-Top5: {100. * val_result[2]:.3f}% |',
                        val_log
                    )
                    if val_result[0] > best_val_accuracy:
                        best_val_accuracy = max(val_result[0], best_val_accuracy)
                        ckpt_name = f'{settings.model}-{settings.dataset["name"]}-valBest.pth'
                        save_checkpoint(settings, model, os.path.join(log_dir, ckpt_name))
                    if settings.write:
                        summary_writer.add_scalars('Val', {'val_top-1': 100. * val_result[0], 'val_top-3': 100. * val_result[1], 'val_top-5': 100

. * val_result[2]}, global_step)
                        summary_writer.add_scalars('loss', {'val_loss': val_result[3]}, global_step)
                        summary_writer.add_scalars('score_diff', {'score_diff': val_result[4]}, global_step)
                        summary_writer.add_scalars('score_avg', {'score_avg': val_result[5]}, global_step)
                            
                if settings.write:
                    summary_writer.add_scalars('loss', {'train_loss': epoch_loss / (batch_index + 1)}, global_step)
                    summary_writer.add_scalars('Train', {'train_top-1': 100. * float(correct_1) / total_samples, 'train_top-3': 100. * float(correct_3) / total_samples, 'train_top-5': 100. * float(correct_5) / total_samples}, global_step)
                    summary_writer.add_scalars('lr', {'learning_rate': scheduler.get_last_lr()[0], 'epoch': epoch}, global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if settings.dataset["test_labels"] is not None:
            if isinstance(data_prep.test_loader, list):
                for i, test_loader in enumerate(data_prep.test_loader):
                    metric, test_acc = test_model(settings, model, test_loader, loss_function)
                    log_message(
                        f'Test     | Epoch: {epoch} (lr={scheduler.get_last_lr()[0]}) | ACC: {100.*metric.accuracy:.5f}% | '
                        f'Precision: {100.*metric.mean_precision:.5f}% | Recall: {100.*metric.mean_sensitivity:.5f}% | F-1: {100.*metric.Macro_F1:.5f}% | '
                        f'ACC-Top1: {100.*test_acc[0]:.3f}% | ACC-Top3: {100.*test_acc[1]:.3f}% | ACC-Top5: {100.*test_acc[2]:.3f}% |',
                        test_log[i]
                    )
                    if settings.write:
                        summary_writer.add_scalars(f'Test-{i}', {'test_top-1': 100. * test_acc[0], 'test_top-3': 100. * test_acc[1], 'test_top-5': 100. * test_acc[2]}, epoch)
                    if metric.accuracy > best_test_accuracy[i]:
                        best_test_accuracy[i] = max(metric.accuracy, best_test_accuracy[i])
                        ckpt_name = f'{settings.model}-{settings.dataset["name"]}-testBest-{i}.pth'
                        save_checkpoint(settings, model, os.path.join(log_dir, ckpt_name))
            else:
                metric, test_acc = test_model(settings, model, data_prep.test_loader, loss_function)
                log_message(
                    f'Test     | Epoch: {epoch} (lr={scheduler.get_last_lr()[0]}) | ACC: {100.*metric.accuracy:.5f}% | '
                    f'Precision: {100.*metric.mean_precision:.5f}% | Recall: {100.*metric.mean_sensitivity:.5f}% | F-1: {100.*metric.Macro_F1:.5f}% | '
                    f'ACC-Top1: {100.*test_acc[0]:.3f}% | ACC-Top3: {100.*test_acc[1]:.3f}% | ACC-Top5: {100.*test_acc[2]:.3f}% |',
                    test_log
                )
                if settings.write:
                    summary_writer.add_scalars('Test', {'metric-top1': metric.accuracy, 'test_top-1': 100. * test_acc[0], 'test_top-3': 100. * test_acc[1], 'test_top-5': 100. * test_acc[2]}, epoch)
                if metric.accuracy > best_test_accuracy:
                    best_test_accuracy = max(metric.accuracy, best_test_accuracy)
                    ckpt_name = f'{settings.model}-{settings.dataset["name"]}-testBest.pth'
                    save_checkpoint(settings, model, os.path.join(log_dir, ckpt_name))

        if settings.write:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    summary_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                    summary_writer.add_histogram(f'{name}/grad', param.grad.clone().cpu().data.numpy(), epoch)

        scheduler.step()
        
    calculate_metrics(main_log, 'Acc-Top1', 5, [*val_log, *test_log])

def evaluate_model(settings: Options, model: nn.Module, data_loader: DataLoader, loss_function: CustomScoreLoss):
    epoch_loss, epoch_score_diff, epoch_score_avg = 0.0, 0.0, 0.0
    model.eval()
    correct_1, correct_3, correct_5 = 0, 0, 0
    
    with torch.no_grad():
        for data in data_loader:  
            images = data['images'].to(device=settings.device, dtype=settings.dtype)
            labels = data['labels'].to(device=settings.device)

            outputs = model(images)
            outputs = outputs.view(-1, loss_function.scoreLevel, settings.num_classes)
            outputs = nn.Softmax(dim=1)(outputs)

            loss, score_diff, score_avg = loss_function(outputs, labels)
            epoch_loss += loss.item() * len(labels)
            epoch_score_diff += score_diff.item() * len(labels)
            epoch_score_avg += score_avg.item() * len(labels)

            probs = calculate_predictions(outputs, labels, loss_function)
            correct_1 += probs[0].cpu()
            correct_3 += probs[1].cpu()
            correct_5 += probs[2].cpu()

    dataset_size = len(data_loader.dataset)
    return [correct_1 / dataset_size, correct_3 / dataset_size, correct_5 / dataset_size, epoch_loss / dataset_size, epoch_score_diff / dataset_size, epoch_score_avg / dataset_size]

def test_model(settings: Options, model: nn.Module, data_loader: DataLoader, loss_function: CustomScoreLoss):
    if settings.num_classes is None:
        log_message("Error: settings.num_classes is None", frontColor=31)
        sys.exit()
    
    confusion_matrix = np.zeros((settings.num_classes, settings.num_classes), dtype=np.int_)
    model.eval()
    correct_1, correct_3, correct_5 = 0, 0, 0

    with torch.no_grad():
        for data in data_loader:  
            images = data['images'].to(device=settings.device, dtype=settings.dtype)
            labels = data['labels'].to(device=settings.device)

            outputs = model(images)
            outputs = outputs.view(-1, loss_function.scoreLevel, settings.num_classes)
            outputs = nn.Softmax(dim=1)(outputs)

            probs = calculate_predictions(outputs, labels, loss_function)
            correct_1 += probs[0].cpu()
            correct_3 += probs[1].cpu()
            correct_5 += probs[2].cpu()

            predicted_classes = probs[3].squeeze(0)
            true_classes = labels

            for i in range(true_classes.shape[0]):
                confusion_matrix[true_classes[i].item(), predicted_classes[i].item()] += 1

    assert np.sum(confusion_matrix) == len(data_loader.dataset), "Evaluation results are incorrect, total confusion matrix count does not match dataset size"
    metric = MetricEvaluator(confusion_matrix)
    return metric, [correct_1 / len(data_loader.dataset), correct_3 / len(data_loader.dataset), correct_5 / len(data_loader.dataset)]

def save_checkpoint(settings: Options, model: nn.Module, save_path):
    if settings.multi_gpu:
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)

def calculate_predictions(outputs: torch.Tensor, labels: torch.Tensor, loss_function: CustomScoreLoss):
    outputs = outputs * loss_function.score_matrix
    outputs = torch.sum(outputs, dim=1)
    
    _, predicted = outputs.topk(5, 1, largest=True, sorted=True)
    labels = labels.view(labels.size(0), -1).expand_as(predicted)
    correct = predicted.eq(labels).float()

    correct_5 = correct[:, :5].sum()
    correct_3 = correct[:, :3].sum()
    correct_1 = correct[:, :1].sum()
    return [correct_1, correct_3, correct_5, predicted[:, :1]]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='project')
    parser.add_argument('--dataset', default='None', type=str, help='train dataset')
    parser.add_argument('--level', action='append', help='train dataset')
    args = parser.parse_args()

    project_info = ProjectDetails()
    random.seed(project_info.Seed)
    torch.manual_seed(project_info.Seed)
    torch.cuda.manual_seed(project_info.Seed)
    torch.cuda.manual_seed_all(project_info.Seed)
    np.random.seed(project_info.Seed)

    opt = Options(cfg)
    if args.dataset != 'None':
        from datasetDriver100 import *
        opt.dataset = eval(args.dataset)

    prepared_data = DataPreparation(project_info, settings=opt)
    prepared_data.load_data()

    level_list = [int(item) for item in args.level] if args.level else [5]

    for model_name in opt.model_group:
        opt.model = model_name
        for level in level_list:
            train_model(project_info, opt, prepared_data, level)
    print('Training complete')
    sys.exit()
