import torch 
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import random
import numpy as np
import seaborn as sns
from cfgparser import CfgParser
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from combination_dataloader import CombinationDataset
from model.ResNet_modified import ModifiedResNet


def main(cfg):
    device = torch.device(f"cuda:{cfg['GPU_ID']}") 
    
    init_seed(cfg["SEED"])
    dataset, model = load_objs(cfg, device)

    testing(dataset, model, device)


def init_seed(seed, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cuda_deterministic
    torch.backends.cudnn.benchmark = not cuda_deterministic

def load_model(cfg):
    model = ModifiedResNet(3, 64, cfg['MODEL']['CATEGORY_NUM'])
    #checkpoint = torch.load(cfg['CHECKPOINT_PATH'])
    #model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def load_objs(cfg, device):
    trfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.522, 0.475, 0.408], std=[0.118, 0.115, 0.117])
    ])
    test_dataset = CombinationDataset(cfg['TEST_CSV_DIR'], transform=trfs)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg["BATCH_SIZE"], drop_last=True, shuffle=False, num_workers=cfg["WORKERS"])
    
    model = load_model(cfg)
    return test_dataloader, model.to(device)

def save_confusion_matrix_img(conf_matrix):
    # 繪製混淆矩陣
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt="d", cbar=True)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.savefig("Confusion_Matrix.png")

def save_every_class_metrics_img(class_wise_precision, class_wise_recall, class_wise_f1, category_num):
    classes = np.arange(len(class_wise_precision))  # 類別索引，例如 0, 1, 2, ...
    bar_width = 0.25
    index = np.arange(len(classes))
    plt.figure(figsize=(15, 8))
    plt.bar(index, class_wise_precision, bar_width, label='Precision', alpha=0.7)
    plt.bar(index + bar_width, class_wise_recall, bar_width, label='Recall', alpha=0.7)
    plt.bar(index + bar_width * 2, class_wise_f1, bar_width, label='F1 Score', alpha=0.7)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Class-wise Metrics', fontsize=16)
    plt.xticks(index + bar_width, [f'Class {i}' for i in classes], rotation=45)
    plt.ylim([0, 1.05])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("Class_wise_Metrics.png")

def testing(dataset, model, device):
    all_preds = []
    all_labels = []

    for idx, (img, label) in enumerate(dataset):
        img, label = img.to(device), label.to(device)
        with torch.no_grad():
            logits = model(img)
        top_pred = logits.argmax(1, keepdim=True)

        all_preds.extend(top_pred.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    save_confusion_matrix_img(conf_matrix)

    accuracy = accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_precision = precision_score(all_labels, all_preds, average='micro')
    micro_recall = recall_score(all_labels, all_preds, average='micro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Macro Precision: {macro_precision:.4f}')
    print(f'Macro Recall: {macro_recall:.4f}')
    print(f'Macro F1 Score: {macro_f1:.4f}')
    print(f'Micro Precision: {micro_precision:.4f}')
    print(f'Micro Recall: {micro_recall:.4f}')
    print(f'Micro F1 Score: {micro_f1:.4f}')

    class_wise_precision = precision_score(all_labels, all_preds, average=None)
    class_wise_recall = recall_score(all_labels, all_preds, average=None)
    class_wise_f1 = f1_score(all_labels, all_preds, average=None)
    save_every_class_metrics_img(class_wise_precision, class_wise_recall, class_wise_f1, cfg['MODEL']['CATEGORY_NUM'])



if __name__ == "__main__":
    cfgparser = CfgParser(config_path="./cfg/CombinationFood.yml")
    cfg = cfgparser.cfg_dict
    main(cfg)