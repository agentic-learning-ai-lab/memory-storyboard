import torch
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from eval_datasets import ImageList
from torchvision import transforms

IMAGENET_ROOT = '/path_to_imagenet'  # Update this path to your ImageNet root directory
LABELEDS_ROOT = '/path_to_labeledS'  # Update this path to your LabeledS root directory

def get_svm_perf(train_features, test_features, train_labels, test_labels):
    alpha_list = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]
    all_perf = []
    for _alpha in alpha_list:
        clf = SGDClassifier(alpha=_alpha, n_jobs=-1)
        clf.fit(train_features, train_labels)
        _perf = clf.score(test_features, test_labels)
        print(f"-_perf: {_perf:.04f}")
        all_perf.append(_perf)
    return max(all_perf)

def get_knn_perf(train_features, test_features, train_labels, test_labels):
    k_values = [1, 3, 5, 10, 20, 50, 100, 150, 200, 250]
    best_k = None
    best_accuracy = 0
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_features, train_labels)
        val_predictions = knn.predict(test_features)
        accuracy = accuracy_score(test_labels, val_predictions)
        print(f"Accuracy for k={k}: {accuracy:.4f}")        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    return best_k, best_accuracy

def get_imagenet_dataloader(batch_size=128, num_workers=8, image_size=112):
    test_pipeline = transforms.Compose([
        transforms.Resize(size=256 * image_size // 224),
        transforms.CenterCrop(size=image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imagenet_dataset = ImageList(list_file='metas/miniinet_train_val.txt', root=IMAGENET_ROOT, pipeline=test_pipeline)
    imagenet_loader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return imagenet_loader

def get_labeledS_dataloader(batch_size=128, num_workers=8, image_size=112):
    test_pipeline = transforms.Compose([
        transforms.Resize(size=256 * image_size // 224),
        transforms.CenterCrop(size=image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    labeledS_dataset = ImageList(list_file='metas/labeledS_train_val.txt', root=LABELEDS_ROOT, pipeline=test_pipeline)
    labeledS_loader = torch.utils.data.DataLoader(labeledS_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return labeledS_loader

