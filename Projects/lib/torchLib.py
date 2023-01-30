import math
import os
from typing import Any
import math
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import random
from pathlib import Path
WORKERS = os.cpu_count()
device = "cuda" if torch.cuda.is_available() else "cpu"

def create_dataloader_from_dir(
        train_dir, test_dir, transform, batch_size
):
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    class_names = train_data.classes

    train_loader = DataLoader(train_data, batch_size, True, num_workers=WORKERS, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size, False, num_workers=WORKERS, pin_memory=True)
    return train_loader, test_loader, class_names


def train_step(
        model, dataloader, loss_fn, optimizer
):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred_logit = model(x)
        loss = loss_fn(pred_logit, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(torch.softmax(pred_logit, dim=1), dim=1)
        train_acc += (preds == y).sum().item() / len(preds)
    train_loss, train_acc = train_loss / len(dataloader), train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
        model, dataloader, loss_fn
):
    model.eval()
    train_loss, train_acc = 0,0
    with torch.inference_mode():
        for batch, (x,y) in enumerate(dataloader):
            x,y = x.to(device), y.to(device)
            pred_logit = model(x)
            loss = loss_fn(pred_logit, y)
            train_loss += loss.item()

            preds = torch.argmax(torch.softmax(pred_logit, dim=1), dim=1)
            train_acc += (preds == y).sum().item()/ len(preds)
    train_loss, train_acc = train_loss/len(dataloader), train_acc/len(dataloader)
    return train_loss, train_acc

def train(
        model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs
):
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }
    model.to(device)
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results

def get_transforms(transform_options, transform):
    if transform is not None:
        image_transform = transform
    elif transform_options is None:
        image_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        trans_list = []
        for o in transform_options:
            if "resize" in o:
                size = int(o.replace("resize", ""))
                trans_list.append(transforms.Resize(size))
        for o in transform_options:
            if "flip" in o:
                prob = int(o.replace("flip", ""))/10
                trans_list.append(transforms.RandomHorizontalFlip(prob))
        for o in transform_options:
            if "rotate" in o:
                deg = int(o.replace("rotate", ""))
                trans_list.append(transforms.RandomRotation(deg))
        if "bins" in transform_options:
            trans_list.append(transforms.TrivialAugmentWide(31))
        trans_list.append(transforms.ToTensor())
        if "norm" in transform_options:
            trans_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ))
        image_transform = transforms.Compose(trans_list)
def pred_and_plot_img(
        model, class_names, img_path, img_size, transform, device, transform_options
):
    img = Image.open(img_path)
    image_transform = get_transforms(transform)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        trans_img = image_transform(img).unsqueese(dim=0)
        pred_logit = model(trans_img.to(device))
    target_image_pred_probs = torch.softmax(pred_logit, dim=1)

    label = torch.argmax(target_image_pred_probs, dim=1)
    plt.imshow(img)
    plt.title(f"Pred: {class_names[label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)

def plot_random(data, class_names):
    fig = plt.figure(figsize = (9,9))
    rows, cols = 4, 4
    for i in range(1, rows*cols + 1):
        randIndex = torch.randint(0, len(data), size=[1]).item()
        img, label = data[randIndex]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(class_names[label])
        plt.axis(False)

def make_predictions(model, data: list):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            #Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model.forward(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())
    return torch.stack(pred_probs)

def plot_rand_preds(data, model, class_names):
    test_samples = []
    test_labels = []
    num = 20
    for sample, label in random.sample(list(data), k=num):
        test_samples.append(sample)
        test_labels.append(label)

    # View the first test sample shape and label
    print(f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})")

    pred_probs = make_predictions(model, test_samples)
    pred_classes = pred_probs.argmax(dim=1)

    # Plot predictions
    plt.figure(figsize=(12, 12))
    nrows = round(math.sqrt(num))
    ncols = round(num/nrows)
    for i, sample in enumerate(test_samples):
        # Create a subplot
        plt.subplot(nrows, ncols, i+1)

        # Plot the target image
        plt.imshow(sample.squeeze(), cmap="gray")

        # Find the prediction label (in text form, e.g. "Sandal")
        pred_label = class_names[pred_classes[i]]

        # Get the truth label (in text form, e.g. "T-shirt")
        truth_label = class_names[test_labels[i]]

        # Create the title text of the plot
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"

        # Check for equality and change title colour accordingly
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g") # green text if correct
        else:
            plt.title(title_text, fontsize=10, c="r") # red text if wrong
        plt.axis=False;

def plot_wrong_preds(test_data, model, class_names, num=20):
    #Show mistakes
    plt.figure(figsize=(12, 12))
    nrows = round(math.sqrt(num))
    ncols = round(num/nrows)

    all_samples = []
    all_labels = []
    for sample, label in random.sample(list(test_data), k=5):
        all_samples.append(sample)
        all_labels.append(label)

    all_pred_probs = make_predictions(model, all_samples)
    all_pred_classes = all_pred_probs.argmax(dim=1)

    showed = 0
    for i, sample in enumerate(all_samples):
        # Find the prediction label (in text form, e.g. "Sandal")
        pred_label = class_names[all_pred_classes[i]]

        # Get the truth label (in text form, e.g. "T-shirt")
        truth_label = class_names[all_labels[i]]

        # Check for equality and change title colour accordingly
        if pred_label != truth_label:
            if showed == 20:
                break
            plt.subplot(nrows, ncols, showed+1)
            plt.imshow(sample.squeeze(), cmap="gray")
            title_text = f"Pred: {pred_label} | Truth: {truth_label}"
            plt.title(title_text, fontsize=10, c="r") # red text if wrong
            showed+=1
        plt.axis=False;

def plot_loss_curves(results):
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)