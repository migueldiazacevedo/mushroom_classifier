import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from torch import nn, optim
from torch.nn import Module
from torch.utils.data import DataLoader


def pixel_dimension_dataframe(image_directory: str) -> pd.DataFrame:
    """
    For a directory of folders containing images, create a Pandas DataFrame of pixel values for each folder

    Parameters:
        image_directory (string): path to the directory containing the images

    Returns:
        pd.DataFrame: A dataframe containing image pixel metadata
    """
    image_metadata = []
    for class_name in os.listdir(image_directory):
        class_path = os.path.join(image_directory, class_name)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        image_metadata.append(
                            {"class": class_name, "width": width, "height": height}
                        )
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    return pd.DataFrame(image_metadata)


def plot_mushroom_dimensions_by_class(pixel_df: pd.DataFrame) -> None:
    """
    Plots the distribution of image dimensions (width and height) by mushroom class.

    This function creates a side-by-side boxplot for 'width' and 'height' based on the 'class'
    column in the given DataFrame. It provides insights into how these dimensions vary across
    different mushroom types.

    Args:
        pixel_df (pd.DataFrame): A DataFrame containing the mushroom data with 'class', 'width',
                                  and 'height' columns.

    Returns:
        None: The function displays a plot, but does not return any values.
    """
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.boxplot(data=pixel_df, x="class", y="width")
    plt.title("Width by Mushroom Type")
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    sns.boxplot(data=pixel_df, x="class", y="height")
    plt.title("Height by Mushroom Type")
    plt.xticks(rotation=45)

    plt.suptitle("Distribution of Image Dimensions by Class")

    plt.tight_layout()
    plt.show()


def plot_mushroom_class_distribution(pixel_df: pd.DataFrame) -> None:
    """
    Plots the distribution of mushroom classes based on the number of images per class.

    This function generates a bar chart displaying the count of images for each mushroom type
    as defined by the 'class' column in the given DataFrame. The chart includes labels showing
    the exact count of images per class.

    Args:
        pixel_df (pd.DataFrame): A DataFrame containing a 'class' column representing different
                                  mushroom types.

    Returns:
        None: The function displays a bar plot, but does not return any values.
    """
    class_counts = pixel_df["class"].value_counts().reset_index()
    class_counts.columns = ["class", "count"]

    ax = class_counts.plot(kind="bar", x="class", y="count", legend=False)

    plt.bar_label(ax.containers[0])

    plt.xlabel("class")
    plt.ylabel("count")
    plt.xticks(rotation=45)
    plt.title("Number of images of each mushroom type")

    plt.show()


def train_model(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10,
) -> Tuple[Module, Dict[str, list]]:
    """
    Train a PyTorch model with a specified training loop and return the best model and metrics.

    Args:
        model (Module): The PyTorch model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function to optimize.
        optimizer (optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to perform computations (e.g., 'cuda' or 'cpu').
        num_epochs (int, optional): Number of epochs to train the model. Default is 10.

    Returns:
        Tuple[Module, Dict[str, list]]: The trained model with the best weights and a dictionary
        containing the training and validation loss and accuracy metrics.

    Metrics:
        - "train_loss": List of training losses for each epoch.
        - "val_loss": List of validation losses for each epoch.
        - "train_acc": List of training accuracies for each epoch.
        - "val_acc": List of validation accuracies for each epoch.

    Example:
        >>> best_model, metrics = train_model(
        ...     model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25
        ... )
    """
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects / len(loader.dataset)

            metrics[f"{phase}_loss"].append(epoch_loss)
            metrics[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f"Best Validation Accuracy: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)

    return model, metrics


def plot_metrics(metrics: Dict[str, list]) -> None:
    """
    Plot training and validation loss and accuracy over epochs.

    Args:
        metrics (Dict[str, list]): A dictionary containing the following keys:
            - "train_loss": List of training losses for each epoch.
            - "val_loss": List of validation losses for each epoch.
            - "train_acc": List of training accuracies for each epoch.
            - "val_acc": List of validation accuracies for each epoch.

    Returns:
        None: This function displays the plots but does not return any values.

    Example:
        >>> metrics = {
        ...     "train_loss": [0.8, 0.6, 0.4],
        ...     "val_loss": [0.9, 0.7, 0.5],
        ...     "train_acc": [0.7, 0.8, 0.9],
        ...     "val_acc": [0.65, 0.75, 0.85]
        ... }
        >>> plot_metrics(metrics)
    """
    epochs = range(1, len(metrics["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics["train_acc"], label="Train Accuracy")
    plt.plot(epochs, metrics["val_acc"], label="Validation Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()


def plot_confusion_matrix(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list[str],
    title: Optional[str] = None,
) -> None:
    """
    Plots a confusion matrix for the given model using the test data loader.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (torch.utils.data.DataLoader): The DataLoader object for the test dataset.
        device (torch.device): The device (CPU or GPU) where the model and data should be loaded.
        class_names (list[str]): List of class labels to display on the confusion matrix.
        title (Optional[str], optional): The title for the confusion matrix plot. Defaults to "Confusion Matrix".

    Returns:
        None: The function displays the confusion matrix plot but does not return any value.
    """
    if title is None:
        title = "Confusion Matrix"
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(title)
    plt.show()


def plot_multiclass_roc_grid(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
) -> None:
    """
    Plots ROC curves for each class in a multiclass classification problem.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (torch.utils.data.DataLoader): The DataLoader object for the test dataset.
        class_names (List[str]): List of class labels to display on the ROC curves.
        device (torch.device): The device (CPU or GPU) where the model and data should be loaded.

    Returns:
        None: The function displays a grid of ROC curves for each class but does not return any value.
    """
    y_true = []
    y_score = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.cpu().numpy()
            y_true.extend(labels)
            outputs = model(inputs)
            y_score.extend(outputs.cpu().numpy())

    y_true = label_binarize(y_true, classes=range(len(class_names)))
    y_score = np.array(y_score)
    n_classes = len(class_names)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        axes[i].plot(fpr, tpr, label="ROC Curve", color="darkorange")
        axes[i].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_title(f"Class: {class_name}")
        axes[i].set_xlabel("False Positive Rate")
        axes[i].set_ylabel("True Positive Rate")

        axes[i].text(
            0.6,
            0.2,
            f"AUC = {roc_auc:.2f}",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.7),
        )

    for j in range(n_classes, 9):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle("One v. Rest ROC Curves", fontsize=16, y=1.02)
    plt.show()


def evaluate_on_test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    """
    Evaluates the model on the test dataset and calculates loss and accuracy.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (torch.utils.data.DataLoader): The DataLoader object for the test dataset.
        criterion (torch.nn.Module): The loss function used for evaluation.
        device (torch.device): The device (CPU or GPU) where the model and data should be loaded.

    Returns:
        Tuple[List[int], List[int]]: A tuple containing two lists:
            - `all_preds`: List of predicted labels for the test set.
            - `all_labels`: List of true labels for the test set.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    return all_preds, all_labels


def measure_inference_time(
    model: torch.nn.Module,
    input_sample: torch.Tensor,
    device: torch.device,
    num_runs: int = 100,
) -> None:
    """
    Measures the average inference time for a single sample.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        input_sample (torch.Tensor): A single input sample to pass through the model.
        device (torch.device): The device (CPU or GPU) where the model and data should be loaded.
        num_runs (int, optional): The number of inference runs to average over. Defaults to 100.

    Returns:
        None: The function prints the average inference time for a single sample.
    """
    model.eval()
    input_sample = input_sample.to(device)

    with torch.no_grad():
        model(input_sample)

    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            model(input_sample)
    total_time = time.time() - start_time

    avg_time = total_time / num_runs
    print(f"Average Inference Time: {avg_time * 1000:.3f} ms")


def measure_batch_inference_time(
    model: torch.nn.Module,
    batch_input: torch.Tensor,
    device: torch.device,
    num_runs: int = 100,
) -> None:
    """
    Measures the average inference time for a batch of inputs.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        batch_input (torch.Tensor): A batch of input samples to pass through the model.
        device (torch.device): The device (CPU or GPU) where the model and data should be loaded.
        num_runs (int, optional): The number of inference runs to average over. Defaults to 100.

    Returns:
        None: The function prints the average inference time for a batch of samples.
    """
    model.eval()
    batch_input = batch_input.to(device)

    with torch.no_grad():
        model(batch_input)

    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            model(batch_input)
    total_time = time.time() - start_time

    avg_time = total_time / num_runs
    print(f"Average Inference Time: {avg_time * 1000:.3f} ms")
