import numpy as np
import torch


def metrics_global_accuracy(ground_truth, prediction) -> float:
    """
    Global accuracy, referred to as global, is the percentage of the correctly classified pixels
    Args:
        ground_truth: numpy.ndarray or torch.Tensor in which every pixel is the index of a class
        prediction: numpy.ndarray or torch.Tensor in which every pixel is the index of a class

    Returns:
        the percentage of the correctly classified pixels
    """

    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()

    prediction = np.expand_dims(a=np.argmax(a=prediction, axis=1), axis=1)

    N = prediction.size
    TP = np.count_nonzero(prediction == ground_truth)   # TRUE POSITIVE

    global_accuracy = (1 / N) * TP

    return global_accuracy * 100


def metrics_mean_accuracy(ground_truth, prediction, num_classes) -> float:
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()

    prediction = np.expand_dims(a=np.argmax(a=prediction, axis=1), axis=1)

    sum = 0
    for c in range(num_classes):
        TP = np.count_nonzero(np.logical_and(prediction == ground_truth, prediction == c))
        FP = np.count_nonzero(np.logical_and(prediction != ground_truth, prediction == c))

        sum += TP / (TP + FP + 1e-5)

    return (sum / num_classes) * 100


def metrics_IoU(ground_truth, prediction, num_classes):
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()

    prediction = np.expand_dims(a=np.argmax(a=prediction, axis=1), axis=1)

    sum = 0
    for c in range(num_classes):
        TP = np.count_nonzero(np.logical_and(prediction == ground_truth, prediction == c))
        FP = np.count_nonzero(np.logical_and(prediction != ground_truth, prediction == c))
        FN = np.count_nonzero(np.logical_and(prediction == ground_truth, prediction != c))

        sum += TP / (TP + FP + FN + 1e-5)

    return (sum / num_classes) * 100


if __name__ == '__main__':
    pred = torch.randn(size=(5, 5, 320, 320))
    gt = torch.randint(size=(pred.shape[0], 1, pred.shape[-2], pred.shape[-1]), high=pred.shape[1])

    global_acc = metrics_global_accuracy(ground_truth=gt, prediction=pred)
    mean_acc = metrics_mean_accuracy(ground_truth=gt, prediction=pred, num_classes=pred.shape[1])
    iou = metrics_IoU(ground_truth=gt, prediction=pred, num_classes=pred.shape[1])

    print(global_acc)
    print(mean_acc)
    print(iou)
