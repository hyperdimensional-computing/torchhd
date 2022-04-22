import torch


class Accuracy:
    """Useful for calculating the accuracy of a classification task"""

    def __init__(self):
        self.reset()

    def step(self, true_label: torch.LongTensor, predicted_label: torch.LongTensor):
        self.true_labels.append(true_label)
        self.predicted_labels.append(predicted_label)

    def value(self) -> torch.FloatTensor:
        true_labels = torch.cat(self.true_labels)
        pred_labels = torch.cat(self.predicted_labels)

        is_correct = true_labels == pred_labels

        return is_correct.sum() / true_labels.size(0)

    def reset(self):
        self.true_labels = []
        self.predicted_labels = []
