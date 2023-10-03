import torch
import torch.nn.functional as F

model_to_classifier_type = {
    "dsmil": "dsmil",
    "dsmil_bin": "dsmil_bin",
    "dsmil_ce": "dsmil_ce",
    "clam_sb": "clam",  "clam_mb": "clam",
    "hipt": "e2e", "avgpooling": "e2e", "maxpooling": "e2e", "abmil": "e2e", "gabmil": "e2e",
    "transmil": "e2e",
    "hipt_dsmil": "dsmil", "hipt_hipt": "e2e",
}

def get_classifer_fuc(classifier_type):
    if classifier_type == 'dsmil':
        return dsmil_forward
    if classifier_type == 'dsmil_bin':
        return dsmil_bin_forward
    if classifier_type == 'dsmil_ce':
        return dsmil_forward_ce
    elif classifier_type == 'e2e':
        return e2e_forward
    elif classifier_type == 'clam':
        return clam_forward
    else:
        raise NotImplementedError


def dsmil_forward_ce(data, classifier, loss, num_classes, label=None):
    # if self.num_classes >= 2:
    # with torch.no_grad():
    #     label = F.one_hot(label, num_classes=num_classes).float()

    ins_prediction, bag_prediction, _, _ = classifier(data)
    max_prediction, _ = torch.max(ins_prediction, 0)
    bag_loss = loss(bag_prediction.view(1, -1), label)
    max_loss = loss(max_prediction.view(1, -1), label)
    loss = 0.5 * bag_loss + 0.5 * max_loss

    Y_prob = torch.sigmoid(bag_prediction)
    # Y_prob = F.softmax(bag_prediction, dim=1)
    return bag_prediction, loss, Y_prob


def dsmil_forward(data, classifier, loss, num_classes, label=None):
    # if self.num_classes >= 2:
    with torch.no_grad():
        label = F.one_hot(label, num_classes=num_classes).float()

    ins_prediction, bag_prediction, _, _ = classifier(data)
    max_prediction, _ = torch.max(ins_prediction, 0)
    bag_loss = loss(bag_prediction.view(1, -1), label.view(1, -1))
    max_loss = loss(max_prediction.view(1, -1), label.view(1, -1))
    loss = 0.5 * bag_loss + 0.5 * max_loss

    Y_prob = torch.sigmoid(bag_prediction)
    # Y_prob = F.softmax(bag_prediction, dim=1)
    return bag_prediction, loss, Y_prob

def dsmil_bin_forward(data, classifier, loss, num_classes, label=None):
    # if self.num_classes >= 2:
    with torch.no_grad():
        # label = F.one_hot(label, num_classes=num_classes).float()
        label_bin = torch.zeros(num_classes, device=label.device)
        label_bin[:label[0]] = 1.

    ins_prediction, bag_prediction, _, _ = classifier(data)
    max_prediction, _ = torch.max(ins_prediction, 0)
    bag_loss = loss(bag_prediction.view(1, -1), label_bin.view(1, -1))
    max_loss = loss(max_prediction.view(1, -1), label_bin.view(1, -1))
    loss = 0.5 * bag_loss + 0.5 * max_loss

    Y_prob = bag_prediction.sum(1).round()
    # Y_prob = torch.sigmoid(bag_prediction)
    # Y_prob = F.softmax(bag_prediction, dim=1)
    return bag_prediction, loss, Y_prob

def clam_forward(data, classifier, loss, num_classes, label=None):
    logits, Y_prob, Y_hat, _, instance_dict = classifier(data, label=label, instance_eval=True)
    loss = loss(logits, label)
    instance_loss = instance_dict['instance_loss']
    total_loss = classifier.bag_weight * loss + (1 - classifier.bag_weight) * instance_loss
    return logits, total_loss, Y_prob

def e2e_forward(data, classifier, loss, num_classes, label=None):
    pred = classifier(data)
    if label is None:
        return pred
    else:
        loss = loss(pred, label)
        pred_prob = F.softmax(pred, dim=1)
        return pred, loss, pred_prob