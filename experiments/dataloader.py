import torch
import torch.utils.data as utils
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch.nn.functional as F

def init_stratified_dataloader(final_timeseires: torch.tensor,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               stratified: np.array,
                               batch_size) -> List[utils.DataLoader]:
    # labels = F.one_hot(labels.to(torch.int64))
    length = final_timeseires.shape[0]
    train_length = int(length*0.7)
    val_length = int(length*0.1)
    test_length = length-train_length-val_length

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=val_length+test_length, train_size=train_length, random_state=38)
    for train_index, test_valid_index in split.split(final_timeseires, stratified):
        final_timeseires_train, final_pearson_train, labels_train = final_timeseires[
            train_index], final_pearson[train_index], labels[train_index]
        final_timeseires_val_test, final_pearson_val_test, labels_val_test = final_timeseires[
            test_valid_index], final_pearson[test_valid_index], labels[test_valid_index]
        stratified = stratified[test_valid_index]

    split2 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_length)
    for test_index, valid_index in split2.split(final_timeseires_val_test, stratified):
        final_timeseires_test, final_pearson_test, labels_test = final_timeseires_val_test[
            test_index], final_pearson_val_test[test_index], labels_val_test[test_index]
        final_timeseires_val, final_pearson_val, labels_val = final_timeseires_val_test[
            valid_index], final_pearson_val_test[valid_index], labels_val_test[valid_index]

    train_dataset = utils.TensorDataset(
        final_timeseires_train,
        final_pearson_train,
        labels_train
    )

    val_dataset = utils.TensorDataset(
        final_timeseires_val, final_pearson_val, labels_val
    )

    test_dataset = utils.TensorDataset(
        final_timeseires_test, final_pearson_test, labels_test
    )

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]
