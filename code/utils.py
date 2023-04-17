from torch_geometric.data import Batch
import torch
import torch.nn.functional as F

def feature_similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


def get_train_3sample_batch(data, train_set_id, train_set):
    border = data.border
    train_id = data.time_id
    sample_id1 = train_id + border
    sample_id2 = train_id + 2 * border
    sample_id3 = train_id + 3 * border
    train_set_sample1 = []
    train_set_sample2 = []
    train_set_sample3 = []
    for i in range(len(sample_id1)):
        if sum(train_set_id == sample_id1[i]):
            a = torch.where(train_set_id == sample_id1[i])[0]
        else:
            a = torch.where(train_set_id == train_id[i])[0]
        train_set_sample1.append(train_set[a])

        if sum(train_set_id == sample_id2[i]):
            a = torch.where(train_set_id == sample_id2[i])[0]
        else:
            a = torch.where(train_set_id == train_id[i])[0]
        train_set_sample2.append(train_set[a])

        if sum(train_set_id == sample_id3[i]):
            a = torch.where(train_set_id == sample_id3[i])[0]
        else:
            a = torch.where(train_set_id == train_id[i])[0]
        train_set_sample3.append(train_set[a])

    train_set_sample_batch1 = Batch.from_data_list(train_set_sample1)
    train_set_sample_batch2 = Batch.from_data_list(train_set_sample2)
    train_set_sample_batch3 = Batch.from_data_list(train_set_sample3)

    return train_set_sample_batch1, train_set_sample_batch2, train_set_sample_batch3


def hp_structure(dataset_name):
    if dataset_name == 'TFF':
        hy_edge = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 5, 5],
                                [1, 2, 3, 5, 0, 2, 0, 1, 5, 0, 4, 5, 3, 0, 2, 3]], dtype=torch.long)
        batch_inner = torch.cat((0 * torch.ones((3,), dtype=torch.int64),
                                 1 * torch.ones((3,), dtype=torch.int64),
                                 2 * torch.ones((3,), dtype=torch.int64),
                                 3 * torch.ones((6,), dtype=torch.int64),
                                 4 * torch.ones((3,), dtype=torch.int64),
                                 5 * torch.ones((6,), dtype=torch.int64)), dim=0)
    else:
        hy_edge = torch.tensor([[0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5],
                                [1, 0, 2, 1, 3, 5, 2, 4, 3, 5, 2, 4]], dtype=torch.long)
        batch_inner = torch.cat((0 * torch.ones((4,), dtype=torch.int64),
                                 1 * torch.ones((8,), dtype=torch.int64),
                                 2 * torch.ones((8,), dtype=torch.int64),
                                 3 * torch.ones((7,), dtype=torch.int64),
                                 4 * torch.ones((12,), dtype=torch.int64),
                                 5 * torch.ones((2,), dtype=torch.int64)), dim=0)
    return hy_edge, batch_inner
















