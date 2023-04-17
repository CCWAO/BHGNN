import argparse
import os
from distutils.util import strtobool

parser = argparse.ArgumentParser()

# Environment
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument("--num_gpu", type=int, default=1)

# Data
parser.add_argument("--save_dir", type=str, default="./uncertainty/")
parser.add_argument("--data_name", type=str, default="TFF",
                   choices=('TFF', 'swat')) # PTC_MR #NCI1, PROTEINS, COLLAB, and RDT-B.
parser.add_argument('--batch_size', type=int, default=128)

# Model
parser.add_argument('--uncertainty', default='BHGNN', choices=('BHGNN'))
parser.add_argument('--drop_rate', type=float, default=0.2)
parser.add_argument('--nhid', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--layer_name', default='GCN', choices=('GCN', 'aram', 'che'))
parser.add_argument('--n_samples', type=int, default=10)
parser.add_argument('--num_features_inner', type=int, default=50)
parser.add_argument('--num_classes', type=int, default=7)

# Train
parser.add_argument("--epochs", type=int, default=350)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-5)


def get_config():
    config = parser.parse_args()
    return config
