import _pickle as pk

import torch
from pylab import *
from torch.utils.data import Dataset


def cnf_to_matrix(cnf_str):
    """Convert cnf file to bipartite graph matrix (num_clauses,num_clauses)

    Args:
        fpath: The path to the cnf file.
            The first two lines of the file are notes.
            The clauses start from the 3rd line.

    Returns:
        A numpy 2-dimensional array.
        The rows are for clauses.
        The columns are for literals.
        example:

        array([[ 0,  0,  0,  0, -1,  1],
               [ 0,  0,  0,  1,  0, -1],
               [ 0,  0,  0,  0,  1, -1]])

        The example denotes a cnf file with 3 clauses and 6 parameters.
        0 denotes that the literal in this column didn't show up in the
            cluase of this row.
        1 denotes the positive literal exists in this clause.
        -1 denotes the negative literal exists in this clause.

    """
    # lines = open(fpath).readlines()[2:]
    lines = cnf_str.split("\n")

    literals = {}
    clauses = []
    for l in lines:
        for ele in l.strip().split(" ")[:-1]:
            literals[abs(int(ele))] = 0
    num_literals = len(literals)
    num_clauses = len(lines)
    literals_list = list(literals)
    l_to_pos = {}
    pos_init = 0
    for l in literals_list:
        l_to_pos[l] = pos_init
        pos_init += 1

    bi_g = np.zeros((num_clauses, num_literals), dtype=np.int)

    for i in range(num_clauses):
        for ele in lines[i].strip().split(" ")[:-1]:
            literal = int(ele)
            pos = l_to_pos[abs(literal)]
            if literal > 0:
                bi_g[i, pos] = 1
            else:
                bi_g[i, pos] = -1
    return bi_g


def read_ic(c,q,SF):
    """read preprocessed data files"""
    """ q is the start index for vector features """

    root_dir = '{}_{}.pk'

    times = np.array(pk.load(open(root_dir.format(c, 'Y'), 'rb')))
    input = np.array(pk.load(open(root_dir.format(c, 'X'), 'rb')))

    inc_feat = [_[q:] for _ in input]
    feat_list = [_[:q] for _ in input]
    
    feat = [Each_data[SF].squeeze(axis=0) for Each_data in feat_list]
    
#    feat = np.asarray(feat)
#    feat = feat[:,SF]
#    feat = feat.squeeze(axis=1)

    print('Data size: {}'.format(times.size))

    train_rate = 0.8
    val_rate = 0.9
    test_rate = 1

    DATA_NUM = len(times)
    data_ind = np.arange(DATA_NUM)
    np.random.seed(8)
    np.random.shuffle(data_ind)

    train_num = sorted(data_ind[range(int(DATA_NUM * train_rate))])
    val_num = sorted(data_ind[range(int(DATA_NUM * train_rate), int(DATA_NUM * val_rate))])
    test_num = sorted(data_ind[range(int(DATA_NUM * val_rate), int(DATA_NUM * test_rate))])

    train_num = torch.LongTensor(train_num)
    val_num = torch.LongTensor(val_num)
    test_num = torch.LongTensor(test_num)
    times = torch.FloatTensor(np.log1p(times))
    feat = torch.FloatTensor(feat)

    return inc_feat, feat, times, train_num, val_num, test_num


def print_network(net):
    """print brief structure of neural networks"""

    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def plot_metric(x, y, z, yc='train', zc='eval'):
    """plot result image and save in local files"""

    matplotlib.style.use('seaborn')
    plt.figure(figsize=(12, 6))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # plt.scatter(x, y)
    # plot(x, y, markerfacecolor='salmon', markeredgewidth=1, markevery=slice(40, len(y), 70),
    #      linestyle=':', marker='o', color='crimson', linewidth=3, label='fit')  # fit result
    plot(x, y, color='crimson', linewidth=5, label=yc)  # fit result

    # plt.scatter(x, z)
    # plot(x, y, markerfacecolor='salmon', markeredgewidth=1, markevery=slice(40, len(y), 70),
    #      linestyle=':', marker='o', color='crimson', linewidth=3, label='fit')  # fit result
    plot(x, z, color='blue', linewidth=5, label=zc)  # fit result
    plt.legend(loc="best", prop={'size': 20})
    # plt.savefig('loss.png')
    plt.savefig('{}-{}.png'.format(yc, zc))
    plt.close()


class GraphDataset(Dataset):
    """wrap function for sampling training instances"""

    def __init__(self, ids):
        self.ids = ids

    def __getitem__(self, index):
        return self.ids[index]

    def __len__(self):
        return len(self.ids)


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def store_model(model, name='test'):
    torch.save(model, '{}'.format(name))


def restore_model(path):
    model = torch.load(path)
    model.eval()

    return model
