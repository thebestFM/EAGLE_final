import numpy as np
import random
import pandas as pd

class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)
        self.tbatch = None
        self.n_batch = 0

    def sample(self, ratio):
        data_size = self.n_interactions
        sample_size = int(ratio * data_size)
        sample_inds = random.sample(range(data_size), sample_size)
        sample_inds = np.sort(sample_inds)
        sources = self.sources[sample_inds]
        destination = self.destinations[sample_inds]
        timestamps = self.timestamps[sample_inds]
        edge_idxs = self.edge_idxs[sample_inds]
        labels = self.labels[sample_inds]
        return Data(sources, destination, timestamps, edge_idxs, labels)


def compute_time_statistics(sources, destinations, timestamps):
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []

    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]

        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0

        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp

    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)
    return (
        mean_time_shift_src,
        std_time_shift_src,
        mean_time_shift_dst,
        std_time_shift_dst,
    )


# transductive setting
def get_data_transductive(dataset_name, use_validation=False):
    used_datasets = ["Contacts", "lastfm", "wikipedia", "reddit", "superuser", "askubuntu", "wikitalk"]
    other_datasets = [
        "mooc",
        "enron",
        "SocialEvo",
        "uci",
        "CollegeMsg",
        "TaobaoSmall",
        "CanParl",
        "Flights",
        "UNtrade",
        "USLegis",
        "UNvote",
        "Taobao",
        "DGraphFin",
        "TaobaoLarge",
        "YoutubeReddit",
        "YoutubeRedditLarge",
    ]
    if dataset_name in used_datasets:
        dir = "data"
    elif dataset_name in other_datasets:
        dir = "benchtemp_datasets"

    graph_df = pd.read_csv(f"../{dir}/{dataset_name}/ml_{dataset_name}.csv")

    # edge_features = np.load('../data/{}/ml_{}.npy'.format(dataset_name,dataset_name))
    # node_features = np.load('../data/{}/ml_{}_node.npy'.format(dataset_name,dataset_name))

    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values.astype(np.float64)

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)
    n_edges = len(sources)

    random.seed(2024)

    train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
    test_mask = timestamps > test_time

    val_mask = (
        np.logical_and(timestamps <= test_time, timestamps > val_time)
        if use_validation
        else test_mask
    )

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    train_data = Data(
        sources[train_mask],
        destinations[train_mask],
        timestamps[train_mask],
        edge_idxs[train_mask],
        labels[train_mask],
    )

    val_data = Data(
        sources[val_mask],
        destinations[val_mask],
        timestamps[val_mask],
        edge_idxs[val_mask],
        labels[val_mask],
    )

    test_data = Data(
        sources[test_mask],
        destinations[test_mask],
        timestamps[test_mask],
        edge_idxs[test_mask],
        labels[test_mask],
    )

    print(
        "The dataset has {} interactions, involving {} different nodes".format(
            full_data.n_interactions, full_data.n_unique_nodes
        )
    )
    print(
        "The training dataset has {} interactions, involving {} different nodes".format(
            train_data.n_interactions, train_data.n_unique_nodes
        )
    )
    print(
        "The validation dataset has {} interactions, involving {} different nodes".format(
            val_data.n_interactions, val_data.n_unique_nodes
        )
    )
    print(
        "The test dataset has {} interactions, involving {} different nodes".format(
            test_data.n_interactions, test_data.n_unique_nodes
        )
    )

    return full_data, train_data, val_data, test_data, n_total_unique_nodes, n_edges
