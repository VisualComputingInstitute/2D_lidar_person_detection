from torch.utils.data import DataLoader


def get_dataloader(split, batch_size, num_workers, shuffle, dataset_cfg):
    if "DROW" in dataset_cfg["DataHandle"]["data_dir"]:
        from .drow_dataset import DROWDataset

        ds = DROWDataset(split, dataset_cfg)
    elif "JRDB" in dataset_cfg["DataHandle"]["data_dir"]:
        if dataset_cfg["DataHandle"]["tracking"]:
            from .jrdb_detr_dataset import JRDBDeTrDataset

            assert dataset_cfg["DataHandle"]["num_scans"] == 1
            ds = JRDBDeTrDataset(split, dataset_cfg)
        else:
            from .jrdb_dataset import JRDBDataset

            ds = JRDBDataset(split, dataset_cfg)
    else:
        raise RuntimeError(f"Unknown dataset {dataset_cfg['name']}.")

    return DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=ds.collate_batch,
    )
