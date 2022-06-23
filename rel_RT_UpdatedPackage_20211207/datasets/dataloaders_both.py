from datasets.dtu_both import DTUDataset


def train_dataloader(self):
    train_dataset = DTUDataset(root_dir=self.hparams.root_dir,
                               split='train',
                               n_views=self.hparams.n_views,
                               n_depths=self.hparams.n_depths,
                               interval_scale=self.hparams.interval_scale,
                               img_size=self.hparams.img_size,
                               img_crop=self.hparams.img_crop,
                               intrinsic_file_path=self.hparams.intrinsic_file_path
                               )
    if self.hparams.num_gpus > 1:
        sampler = DistributedSampler(train_dataset)
    else:
        sampler = None
    return DataLoader(train_dataset,
                      shuffle=False,  # (sampler is None),
                      sampler=sampler,
                      num_workers=4,
                      batch_size=self.hparams.batch_size,
                      pin_memory=True)


def val_dataloader(self):
    val_dataset = DTUDataset(root_dir=self.hparams.root_dir,
                             split='val',
                             n_views=self.hparams.n_views,
                             n_depths=self.hparams.n_depths,
                             interval_scale=self.hparams.interval_scale,
                             img_size=self.hparams.img_size,
                             img_crop=self.hparams.img_crop,
                             intrinsic_file_path=self.hparams.intrinsic_file_path
                             )
    if self.hparams.num_gpus > 1:
        sampler = DistributedSampler(val_dataset)
    else:
        sampler = None
    return DataLoader(val_dataset,
                      shuffle=False,  # (sampler is None),
                      sampler=sampler,
                      num_workers=4,
                      batch_size=self.hparams.batch_size,
                      pin_memory=True)

def test_dataloader(self):
    test_dataset = DTUDataset(root_dir=self.hparams.root_dir,
                             split='test',
                             n_views=self.hparams.n_views,
                             n_depths=self.hparams.n_depths,
                             interval_scale=self.hparams.interval_scale,
                             img_size=self.hparams.img_size,
                             img_crop=self.hparams.img_crop,
                             intrinsic_file_path=self.hparams.intrinsic_file_path
                             )
    if self.hparams.num_gpus > 1:
        sampler = DistributedSampler(val_dataset)
    else:
        sampler = None
    return DataLoader(val_dataset,
                      shuffle=False,  # (sampler is None),
                      sampler=sampler,
                      num_workers=4,
                      batch_size=self.hparams.batch_size,
                      pin_memory=True)