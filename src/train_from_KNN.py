def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # TODO if --output_dir exists, load model from there, otherwise load from --pretrained_model_path
    # load model from --pretrained_model_path
    inseg_model_class, inseg_global_model = get_model(args.pretrained_model_path, 'cpu') #, device)

    optimizer = optim.SGD(
        inseg_global_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    train_dataset = CustomDataLoader(args.dataset_path, points_per_object=5, verbose=False)
    # train_dataloader = CustomDataLoader(args.dataset_path, points_per_object=5, verbose=False)
    val_dataloader = CustomDataLoader(args.dataset_path, points_per_object=5)  # TODO change dataset_path to validation set

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=ME.utils.batch_sparse_collate,)
        # num_workers=1)

    train_step = 0  # TODO get number of already trained steps if loading trained checkpoint
    val_ious, train_losses = [], []
    voxel_size = 0.05  # TODO get voxel size from args
    test_step_time = time.time()
    start_time = time.time()

    for epoch in range(args.max_epochs):
        train_dataset.new_epoch()  # TODO test if this works on a smaller dataset
        epoch_time = time.time()
        train_iter = iter(train_dataloader)
        inseg_global_model.train()

        for train_batch in train_iter:

            if train_step % args.test_step == 0:
                print(f'\nEpoch: {epoch} train_step: {train_step}, mean loss: {sum(train_losses[-args.test_step:]) / args.test_step:.2f}, '
                      f'time of test_step: {utils.timeit(test_step_time)}, '
                      f'time from start: {utils.timeit(start_time)}')
                # print time from start in hours, minutes, seconds
                val_iou = test_step(inseg_model_class, inseg_global_model, val_dataloader)
                val_ious.append(val_iou)
                plot_stats(train_losses, val_ious, train_step)
                test_step_time = time.time()

            if train_step % args.save_step == 0:
                save_step(inseg_global_model, args.output_dir, train_step)

            coords, feats, labels = train_batch
            labels = labels_to_logit_shape(labels)
            # print(f'Batch: {coords.shape=}, {feats.shape=}, {labels.shape=}')
            sinput = ME.SparseTensor(feats.float(), coords)
            # print(f'sinput.F.shape: {sinput.F.shape}')

            out = inseg_global_model(sinput)
            # print(f'outputs: {out.F.shape=}\n')
            out = out.slice(sinput)
            # print(f'outputs: {out.F.shape=}\n')
            optimizer.zero_grad()
            loss = criterion(out.F.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_step+=1

            train_losses.append(loss.item())
            train_step+=1
            print('.', end='', flush=True)

        print(f'\n\nEpoch {epoch} took {time.time() - epoch_time:.2f} seconds\n')


def get_model(pretrained_weights_file, device):
    inseg_global = InteractiveSegmentationModel(pretraining_weights=pretrained_weights_file)
    global_model = inseg_global.create_model(device, inseg_global.pretraining_weights_file)
    return inseg_global, global_model

def labels_to_logit_shape(labels: torch.Tensor):
    if len(labels.shape) == 3:
        labels = labels[0]

    labels_new = torch.zeros((len(labels), 2))
    labels_new[labels[:, 0] == 0, 0] = 1
    labels_new[labels[:, 0] == 1, 1] = 1
    return labels_new

def create_input(feats, coords, voxel_size: int = 0.05):
    if len(feats.shape) == 3:
        feats = feats[0]
    if len(coords.shape) == 3:
        coords = coords[0]

    sinput = ME.SparseTensor(
        features=feats,
        coordinates=ME.utils.batched_coordinates([coords / voxel_size]),
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        # device=device
    )  # .to(device)
    return sinput

def test_step(model_class, model, val_dataloader):
    model.eval()

    # TODO do forward pass (model_class.prediction) for every data from eval set, get mean_iou and return it
    ious = []
    # while True:
    #     batch = val_dataloader.get_batch(args.batch_size)
    #     if not train_batch: break

    #     coords, feats, labels = batch
    #     coords = coords[0]
    #     ...

    #     pred = model_class.prediction(feats, coords, model)
    #     iou = model_class.mean_iou(pred, labels)
    #     ious.append(iou)

    # TODO export few examples of rendered result images using utils.save_point_cloud_views

    return sum(ious) / len(ious) if len(ious) > 0 else 0

def save_step(model, path, train_step):
    export_path = os.path.join(path, f'model_{train_step}.pth')
    torch.save(model.state_dict(), export_path)
    print(f'Model saved to: {export_path}')

def plot_stats(train_losses, val_ious, train_step):
    train_losses_str = ', '.join([f'{loss:.5f}' for loss in train_losses])
    print(f'\nTest step. Train losses: [{train_losses_str}]') # , Val IoUs: {val_ious}')

    # TODO produce chart of train_losses and val_ious
    # TODO also save train_losses and val_ious to .npy or something for future reference

if __name__ == '__main__':
    main(parse_args())
