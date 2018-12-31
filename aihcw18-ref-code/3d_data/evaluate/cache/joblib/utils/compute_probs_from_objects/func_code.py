# first line: 371
@memory.cache
def compute_probs_from_objects(model, loader):
    # NOTE: will have to change sigmoid to softmax for multiclass setting.

    probs = []
    for batch in loader:
        batch_inputs, _ = transform_data(batch, use_gpu=True)

        batch_logits = model(batch_inputs)
        batch_probs = torch.sigmoid(batch_logits)
        batch_probs_npy = batch_probs.cpu().data.numpy()

        probs.append(batch_probs_npy)

    probs_concat = np.concatenate(probs, axis=0)

    return probs_concat
