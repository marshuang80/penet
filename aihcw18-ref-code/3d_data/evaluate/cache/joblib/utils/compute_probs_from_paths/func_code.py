# first line: 391
@memory.cache
def compute_probs_from_paths(model_path, dataset):

    model, loader = get_model_and_loader(model_path, dataset)
    print ("BACK HERE")
    return compute_probs_from_objects(model, loader)
