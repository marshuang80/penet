# first line: 350
@memory.cache
def get_model_and_loader(model_path, dataset):

    # Load model args
    with open(Path(model_path).parent / 'args.json') as args_f:
        model_args_dict = json.load(args_f)
    model_args = Struct(**model_args_dict)
    print ("GOT MODEL ARGS")
    # Get loader from args
    _, val_loader, test_loader = load_data(model_args)
    loaders = {'valid': val_loader, 'test': test_loader}
    loader = loaders[dataset]
    print ("GOT LOADER")
    # Load model
    model = model_dict[model_args.model](model_args, loader.dataset.num_classes).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print ("GOT MODEL")
    return model, loader
