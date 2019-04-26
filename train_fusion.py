import data_loader
import models
import torch
import torch.nn as nn
import util

from args import TrainArgParser
from evaluator import ModelEvaluator
from logger import TrainLogger
from saver import ModelSaver


def train(args):

    # count number of dimentions for extra metadat features
    features_dimentions = {"age":1,
                         "is_smoker":1,
                         "race":7,
                         "sex":1}


    features_to_use = args.features.split(",")
    num_meta = 0
    for feature in features_to_use:
        try:
            num_dim = features_dimentions[feature]
        except:
            print("feature listed not permitted")
        num_meta += num_dim

    model = models.FusionNet(args, num_meta)
    #model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    model.train()

    parameters = model.parameters()
    optimizer = util.optim_util.get_optimizer(parameters, args)

    # Get logger, evaluator, saver
    cls_loss_fn = util.optim_util.get_loss_fn(is_classification=True, dataset=args.dataset, size_average=False)
    data_loader_fn = data_loader.__dict__[args.data_loader]
    train_loader = data_loader_fn(args, phase='train', is_training=True)

    logger = TrainLogger(args, len(train_loader.dataset), train_loader.dataset.pixel_dict)
    eval_loaders = [data_loader_fn(args, phase='val', is_training=False)]
    evaluator = ModelEvaluator(args.do_classify, args.dataset, eval_loaders, logger,
                               args.agg_method, args.num_visuals, args.max_eval, args.epochs_per_eval)
    saver = ModelSaver(args.save_dir, args.epochs_per_save, args.max_ckpts, args.best_ckpt_metric, args.maximize_metric)

    # Train model
    for _ in range(args.num_epochs):
        logger.start_epoch()

        for inputs, target_dict, meta in train_loader:
            logger.start_iter()

            #meta = []

            #for f in features_to_use:
            #    meta += meta_dict[f]
            #meta = torch.FloatTensor(meta)
            #print(meta)
            #meta_tensor = torch.FloatTensor(meta)

            with torch.set_grad_enabled(True):
                #inputs = inputs.to(args.device)
                meta = meta.to(args.device)
                cls_logits = model.forward(inputs, meta)
                cls_targets = target_dict['is_abnormal']
                cls_loss = cls_loss_fn(cls_logits, cls_targets.to(args.device))
                loss = cls_loss.mean()

                logger.log_iter(inputs, cls_logits, target_dict, cls_loss.mean(), optimizer)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.end_iter()
            #util.optim_util.step_scheduler(lr_scheduler, global_step=logger.global_step)

        metrics, curves = evaluator.evaluate(model, args.device, logger.epoch)
        saver.save(logger.epoch, model, optimizer, lr_scheduler=None, device=args.device,metric_val=metrics.get(args.best_ckpt_metric, None))
        logger.end_epoch(metrics, curves)
        #util.optim_util.step_scheduler(lr_scheduler, metrics, epoch=logger.epoch, best_ckpt_metric=args.best_ckpt_metric)


if __name__ == '__main__':
    util.set_spawn_enabled()
    parser = TrainArgParser()

    args_ = parser.parse_args()
    train(args_)
