import data_loader
import models
import torch
import torch.nn as nn
import util
from models import resnet50
from args import TrainArgParser
from evaluator import ModelEvaluator
from logger import TrainLogger
from saver import ModelSaver


def train(args):

    # train resnet 50 
    model = resnet50(num_classes=400, shortcut_type='B',
                             sample_size=112, sample_duration=16, 
                             last_fc=True)
    model = model.to(args.device)
    model = nn.DataParallel(model, device_ids=args.gpu_ids)
    model_data = torch.load(args.ckpt_path)
    model.load_state_dict(model_data['state_dict'])
    model.module.fc = nn.Linear(73728, 1, bias=True)

    model = model.to(args.device)
    model.train()

    # Get optimizer and scheduler
    if args.use_pretrained or args.fine_tune:
        parameters = model.module.fine_tuning_parameters(args.fine_tuning_boundary, args.fine_tuning_lr)
    else:
        parameters = model.parameters()
    optimizer = util.get_optimizer(parameters, args)
    lr_scheduler = util.get_scheduler(optimizer, args)

    # Get logger, evaluator, saver
    cls_loss_fn = util.get_loss_fn(is_classification=True, dataset=args.dataset, size_average=False)
    data_loader_fn = data_loader.__dict__[args.data_loader]
    train_loader = data_loader_fn(args, phase='train', is_training=True)
    logger = TrainLogger(args, len(train_loader.dataset), train_loader.dataset.pixel_dict)
    eval_loaders = [data_loader_fn(args, phase='val', is_training=False)]
    evaluator = ModelEvaluator(args.do_classify, args.dataset, eval_loaders, logger,
                               args.agg_method, args.num_visuals, args.max_eval, args.epochs_per_eval)
    saver = ModelSaver(args.save_dir, args.epochs_per_save, args.max_ckpts, args.best_ckpt_metric, args.maximize_metric)

    # Train model
    while not logger.is_finished_training():
        logger.start_epoch()
       
        for inputs, target_dict in train_loader:
            logger.start_iter()
            with torch.set_grad_enabled(True):

                b, c, s, w, h  = inputs.shape
                inputs = inputs.expand(b, 3, s, w, h)
                
                inputs.to(args.device)
                cls_logits = model.forward(inputs)
                cls_targets = target_dict['is_abnormal']

                cls_loss = cls_loss_fn(cls_logits, cls_targets.to(args.device))
                loss = cls_loss.mean()

                logger.log_iter(inputs, cls_logits, target_dict, cls_loss.mean(), optimizer)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.end_iter()
            util.step_scheduler(lr_scheduler, global_step=logger.global_step)

        metrics, curves = evaluator.evaluate(model, args.device, logger.epoch)
        saver.save(logger.epoch, model, optimizer, lr_scheduler, args.device,
                   metric_val=metrics.get(args.best_ckpt_metric, None))
        logger.end_epoch(metrics, curves)
        util.step_scheduler(lr_scheduler, metrics, epoch=logger.epoch, best_ckpt_metric=args.best_ckpt_metric)


if __name__ == '__main__':
    util.set_spawn_enabled()
    parser = TrainArgParser()
    args_ = parser.parse_args()
    train(args_)
