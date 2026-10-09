r""" Logging during training/testing """
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch
from .utils import is_main_process, save_on_master, reduce_metric


class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, dataset):
        self.benchmark = dataset.benchmark
        self.class_ids_interest = dataset.class_ids
        self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()

        if self.benchmark == 'pascal':
            self.nclass = 20
        elif self.benchmark == 'coco':
            self.nclass = 80
        elif self.benchmark == 'fss':
            self.nclass = 1000
        elif self.benchmark == 'btcv':
            self.nclass = 13 # 或者直接写 self.nclass = 2
        elif self.benchmark == 'joint':
            # 联合训练：nclass 是所有子 dataset 的类数之和
            # 优先用 nclass_total（如果 dataset 暴露），否则回退到 max
            if hasattr(dataset, 'nclass_total'):
                self.nclass = dataset.nclass_total
            else:
                self.nclass = max(getattr(dataset, 'nclass_list', [1]))
        else:
            # 万能适配方案：直接读取数据集定义的 nclass
            self.nclass = dataset.nclass

        self.intersection_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.union_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.ones = torch.ones_like(self.union_buf)
        self.loss_buf = []

    def update(self, inter_b, union_b, class_id, loss):
        self.intersection_buf.index_add_(1, class_id, inter_b.float())
        self.union_buf.index_add_(1, class_id, union_b.float())
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

    def compute_iou(self):
        iou = self.intersection_buf.float() / \
              torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100

        fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
                  self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

        return miou, fb_iou

    def compute_dice(self):
        # 前景行（索引 1），按类别算 Dice = 2*inter/(pred+gt)，再平均
        inter = self.intersection_buf.index_select(1, self.class_ids_interest)
        union = self.union_buf.index_select(1, self.class_ids_interest)
        dice_fg = 2.0 * inter[1] / (union[1] + inter[1] + 1e-6)
        mdice = dice_fg.mean() * 100
        return mdice

    def write_result(self, split, epoch):
        self.intersection_buf, self.union_buf = self.reduce_metrics([self.intersection_buf, self.union_buf], False)
        iou, fb_iou = self.compute_iou()
        dice = self.compute_dice()

        # loss_buf = torch.stack(self.loss_buf)
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch if epoch != -1 else ''
        if epoch != -1:
            loss_buf = torch.stack(self.loss_buf)
            loss_buf = self.reduce_metrics([loss_buf])[0]
            msg += 'Avg L: %6.5f  ' % loss_buf.mean()
        msg += 'Dice: %5.2f   ' % dice
        msg += 'mIoU: %5.2f   ' % iou
        msg += 'FB-IoU: %5.2f   ' % fb_iou

        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            dice = self.compute_dice()
            fb_iou = self.compute_iou()[1]
            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf)
                msg += 'L: %6.5f  ' % loss_buf[-1]
                msg += 'Avg L: %6.5f  ' % loss_buf.mean()
            msg += 'Dice: %5.2f  |  ' % dice
            msg += 'FB-IoU: %5.2f' % fb_iou
            Logger.info(msg)
    def reduce_metrics(self, metrics, average=True):
        reduced_metrics = []
        for m in metrics:
            reduce_metric(m, average)
            reduced_metrics.append(m)
        return reduced_metrics


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.logpath if training else '_TEST_' + args.load.split('/')[-2].split('.')[0] + logtime
        if logpath == '': logpath = logtime

        cls.logpath = os.path.join('logs', logpath + '.log')
        cls.benchmark = args.benchmark
        # 防覆盖保护：若历史日志已存在，自动追加时间戳后缀，避免覆盖旧实验记录
        if os.path.exists(os.path.join(cls.logpath, 'log.txt')):
            cls.logpath = cls.logpath + logtime
        os.makedirs(cls.logpath, exist_ok=True)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        beijing_now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)
        logging.info('[Start] 训练开始: %s (北京时间, UTC+8)' % beijing_now.strftime('%Y-%m-%d %H:%M:%S'))
        logging.info('\n:=========== Few-shot Seg. with VRP-SAM ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':==================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, sam_model, epoch, val_miou):
        # 同时保存 VRP_encoder 与微调后的 SAM（mask_decoder 等），否则测试会丢分
        ckpt = {'model': model.state_dict(),
                'sam_model': sam_model.state_dict()}
        torch.save(ckpt, os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. mIoU: %5.2f.\n' % (epoch, val_miou))

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0
        for k in model.state_dict().keys():
            n_param = model.state_dict()[k].view(-1).size(0)
            if [i for i in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4'] if i in k]:
                backbone_param += n_param
            else:
                learner_param += n_param
        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % (backbone_param + learner_param))

