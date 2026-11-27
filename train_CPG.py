import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import time
import torch.autograd
from skimage import io
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

working_path = os.path.dirname(os.path.abspath(__file__))
from utils.utils import accuracy, SCDD_eval_all, AverageMeter
from utils.loss import *
from datasets import RS_ST as RS
from lib.cpgv4 import CPGNet as Net

NET_NAME = 'CPGv4'
DATA_NAME = 'CPGNet_SECOND'
args = {
    'train_batch_size': int(os.environ.get('CPG_TRAIN_BATCH_SIZE', 1)),
    'val_batch_size': int(os.environ.get('CPG_VAL_BATCH_SIZE', 1)),
    'lr': 0.0001,
    'epochs': int(os.environ.get('CPG_EPOCHS', 2)),
    'gpu': True,
    'amp': os.environ.get('CPG_AMP', '1') != '0',
    'num_workers': int(os.environ.get('CPG_NUM_WORKERS', 0)),
    'lr_decay_power': 1.5,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'print_freq': int(os.environ.get('CPG_PRINT_FREQ', 1)),
    'predict_step': 1,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME),
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'pretrained.pth'),
    'pretrained_path': os.environ.get(
        'CPG_PRETRAIN_PATH',
        os.path.join(working_path, 'pretrain_weight', 'pvt_v2_b2.pth')
    )
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train CPGNet for semantic change detection.')
    parser.add_argument('--data-root', default=os.environ.get('CPG_DATA_ROOT', RS.root),
                        help='SECOND dataset root. It must contain train/test folders.')
    parser.add_argument('--epochs', type=int, default=args['epochs'])
    parser.add_argument('--train-batch-size', type=int, default=args['train_batch_size'])
    parser.add_argument('--val-batch-size', type=int, default=args['val_batch_size'])
    parser.add_argument('--num-workers', type=int, default=args['num_workers'])
    parser.add_argument('--lr', type=float, default=args['lr'])
    parser.add_argument('--pretrained-path', default=args['pretrained_path'],
                        help='Path to pvt_v2_b2.pth. Missing file falls back to random init.')
    parser.add_argument('--print-freq', type=int, default=args['print_freq'])
    parser.add_argument('--predict-step', type=int, default=args['predict_step'])
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=args['amp'],
                        help='Use CUDA automatic mixed precision.')
    return parser.parse_args()


cli_args = parse_args()
RS.root = cli_args.data_root
args.update({
    'epochs': cli_args.epochs,
    'train_batch_size': cli_args.train_batch_size,
    'val_batch_size': cli_args.val_batch_size,
    'num_workers': cli_args.num_workers,
    'lr': cli_args.lr,
    'pretrained_path': cli_args.pretrained_path,
    'print_freq': cli_args.print_freq,
    'predict_step': cli_args.predict_step,
    'amp': cli_args.amp,
})

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
writer = SummaryWriter(args['log_dir'])


def main():
    net = Net(64, num_classes=RS.num_classes, pretrained_path=args['pretrained_path']).cuda()

    train_set = RS.Data('train', random_flip=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args['train_batch_size'],
        num_workers=args['num_workers'],
        shuffle=True,
        pin_memory=args['gpu']
    )
    val_set = RS.Data('test')
    val_loader = DataLoader(
        val_set,
        batch_size=args['val_batch_size'],
        num_workers=args['num_workers'],
        shuffle=False,
        pin_memory=args['gpu']
    )

    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    optimizer = optim.AdamW(net.parameters(), lr=args['lr'])

    scaler = torch.cuda.amp.GradScaler(enabled=args['amp'] and args['gpu'])
    train(train_loader, net, criterion, optimizer, val_loader, scaler)
    writer.close()
    print('Training finished.')


def train(train_loader, net, criterion, optimizer, val_loader, scaler):
    bestaccT = 0
    bestFscdV = 0.0
    bestaccV = 0.0
    bestloss = 1.0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    criterion_sc = ChangeSimilarity().cuda()
    curr_epoch = 0
    while True:
        torch.cuda.empty_cache()
        net.train()

        start = time.time()
        acc_meter = AverageMeter()
        train_seg_loss = AverageMeter()
        train_bn_loss = AverageMeter()
        train_sc_loss = AverageMeter()
        train_spacc_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            adjust_lr(optimizer, running_iter, all_iters)
            imgs_A, imgs_B, labels_A, labels_B = data
            if args['gpu']:
                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                labels_bn = (labels_A > 0).unsqueeze(1).cuda().float()
                labels_A = labels_A.cuda().long()
                labels_B = labels_B.cuda().long()

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args['amp'] and args['gpu']):
                out_change, outputs_A, outputs_B, spacc_loss = net(
                    imgs_A,
                    imgs_B,
                    labels_A=labels_A,
                    labels_B=labels_B,
                    return_spacc_loss=True,
                )

                assert outputs_A.size()[1] == RS.num_classes

                loss_seg = criterion(outputs_A, labels_A) * 0.5 + criterion(outputs_B, labels_B) * 0.5
                loss_bn = weighted_BCE_logits(out_change, labels_bn)
                loss_sc = criterion_sc(outputs_A[:, 1:], outputs_B[:, 1:], labels_bn)
                loss_spacc = spacc_loss["loss"]
                loss = loss_seg + loss_bn + loss_sc + loss_spacc
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            labels_A = labels_A.cpu().detach().numpy()
            labels_B = labels_B.cpu().detach().numpy()
            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            change_mask = F.sigmoid(out_change).cpu().detach() > 0.5
            preds_A = torch.argmax(outputs_A, dim=1)
            preds_B = torch.argmax(outputs_B, dim=1)
            preds_A = (preds_A * change_mask.squeeze().long()).numpy()
            preds_B = (preds_B * change_mask.squeeze().long()).numpy()
            acc_curr_meter = AverageMeter()
            for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                acc_A, valid_sum_A = accuracy(pred_A, label_A)
                acc_B, valid_sum_B = accuracy(pred_B, label_B)
                acc = (acc_A + acc_B) * 0.5
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_seg_loss.update(loss_seg.cpu().detach().numpy())
            train_bn_loss.update(loss_bn.cpu().detach().numpy())
            train_sc_loss.update(loss_sc.cpu().detach().numpy())
            train_spacc_loss.update(loss_spacc.cpu().detach().numpy())

            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train seg_loss %.4f bn_loss %.4f spacc_loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_seg_loss.val, train_bn_loss.val, train_spacc_loss.val, acc_meter.val * 100))
                writer.add_scalar('train seg_loss', train_seg_loss.val, running_iter)
                writer.add_scalar('train sc_loss', train_sc_loss.val, running_iter)
                writer.add_scalar('train spacc_loss', train_spacc_loss.val, running_iter)
                writer.add_scalar('train spacc_loss_ce', spacc_loss["loss_ce"].cpu().detach().item(), running_iter)
                writer.add_scalar('train spacc_loss_ppc', spacc_loss["loss_ppc"].cpu().detach().item(), running_iter)
                writer.add_scalar('train spacc_loss_ppd', spacc_loss["loss_ppd"].cpu().detach().item(), running_iter)
                writer.add_scalar('train accuracy', acc_meter.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        Fscd_v, mIoU_v, Sek_v, acc_v, loss_v = validate(val_loader, net, criterion, curr_epoch)
        if acc_meter.avg > bestaccT: bestaccT = acc_meter.avg
        if Fscd_v > bestFscdV:
            bestFscdV = Fscd_v
            bestaccV = acc_v
            bestloss = loss_v
            torch.save(net.state_dict(),
                       os.path.join(args['chkpt_dir'], NET_NAME + '_%de_mIoU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth' \
                                    % (curr_epoch, mIoU_v * 100, Sek_v * 100, Fscd_v * 100, acc_v * 100)))
        print('Total time: %.1fs Best rec: Train acc %.2f, Val Fscd %.2f acc %.2f loss %.4f' % (
        time.time() - begin_time, bestaccT * 100, bestFscdV * 100, bestaccV * 100, bestloss))
        curr_epoch += 1
        if curr_epoch >= args['epochs']:
            return


def validate(val_loader, net, criterion, curr_epoch):
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []
    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels_A, labels_B = data
        if args['gpu']:
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels_A = labels_A.cuda().long()
            labels_B = labels_B.cuda().long()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args['amp'] and args['gpu']):
                out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)
                loss_A = criterion(outputs_A, labels_A)
                loss_B = criterion(outputs_B, labels_B)
                loss = loss_A * 0.5 + loss_B * 0.5
        val_loss.update(loss.cpu().detach().numpy())

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = F.sigmoid(out_change).cpu().detach() > 0.5
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A * change_mask.squeeze().long()).numpy()
        preds_B = (preds_B * change_mask.squeeze().long()).numpy()
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc = (acc_A + acc_B) * 0.5
            acc_meter.update(acc)

        if curr_epoch % args['predict_step'] == 0 and vi == 0:
            pred_A_color = RS.Index2Color(preds_A[0])
            pred_B_color = RS.Index2Color(preds_B[0])
            io.imsave(os.path.join(args['pred_dir'], NET_NAME + '_A.png'), pred_A_color)
            io.imsave(os.path.join(args['pred_dir'], NET_NAME + '_B.png'), pred_B_color)
            print('Prediction saved!')

    Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, RS.num_classes)

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f' \
          % (curr_time, val_loss.average(), Fscd * 100, IoU_mean * 100, Sek * 100, acc_meter.average() * 100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Fscd', Fscd, curr_epoch)
    writer.add_scalar('val_Accuracy', acc_meter.average(), curr_epoch)

    return Fscd, IoU_mean, Sek, acc_meter.avg, val_loss.avg


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def adjust_lr(optimizer, curr_iter, all_iter, init_lr=args['lr']):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    main()
