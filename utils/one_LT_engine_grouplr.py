import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from .util import *
from .util import AveragePrecisionMeter
from torch.cuda.amp import GradScaler, autocast
import torchnet as tnt
from tqdm import tqdm
import time

import torch.nn.functional as F

from src.helper_functions.metrics import *


class MultiLabelEngine():
    def __init__(self, args):
        # hyper-parameters
        self.evaluation = args['evaluation']
        self.threshold = args['threshold']
        self.resume = args['resume']
        self.max_epoch = args['max_epoch']
        self.start_lr = args['lr']
        self.dataset = args['dataset']

        # measure mAP
        self.regular_ap_meter = AveragePrecisionMeter(threshold=self.threshold, difficult_examples=False)
        self.ema_ap_meter = AveragePrecisionMeter(threshold=self.threshold, difficult_examples=False)
        self.regular_loss_meter = tnt.meter.AverageValueMeter()
        self.ema_loss_meter = tnt.meter.AverageValueMeter()

        # best model
        self.highest_regular_map = 0
        self.highest_ema_map = 0
        self.highest_regular_model = None
        self.highest_ema_model = None
        self.use_logit = args['use_logit']


    def meter_reset(self):
        self.regular_ap_meter.reset()
        self.ema_ap_meter.reset()
        self.regular_loss_meter.reset()
        self.ema_loss_meter.reset()


    def meter_print(self, model_name):
        regular_loss = self.regular_loss_meter.value()[0]
        regular_ap = 100 * self.regular_ap_meter.value()
        regular_map = regular_ap.mean()
        OP, OR, OF1, CP, CR, CF1 = self.regular_ap_meter.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.regular_ap_meter.overall_topk(3)
        print('=================================================>>>>>>> Experimental Results on regular {}'.format(model_name))
        print('mAP score: {map:.3f}\t loss: {loss:.3f}'.format(map=regular_map, loss=regular_loss))
        print('OP: {OP:.4f}\t'
              'OR: {OR:.4f}\t'
              'OF1: {OF1:.4f}\t'
              'CP: {CP:.4f}\t'
              'CR: {CR:.4f}\t'
              'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR,
                                      CF1=CF1))
        print('OP_3: {OP:.4f}\t'
              'OR_3: {OR:.4f}\t'
              'OF1_3: {OF1:.4f}\t'
              'CP_3: {CP:.4f}\t'
              'CR_3: {CR:.4f}\t'
              'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k,
                                        CR=CR_k, CF1=CF1_k))

        ema_loss_meter = self.ema_loss_meter.value()[0]
        ema_ap = 100 * self.ema_ap_meter.value()
        ema_map = ema_ap.mean()
        OP, OR, OF1, CP, CR, CF1 = self.ema_ap_meter.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.ema_ap_meter.overall_topk(3)
        print('#################################################>>>>>>> Experimental Results on EMA {}'.format(model_name))
        print('mAP score: {map:.3f}\t loss: {loss:.3f}'.format(map=ema_map, loss=ema_loss_meter))
        print('OP: {OP:.4f}\t'
              'OR: {OR:.4f}\t'
              'OF1: {OF1:.4f}\t'
              'CP: {CP:.4f}\t'
              'CR: {CR:.4f}\t'
              'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR,
                                      CF1=CF1))
        print('OP_3: {OP:.4f}\t'
              'OR_3: {OR:.4f}\t'
              'OF1_3: {OF1:.4f}\t'
              'CP_3: {CP:.4f}\t'
              'CR_3: {CR:.4f}\t'
              'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k,
                                        CR=CR_k, CF1=CF1_k))
        return regular_ap, regular_map, ema_ap, ema_map



    
        model.eval()
        device = next(model.parameters()).device
        dummy = torch.randn(*input_size).to(device)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy, is_train=False)

            t_sum = 0.0
            for _ in range(repeat):
                t0 = time.perf_counter()
                _ = model(dummy, is_train=False)
                t1 = time.perf_counter()
                t_sum += (t1 - t0)
        avg_ms = (t_sum / repeat) * 1000
        return avg_ms

    def learning(self, models,clip_model, train_loader, val_loader, criterion, optimizer, scheduler, model_name):
        # optionally resume from a checkpoint
        self.start_epoch = 0
        if self.resume != '':
            if os.path.isdir(self.resume):
                if os.path.isfile(os.path.join(self.resume,'regular_checkpoint_cur.ckpt')):
                    print("=> loading checkpoint '{}'".format(self.resume))
                    checkpoint = torch.load(os.path.join(self.resume,'regular_checkpoint_cur.ckpt'))
                    self.start_epoch = checkpoint['epoch']
                    self.highest_regular_map = checkpoint['score']
                    models[0].load_state_dict(checkpoint['state_dict'])
                    
                    checkpoint = torch.load(os.path.join(self.resume,'EMA_checkpoint_cur.ckpt'))
                    self.highest_ema_map = checkpoint['score']
                    models[1].module.load_state_dict(checkpoint['state_dict'])
                    print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.resume, self.start_epoch))

                    print('update scheduler, totoal {} x {} steps'.format(self.start_epoch+1,len(train_loader)))
                    for _ in range((self.start_epoch+1)*len(train_loader)):
                        scheduler.step()
                else:
                    print("=> no checkpoint found at '{}'".format(self.resume))
            else:
                print("=> no checkpoint found at '{}'".format(self.resume))


        if self.evaluation:
            models[0].eval()
            self.meter_reset()
            self.validate(models, clip_model,val_loader, criterion)
            _, _ = self.meter_print(model_name)
        else:
            for epoch in range(self.start_epoch+1, self.max_epoch):
                # if epoch == 30:# for voc
                if epoch == 80:#for coco and nus-wide
                    assert 1 == 2
                # train step
                models[0].train()
                self.meter_reset()
                self.train(models, clip_model,train_loader, criterion, optimizer, scheduler, epoch)
                # _, _ = self.meter_print()
                if epoch >10: #for coco and nus-wide
                # if epoch >20: #for voc 2007
                # evaluate step
                    models[0].eval()
                    self.meter_reset()
                    self.validate(models, clip_model,val_loader, criterion)
                    regular_ap, regular_map, ema_ap, ema_map = self.meter_print(model_name)
                    self.save_checkpoint(models, regular_ap, regular_map, ema_ap, ema_map, epoch, model_name)
                    models[0].train()
    def validate(self, models, clip_model,val_loader, criterion):
        regular_model = models[0]
        ema_model = models[1]
        val_loader = tqdm(val_loader, desc='Test')
        for i, (inputData, target, semantic) in enumerate(val_loader):
            # compute output
            target = target.cuda()
            semantic = semantic[0].cuda().float()
            target = target.squeeze()
            # regular model
            with torch.no_grad():
                with autocast():
                    # regular model
                    regular_text_features,regular_template_features,regular_cls_output, _,regular_dist_feat= regular_model(inputData.cuda())
                    regular_cls_output = regular_cls_output.float()

                    regular_overall_loss = criterion[0](regular_cls_output, target)

                    regular_loss = regular_overall_loss

            # ema model
            with torch.no_grad():
                ema_text_features,ema_template_features,ema_cls_output, _,ema_dist_feat= ema_model.module(inputData.cuda())
                ema_cls_output = ema_cls_output.float()
                ema_overall_loss = criterion[0](ema_cls_output, target)

                ema_loss = ema_overall_loss

            regular_dist_loss1=0.0
            ema_dist_loss1=0.0

            val_loader.set_postfix(cls=regular_overall_loss.item(), cos = regular_dist_loss1, e_cls=ema_overall_loss.item(), e_cos = ema_dist_loss1)


            self.regular_ap_meter.add(regular_cls_output.data, target)
            self.ema_ap_meter.add(ema_cls_output.data, target)
            self.regular_loss_meter.add(regular_loss.item())
            self.ema_loss_meter.add(ema_loss.item())

    # def validate(self, models, clip_model, val_loader, criterion):
    #     regular_model = models[0]
    #     ema_model = models[1]
    #     val_loader = tqdm(val_loader, desc='Test')
    #     regular_model.eval()
    #     ema_model.eval()

    #     for i, (inputData, target, semantic) in enumerate(val_loader):
    #         target = target.cuda(non_blocking=True)
    #         semantic = semantic[0].cuda().float()
    #         target = target.squeeze()
    #         inputData = inputData.cuda(non_blocking=True)

    #         with torch.no_grad():
    #             with autocast():
    #                 crop_logits = []
    #                 for idx in range(5):
    #                     crop_img = inputData[:, idx, :]
    #                     _, _, logit, _, _ = regular_model(crop_img)
    #                     crop_logits.append(logit)
    #                 regular_cls_output = torch.stack(crop_logits).mean(dim=0)
    #                 regular_text_features, regular_template_features, _, _, regular_dist_feat = regular_model(inputData[:,0,:])
    #                 regular_cls_output = regular_cls_output.float()
    #                 regular_overall_loss = criterion[0](regular_cls_output, target)
    #                 regular_loss = regular_overall_loss

    #                 crop_logits = []
    #                 for idx in range(5):
    #                     crop_img = inputData[:, idx, :]
    #                     _, _, logit, _, _ = ema_model.module(crop_img)
    #                     crop_logits.append(logit)
    #                 ema_cls_output = torch.stack(crop_logits).mean(dim=0)
    #                 ema_text_features, ema_template_features, _, _, ema_dist_feat = ema_model.module(inputData[:,0,:])
    #                 ema_cls_output = ema_cls_output.float()
    #                 ema_overall_loss = criterion[0](ema_cls_output, target)
    #                 ema_loss = ema_overall_loss

    #         regular_dist_loss1 = 0.0
    #         ema_dist_loss1 = 0.0
    #         val_loader.set_postfix(
    #             cls=regular_overall_loss.item(),
    #             cos=regular_dist_loss1,
    #             e_cls=ema_overall_loss.item(),
    #             e_cos=ema_dist_loss1
    #         )

    #         self.regular_ap_meter.add(regular_cls_output.data, target)
    #         self.ema_ap_meter.add(ema_cls_output.data, target)
    #         self.regular_loss_meter.add(regular_loss.item())
    #         self.ema_loss_meter.add(ema_loss.item())

    def train(self, models,clip_model,train_loader, criterion, optimizer, scheduler, epoch):
        regular_model = models[0]
        ema_model = models[1]
        train_loader = tqdm(train_loader, desc='Train Epoch '+str(epoch))
        scaler = torch.cuda.amp.GradScaler() 
        for i, (inputData, target, semantic) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()
            semantic = semantic[0].cuda().float()
            target = target.squeeze()
            with autocast():  # mixed precision
                text_feature,template_feature,cls_output, _,dist_feat = regular_model(inputData)
                cls_output = cls_output.float()
                if self.use_logit:
                    mu_p = regular_model.mu_p
                    sigma_p = regular_model.sigma_p
                    mu_n = regular_model.mu_n
                    sigma_n = regular_model.sigma_n
                    tau = 0.001  
                    class_freq = getattr(self, 'class_freq', None)
                    if class_freq is None:
                        class_freq = torch.ones(models[0].num_classes) / models[0].num_classes

                    class_freq = class_freq.cuda()
                    prior_bias = - tau * torch.log(class_freq + 1e-6) 
                    logit_pos = cls_output * sigma_p + mu_p + prior_bias
                    logit_neg = cls_output * sigma_n + mu_n + prior_bias

                    corrected_cls_output = torch.where(target == 1, logit_pos, logit_neg)

                    cls_loss = criterion[0](corrected_cls_output, target)
                else:
                    cls_loss = criterion[0](cls_output, target)

            regular_loss = cls_loss
                
            regular_model.zero_grad()

            optimizer.param_groups[3]["lr"] = optimizer.param_groups[3]["lr"]*0.1
            optimizer.param_groups[2]["lr"] = optimizer.param_groups[2]["lr"]*0.1

            # regular_loss.requires_grad_(True)

            scaler.scale(regular_loss).backward()

            # loss.backward()
            scaler.step(optimizer)
            temp_lr = optimizer.param_groups[3]["lr"]
            scaler.update()
            # optimizer.step()
            scheduler.step()

            ema_model.update(regular_model)
            with torch.no_grad():
                # ema_cls_output, ema_semantic_output = ema_model.module(inputData, semantic)
                ema_text_feature, ema_template_feature,ema_cls_output,_,ema_dist_feat= ema_model.module(inputData.float())
                ema_cls_output = ema_cls_output.float()

            ema_cls_loss = criterion[0](ema_cls_output, target)

            ema_loss = ema_cls_loss
            
            # store information
            dist_loss1=0.0
            ema_dist_loss1=0.0
            # train_loader.set_postfix(cls=cls_loss.item(), cos = dist_loss1.item(), e_cls=ema_cls_loss.item(), e_cos = ema_dist_loss1.item(), lr=temp_lr)
            train_loader.set_postfix(cls=cls_loss.item(), cos = dist_loss1, e_cls=ema_cls_loss.item(), e_cos = ema_dist_loss1, lr=temp_lr)

            self.regular_ap_meter.add(cls_output.data, target)
            self.ema_ap_meter.add(ema_cls_output.data, target)
            self.regular_loss_meter.add(regular_loss.item())
            self.ema_loss_meter.add(ema_loss.item())



    def save_checkpoint(self, models, regular_ap, regular_map, ema_ap, ema_map, epoch, model_name):

        regular_model = models[0]
        ema_model = models[1]

        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)  
        if regular_map > self.highest_regular_map:
            self.highest_regular_map = regular_map
            
            if self.highest_regular_model is not None:
                try:
                    os.remove(os.path.join(save_dir, self.highest_regular_model))
                except:
                    pass

            self.highest_regular_model = f'regular_{model_name}_best_{regular_map:.3f}_e{epoch}.ckpt'
            
            torch.save({
                'epoch': epoch,
                'score': regular_map,
                'model': 'regular',
                'state_dict': regular_model.state_dict(),
            }, os.path.join(save_dir, self.highest_regular_model))

            if self.dataset in ['coco-lt', 'voc-lt']:
                ltAnalysis(regular_ap, self.dataset)

        if ema_map > self.highest_ema_map:
            self.highest_ema_map = ema_map
            
            if self.highest_ema_model is not None:
                try:
                    os.remove(os.path.join(save_dir, self.highest_ema_model))
                except:
                    pass

            self.highest_ema_model = f'ema_{model_name}_best_{ema_map:.3f}_e{epoch}.ckpt'
            
            state_dict = ema_model.module.state_dict() if hasattr(ema_model, 'module') else ema_model.state_dict()
            torch.save({
                'epoch': epoch,
                'score': ema_map,
                'model': 'ema',
                'state_dict': state_dict,
            }, os.path.join(save_dir, self.highest_ema_model))

            if self.dataset in ['coco-lt', 'voc-lt']:
                ltAnalysis(ema_ap, self.dataset)

        print('------------------------------------------------->>>>>>> Highest Experimental Results on {}'.format(model_name))
        print('Highest regular model mAP: {}\t Highest ema model mAP: {}'.format(self.highest_regular_map, self.highest_ema_map))