# =================================================================================================
# TRAINING LOGIC "WITHOUT" MIXED PRECISION, CHARACTER ERROR RATE (CER), LEARNING RATE AND NEW FORMATTING 
# BATCH CER PLUS AVERAGE CER
# =================================================================================================

import os
import sys
import time
import random
import string
import argparse
import csv

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation
from difflib import SequenceMatcher

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate (CER) between two strings.
    CER = (Substitutions + Deletions + Insertions) / Reference Length
    """
    matcher = SequenceMatcher(None, reference, hypothesis)
    substitutions, deletions, insertions = 0, 0, 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            substitutions += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            deletions += i2 - i1
        elif tag == 'insert':
            insertions += j2 - j1
    cer = (substitutions + deletions + insertions) / max(1, len(reference))
    return cer

def train(opt):
    """
    Run training
    """
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))

    print("Model:")
    print(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            # need to install warpctc. see our guideline.
            from warpctc_pytorch import CTCLoss 
            criterion = CTCLoss()
        else:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
        
    scheduler = None 
    # Setup learning rate scheduler
    if opt.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)
    elif opt.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.T_max)
    elif opt.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=opt.mode, factor=opt.factor, patience=opt.patience, verbose=opt.verbose)
    print("Optimizer:")
    print(optimizer)

    if scheduler:
        print(f"Using learning rate scheduler: {scheduler}")
    else:
        print("No learning rate scheduler specified.")

    """ Create history file for tracking """
    history_file = f'./saved_models/{opt.exp_name}/history_{opt.exp_name}.csv'
    with open(history_file, 'w', newline='') as f:
        writer = csv.writer(f)
        row= ['iteration', 'train_loss', 'valid_loss', 'accuracy', 'norm_ED', 'batch_cer', 'average_cer']
        
        if opt.scheduler == 'plateau':
            row.append('learning_rate')
        
        writer.writerow(row)
    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter

    cumulative_cer = 0.0
    iteration_count = 0
    
    
    while(True):
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                cost = criterion(preds, text, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)

        else:
            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)

        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            elapsed_time = time.time() - start_time
            # for log
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                    # Apply the learning rate scheduler for ReduceLROnPlateau
                if opt.scheduler == 'plateau':
                    #current_lr = optimizer.param_groups[0]['lr']
                    #rop_log= f'Current learning rate: {current_lr:.5e}' # Log for ReduceLROnPlateau
                    scheduler.step(valid_loss)  # Adjust learning rate based on validation loss
                
                
                # CER Calculation and Logging
                cer_sum = 0
                total_length = 0
                for gt, pred in zip(labels, preds):
                    if 'Attn' in opt.Prediction:  # Remove special end tokens
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]
                    cer_sum += calculate_cer(gt, pred) * len(gt)
                    total_length += len(gt)
                batch_cer = cer_sum / max(1, total_length)  # Average CER

                # Update cumulative CER and count
                cumulative_cer += batch_cer
                iteration_count += 1
                average_cer = cumulative_cer / iteration_count
                
                # Current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                model.train()

                # Log iterations
                iter_log= f'{"Iterations":17s}: [{iteration+1} / {opt.num_iter}] \t\t\t{"Elapsed_time":17s}: {elapsed_time:.3f} \t\t\t{"Learning_rate":17s}: {current_lr:.3e}\n'
        
                # Log training and validation metrics
                loss_log = f'{"Train_loss":17s}: {loss_avg.val():.5f} \t\t\t{"Valid_loss":17s}: {valid_loss:0.5f} \t\t\t{"Batch_CER":17s}: {batch_cer:0.4f}'
        
                # Log metrics to history file
                with open(history_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row= [iteration + 1, loss_avg.val().item(), valid_loss.item(), current_accuracy, current_norm_ED, batch_cer, average_cer]
                    if opt.scheduler == 'plateau':
                        row.append(current_lr)
                    writer.writerow(row)
                loss_avg.reset()


                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
                
                # Current accuracy log
                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f} \t\t\t{"Current_norm_ED":17s}: {current_norm_ED:.2f} \t\t\t{"Average_CER":17s}: {average_cer:.4f}'
                
                # Best accuracy log
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f} \t\t\t{"Best_norm_ED":17s}: {best_norm_ED:.2f}'
                
                # Log details
                loss_model_log = f'{iter_log}\n{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # Show some predicted results
                dashed_line = '-' * 125
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | {"Confidence & True/False":25s}  | Character Error Rate (CER)'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    cer = calculate_cer(gt, pred)
                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:.4f}\t\t{str(pred == gt):10s} | {cer:.2f}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e+5 iter.
        if (iteration + 1) % 1e+5 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{iteration+1}.pth')

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    
    parser.add_argument('--scheduler', type=str, choices=['step', 'cosine', 'plateau'], default=None, help='Type of learning rate scheduler to use.')
    
    # For StepLR scheduler, you need step_size and gamma arguments
    parser.add_argument('--step_size', type=int, default=10, help='The period of learning rate decay (for StepLR). Default is 10.')
    parser.add_argument('--gamma', type=float, default=0.1, help='The factor by which the learning rate will be reduced (for StepLR). Default is 0.1.')
    # For CosineAnnealingLR scheduler, you need T_max argument
    parser.add_argument('--T_max', type=int, default=50, help='The maximum number of iterations for cosine annealing (for CosineAnnealingLR). Default is 50.')
    # For ReduceLROnPlateau scheduler, you might want a patience argument
    parser.add_argument('--mode', type=str, default="min", help='Scheduler mode (for ReduceLROnPlateau).')
    parser.add_argument('--factor', type=float, default=0.1, help='Factor by which learning rate will be reduced (for ReduceLROnPlateau).')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs with no improvement after which learning rate will be reduced (for ReduceLROnPlateau).')
    parser.add_argument('--verbose', type=bool, default=True, help='ReduceLROnPlateau verbose.')

    opt = parser.parse_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)
