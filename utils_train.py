from selectors import EpollSelector
import numpy as np
from tqdm.notebook import trange, tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad, Variable

from trace_regressor import *
from poss_classifier import *
from transformers import AdamW, get_constant_schedule_with_warmup
# device = "cuda" if torch.cuda.is_available() else "cpu"

from torch.utils.tensorboard import SummaryWriter


def train(model, train_loader, val_loader, args, critic=None,
          gan_loss=False, phys_loss=False, phys_coeff=1, dist_coeff=0, device='cuda', option_type='no'):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if critic is not None:
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)

    if isinstance(model, TraceSetTransformer):
        optimizer = AdamW(model.parameters(), lr=1e-3)
        #scheduler = get_constant_schedule_with_warmup(optimizer, len(train_loader) * 2)

    num_epochs = 1000
    iter_count = 0
    best_val_loss = 1000  # for trace_regressor
    best_val_acc = 0.5    # for poss_classifier
    best_val_dist = 1000

    lr_count = 0
    lr_patience = 5

    stop_count = 0

    stop_patience = 10

    model_type = ''
    loss_type = ''
    if (
        isinstance(model, TraceLSTM) or
        isinstance(model, TraceSeq2Seq) or
        isinstance(model, TraceTransformer) or
        isinstance(model, TraceSetTransformer) or
        isinstance(model, TraceSetLSTM)
    ):
        model_type = model.target_type

        if isinstance(model, TraceLSTM):
            model_type += '_lstm'
        elif isinstance(model, TraceSetLSTM):
            model_type += '_setlstm'
        elif isinstance(model, TraceSeq2Seq):
            model_type += '_seq2seq'
        elif isinstance(model, TraceTransformer):
            model_type += '_transformer'
        else:
            model_type += '_settransformer'

        loss_type = 'phys' if phys_loss else 'mse'
        loss_type = 'gan' if gan_loss else loss_type

    elif isinstance(model, PossClassifier) or isinstance(model, PossTransformerClassifier):
        model_type = 'poss_classifier'

    writer = SummaryWriter(f'./runs/{args.model_type}_bs_{args.batch_size}_hidden_{args.hidden_dim}_phy_{args.phy_loss}_FM_metrica_train_Metrica_valid_GK')

    for epoch in range(num_epochs):
        train_loss = 0
        train_dist = 0  # for trace_regressor
        train_acc = 0   # for poss_classifier

        val_loss = 0
        val_dist = 0    # for trace_regressor
        val_acc = 0     # for poss_classifier

        train_physloss = 0
        train_distloss = 0
        train_ganloss = 0
        train_criticloss = 0
        train_critic_accuracy = 0
        train_reconloss = 0

        val_physloss = 0
        val_distloss = 0
        val_ganloss = 0
        val_criticloss = 0
        val_critic_accuracy = 0
        val_reconloss = 0

        for i, data in enumerate(tqdm(train_loader)):

            input_traces = data[0].to(device)
            target_traces = data[1].to(device)
            #input_heatmaps = data[2].to(device)
            input_heatmaps = torch.zeros(data[0].shape[0], target_traces.shape[1], 2, 108, 72).to(device)
            target_team_poss = (data[-1].to(device)).long()
            target_team_poss = target_team_poss.view(-1)


            # Forward propagation and loss computation
            if 'ball' in model_type:

                pred_traces, pred_team_poss = model(input_heatmaps, input_traces, target_traces)
                
                player_pred_traces = pred_traces[:,:,:-2]  
                ball_pred_traces = pred_traces[:,:,-2:]
                
                delayed_input_traces = torch.stack([torch.cat([input_traces[:,1:,0::4], input_traces[:,-1:,0::4]], -2), torch.cat([input_traces[:,1:,1::4], input_traces[:,-1:,1::4]], -2)], -1).view(input_traces.shape[0], input_traces.shape[1], -1)           
                #delayed_input_traces = torch.cat([input_traces[:,1:,:], input_traces[:,-1:,:]], -2)
                                
                loss = nn.MSELoss()(ball_pred_traces, target_traces) + nn.MSELoss()(player_pred_traces[:,:-1,:], delayed_input_traces[:,:-1,:]) + nn.CrossEntropyLoss()(pred_team_poss.reshape(-1, 3), target_team_poss)
                train_dist += trace_dist_fn(ball_pred_traces, target_traces).item()

                pred_team_poss = torch.argmax(pred_team_poss, -1).cuda().view(-1)
                
                train_acc += (target_team_poss == pred_team_poss).sum().item() / torch.ones(target_team_poss.shape).sum().item()

                if gan_loss:
                    D_fake = critic(player_heatmaps, ball_pred_traces)
                    G_loss = -torch.mean(D_fake)
                    loss += G_loss

                train_distloss += dist_coeff * (covered_dist_fn(ball_pred_traces, target_traces)).item()
                train_physloss += phys_coeff * (phys_loss_fn(ball_pred_traces, input_traces)).item()

                if phys_loss:
                    loss += (
                        phys_coeff * phys_loss_fn(ball_pred_traces, input_traces) +
                        dist_coeff * covered_dist_fn(ball_pred_traces, target_traces)
                    )

            elif 'gk' in model_type:
                pred_traces = model(input_heatmaps, input_traces, target_traces)
                loss = nn.MSELoss()(pred_traces, target_traces)
                train_dist += (
                    trace_dist_fn(pred_traces[:, :, 0:2], target_traces[:, :, 0:2]).item() +
                    trace_dist_fn(pred_traces[:, :, 2:4], target_traces[:, :, 2:4]).item()
                ) / 2

            elif model_type == 'poss_classifier':
                reshape_dim = input_traces.shape[0] * input_traces.shape[1]
                target_team_poss = (data[-1].to(device)).long()
                
                log_likelihood, pred_team_poss, pred_traces, first_traces = model(input_heatmaps, input_traces, target_traces, target_team_poss)
                
                target_team_poss = target_team_poss.view(-1)
                
                pred_team_poss = torch.tensor(pred_team_poss).cuda().view(-1)

                train_dist += trace_dist_fn(pred_traces, target_traces).item()

                train_distloss += dist_coeff * (covered_dist_fn(pred_traces, target_traces)).item()
                train_physloss += phys_coeff * (phys_loss_fn(pred_traces, input_traces)).item()                                

                #loss = nn.CrossEntropyLoss()(pred_team_poss, target_team_poss) + nn.MSELoss()(pred_traces, target_traces)
                loss = -1 * log_likelihood + nn.MSELoss()(pred_traces, target_traces) + nn.MSELoss()(first_traces, target_traces)

                if phys_loss:
                    loss += (
                        phys_coeff * phys_loss_fn(pred_traces, input_traces) +
                        dist_coeff * covered_dist_fn(pred_traces, target_traces)
                    )

                total_points = target_team_poss.shape[0]
                
                correct_points = torch.sum(pred_team_poss == target_team_poss).item()

                team_1_correct_points = 0 # torch.sum(((target_team_poss > 0) & (target_team_poss <= 22)) * ((pred_team_poss.argmax(dim=1) > 0) & (pred_team_poss.argmax(dim=1) <= 22) )).item()
                team_2_correct_points = 0 # torch.sum(((target_team_poss > 0) & (target_team_poss > 11)) * ((pred_team_poss.argmax(dim=1) > 0) & (pred_team_poss.argmax(dim=1) > 11) )).item()


                acc = (correct_points + team_1_correct_points + team_2_correct_points) / total_points
                train_acc += acc

            else:
                loss = 0

            # Parameter Initialization
            optimizer.zero_grad()

            # Backpropagation to get gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            # Parameter Update
            optimizer.step()
            #if isinstance(model, TraceTransformer):
            #    scheduler.step()

            if gan_loss:
                #with torch.backends.cudnn.flags(enabled=False):

                pred_traces, pred_team_poss = model(input_heatmaps, input_traces)
                
                player_pred_traces = pred_traces[:,:,:-2]
                ball_pred_traces = pred_traces[:,:,-2:]
                
                D_fake = critic(input_heatmaps, ball_pred_traces.detach())
                D_real = critic(input_heatmaps, target_traces)

                gradient_penalty = _gradient_penalty(
                    player_heatmaps, target_trace, ball_pred_traces, critic, 10, device
                )
                D_loss = -(torch.mean(D_real) - torch.mean(D_fake)) + gradient_penalty

                critic_optimizer.zero_grad()
                D_loss.backward()
                critic_optimizer.step()

                train_criticloss += float(D_loss.detach().cpu())
                train_ganloss += float(G_loss.detach().cpu())

                train_critic_accuracy += (
                    (D_real > 0.5).float().mean() + (D_fake < 0.5).float().mean()
                ).detach().cpu() / 2

                D_fake = critic(player_heatmaps, pred_trace)

            train_loss += loss.item()
            model.progress += [loss.item()]

            iter_count += 1

            if i % 100 == 0:
                print(
                    "Epoch {}/{} - Train Loss: {:.4f}, Dist: {:.4f}, "
                    "Phys: {:.4f}, Dist Loss: {:.4f}, Acc: {:.4f}, "
                    "GAN: {:.4f}, Critic Loss: {:.4f},  Critic Acc: {:.4f}, Val Recon: {:.4f}".format(
                        epoch + 1, len(train_loader), train_loss / (i+1), train_dist / (i+1),
                        train_physloss / (i+1), train_distloss / (i+1), train_acc / (i+1),
                        train_ganloss / (i+1), train_criticloss / (i+1), train_critic_accuracy / (i+1), val_reconloss / (i+1)
                    )
                )

        model.eval()
        for data in tqdm(val_loader):

            input_traces = data[0].to(device)
            target_traces = data[1].to(device)
            #input_heatmaps = data[2].to(device)
            input_heatmaps = torch.zeros(data[0].shape[0], target_traces.shape[1], 2, 108, 72)
            target_team_poss = (data[-1].to(device)).long()
            target_team_poss = target_team_poss.view(-1)

            # target_weights = data[-3].to(device)
            # pitch_mask = data[-1].to(device)

            # Forward propagation and loss computation
            if 'ball' in model_type:
                with torch.no_grad():
                    pred_traces, pred_team_poss = model(input_heatmaps, input_traces, target_traces)
                    
                player_pred_traces = pred_traces[:,:,:-2]
                ball_pred_traces = pred_traces[:,:,-2:]

                loss = nn.MSELoss()(ball_pred_traces, target_traces).item()
                val_dist += trace_dist_fn(ball_pred_traces, target_traces).item()
                
                pred_team_poss = torch.argmax(pred_team_poss, -1).cuda().view(-1)
                
                val_acc += (target_team_poss == pred_team_poss).sum().item() / torch.ones(target_team_poss.shape).sum().item()

                if phys_loss:
                    loss += (
                        phys_coeff * phys_loss_fn(ball_pred_traces, input_traces).item() +
                        dist_coeff * covered_dist_fn(ball_pred_traces, target_traces).item()
                    )

                val_distloss += dist_coeff * covered_dist_fn(ball_pred_traces, target_traces).item()
                val_physloss += phys_coeff * phys_loss_fn(ball_pred_traces, input_traces).item()

            elif 'gk' in model_type:
                pred_traces = model(input_heatmaps, input_traces, target_traces)
                loss = nn.MSELoss()(pred_traces, target_traces).item()
                val_dist += (
                    trace_dist_fn(pred_traces[:, :, 0:2], target_traces[:, :, 0:2]).item() +
                    trace_dist_fn(pred_traces[:, :, 2:4], target_traces[:, :, 2:4]).item()
                ) / 2

            elif model_type == 'poss_classifier':
                reshape_dim = input_traces.shape[0] * input_traces.shape[1]
                
                target_team_poss = (data[-1].to(device)).long().view(-1)

                with torch.no_grad():
                    target_traces_zeros = torch.zeros(target_traces.shape).to(device)
                    pred_team_poss, pred_traces, first_traces = model(input_heatmaps, input_traces, target_traces_zeros)

                pred_team_poss = torch.tensor(pred_team_poss).cuda().view(-1)

                val_dist += trace_dist_fn(pred_traces, target_traces).item() 

                val_distloss += dist_coeff * (covered_dist_fn(pred_traces, target_traces)).item()
                val_physloss += phys_coeff * (phys_loss_fn(pred_traces, input_traces)).item()

                loss = nn.MSELoss()(pred_traces, target_traces) # + nn.MSELoss()(first_traces, target_traces)

                if phys_loss:
                    loss += (
                        phys_coeff * phys_loss_fn(pred_traces, input_traces) +
                        dist_coeff * covered_dist_fn(pred_traces, target_traces)
                    )

                total_points = target_team_poss.shape[0]
                correct_points = torch.sum(pred_team_poss == target_team_poss).item()

                team_1_correct_points = 0 # torch.sum(((target_team_poss > 0) & (target_team_poss <= 22)) * ((pred_team_poss.argmax(dim=1) > 0) & (pred_team_poss.argmax(dim=1) <= 22))).item()
                team_2_correct_points = 0 # torch.sum(((target_team_poss > 0) & (target_team_poss > 11)) * ((pred_team_poss.argmax(dim=1) > 0) & (pred_team_poss.argmax(dim=1) > 11) )).item()


                acc = (correct_points + team_1_correct_points + team_2_correct_points) / total_points

                val_acc += acc

                # input_trace = torch.cat([target_trace, player_trace], dim=-1)
                # pred_poss = model.forward(input_trace)
                # pred_poss = pred_poss.view(pred_poss.size(0) * pred_poss.size(1), -1)

                # target_poss = weights_to_poss(target_weights, device)
                # target_poss = target_poss.view(target_poss.size(0) * target_poss.size(1))

                # loss = cross_entropy_fn(pred_poss, target_poss).item()
                # loss, trace_dist = radar_loss(pred_poss, target_poss, player_trace, target_trace)
                # val_dist += trace_dist.mean().item()

            else:
                loss = 0

            val_loss += loss

        mean_train_loss = train_loss / len(train_loader)
        mean_train_dist = train_dist / len(train_loader)
        mean_val_loss = val_loss / len(val_loader)
        mean_val_dist = val_dist / len(val_loader)
        mean_train_acc = train_acc / len(train_loader)
        mean_val_acc = val_acc / len(val_loader)

        writer.add_scalar('Loss/train', mean_train_loss, epoch)
        writer.add_scalar('Loss/val', mean_val_loss, epoch)

        writer.add_scalar('Loss/train_phys', train_physloss / len(train_loader), epoch)
        
        writer.add_scalar('Loss/train_dist', train_distloss / len(train_loader), epoch)
        writer.add_scalar('Loss/train_gan', train_ganloss / len(train_loader), epoch)
        writer.add_scalar('Loss/train_critic', train_criticloss / len(train_loader), epoch)
        writer.add_scalar('Loss/train_gan_accuracy', train_critic_accuracy / len(train_loader), epoch)
        writer.add_scalar('Loss/train_recon', train_reconloss / len(train_loader), epoch)
        writer.add_scalar('Loss/val_recon', val_reconloss / len(train_loader), epoch)

        writer.add_scalar('Loss/val_phys', val_physloss / len(val_loader), iter_count)
        writer.add_scalar('Loss/val_dist', val_distloss / len(val_loader), epoch)

        writer.add_scalar('Distance/train', mean_train_dist, epoch)
        writer.add_scalar('Distance/val', mean_val_dist, epoch)
        
        writer.add_scalar('Pos Accuracy/train', mean_train_acc, epoch)
        writer.add_scalar('Pos Accuracy/val', mean_val_acc, epoch)

        if 'ball' in model_type or 'gk' in model_type:
            print("Train Loss: {:.4f}, Acc: {:.4f}, Dist: {:.4f}, Phys: {:.4f}, Val Loss: {:.4f}, Acc: {:.4f}, Dist: {:.4f}, Phys: {:.4f}".format(
                mean_train_loss, mean_train_acc, mean_train_dist, train_physloss / len(train_loader), mean_val_loss, mean_val_acc, mean_val_dist, val_physloss / len(val_loader)
            ))
            print("Val Loss: {:.4f}, Acc: {:.4f}, Dist: {:.4f}, Phys: {:.4f}, Dist Loss: {:.4f}, Recon: {:.4f}".format(
                mean_val_loss, mean_val_acc, mean_val_dist, val_physloss / len(val_loader), val_distloss / len(val_loader), val_reconloss / len(val_loader)
            ))
            if mean_val_dist < best_val_dist:
                torch.save(model.state_dict(), f'saved_models/{args.model_type}_bs_{args.batch_size}_hidden_{args.hidden_dim}_phy_{args.phy_loss}_FM_metrica_train_Metrica_valid_GK')
                # torch.save(critic.state_dict(), f'saved_models/trace_discriminator.pt')
                print(f"Model saved in '{model_type}_{loss_type}_{option_type}.pt'.")
                # print(f"Critic saved in trace_discriminator.pt'.")
                lr_count = 0
                stop_count = 0
                best_val_dist = mean_val_dist
            else:
                lr_count += 1
                stop_count += 1
            
        elif model_type == 'poss_classifier':
            print("Train Loss: {:.4f}, Acc: {:.4f}, Dist: {:.4f}, Phys: {:.4f}, Val Loss: {:.4f}, Acc: {:.4f}, Dist: {:.4f}, Phys: {:.4f}".format(
                mean_train_loss, mean_train_acc, mean_train_dist, train_physloss / len(train_loader), mean_val_loss, mean_val_acc, mean_val_dist, val_physloss / len(val_loader)
            ))
            print("Val Loss: {:.4f}, Acc: {:.4f}, Dist: {:.4f}, Phys: {:.4f}, Dist Loss: {:.4f}, Recon: {:.4f}".format(
                mean_val_loss, mean_val_acc, mean_val_dist, val_physloss / len(val_loader), val_distloss / len(val_loader), val_reconloss / len(val_loader)
            ))
            if mean_val_dist < best_val_dist:
                torch.save(model.state_dict(), f'saved_models/{args.model_type}_{args.hidden_dim}_{args.batch_size}_pysic_{args.phy_loss}_Metrica_GPS_FM.pt')
                print(f"Model saved in '{args.model_type}_{args.hidden_dim}_{args.batch_size}_pysic_{args.phy_loss}.pt'.")
                lr_count = 0
                stop_count = 0
                best_val_dist = mean_val_dist

            else:
                lr_count += 1
                stop_count += 1

        if lr_count > lr_patience:
            lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr * 0.5
            print(f"Learning rate decay to {lr * 0.5}.")
            lr_count = 0
            # lr_patience += 1

        if stop_count > stop_patience:
            print("Early stopping.")
            return
        else:
            model.train()


def weights_to_poss(target_weights, device='cuda'):
    target_poss = torch.zeros(target_weights.shape).to(device)
    target_poss[:, :, :23] = target_weights[:, :, :23]
    target_poss[:, :, 23] = target_weights[:, :, [23, 24]].sum(dim=-1)  # lower side line
    target_poss[:, :, 24] = target_weights[:, :, [25, 26]].sum(dim=-1)  # upper side line
    target_poss[:, :, 25] = target_weights[:, :, [23, 25]].sum(dim=-1)  # left end line
    target_poss[:, :, 26] = target_weights[:, :, [24, 26]].sum(dim=-1)  # right end line
    # return target_poss.max(dim=-1).indices
    return target_poss.clamp(max=1)


def radar_loss_fn(pred_encoded, true_encoded, player_trace, true_trace, alpha=200):
    # pred_encoded = torch.maximum(pred_encoded, eps)
    # denom = torch.sum(pred_encoded[:, :, 1:], dim=2)
    # pred_x = torch.sum(pred_encoded[:, :, 1:] * player_trace[:, :, 0::2], dim=2) / denom
    # pred_y = torch.sum(pred_encoded[:, :, 1:] * player_trace[:, :, 1::2], dim=2) / denom

    pred_x = torch.sum(pred_encoded * player_trace[:, :, 0::2], dim=2)
    pred_y = torch.sum(pred_encoded * player_trace[:, :, 1::2], dim=2)
    pred_trace = torch.stack([pred_x, pred_y], dim=2)

    cross_entropy = cross_entropy_fn(pred_encoded, true_encoded)
    trace_dist = trace_dist_fn(pred_trace, true_trace)
    phys_loss = phys_loss_fn(pred_encoded, pred_trace, player_trace)

    loss = (cross_entropy * (trace_dist ** 2)).mean() + alpha * phys_loss
    return loss, trace_dist


def cross_entropy_fn(pred_poss, target_poss):
#     true_free = 1 - true_encoded[:, :, 1:23].sum(dim=-1, keepdim=True)
#     pred_free = 1 - pred_encoded[:, :, 0:22].sum(dim=-1, keepdim=True)
#     true_poss = torch.cat((true_free, true_encoded[:, :, 1:23]), dim=-1)
#     pred_poss = torch.cat((pred_free, pred_encoded[:, :, 0:22]), dim=-1)
    return -torch.sum(target_poss * pred_poss.log(), dim=-1).mean()


def trace_dist_fn(pred_trace, target_trace):
    # distances = torch.norm((true_trace - pred_trace) * pitch_mask, dim=2)
    return torch.norm(pred_trace - target_trace, dim=-1).mean()


def covered_dist_fn(pred_trace, target_trace):
    return torch.abs(torch.norm(target_trace[:,:-1] - target_trace[:,1:], dim=-1).mean(-1) - torch.norm(pred_trace[:,:-1] - pred_trace[:,1:], dim=-1).mean(-1)).mean()


def phys_loss_fn(pred_trace, player_trace, eps=torch.tensor(1e-7)):
    # Compute the angle between two consecutive velocity vectors
    # We skip the division by time difference, which is eventually reduced
    vels = pred_trace.diff(dim=1)
    eps = eps.to(vels.device)
    speeds = torch.linalg.norm(vels, dim=-1)
    cos_num = torch.sum(vels[:, :-1] * vels[:, 1:], dim=-1) + eps
    cos_denom = speeds[:, :-1] * speeds[:, 1:] + eps
    cosines = torch.clamp(cos_num / cos_denom, -1 + eps, 1 - eps)
    angles = torch.acos(cosines)

    # Compute the distance between the ball and the nearest player
    pred_trace = torch.unsqueeze(pred_trace, dim=2)
    player_x = player_trace[:, :, 0:44:2]
    player_y = player_trace[:, :, 1:44:2]
    player_trace = torch.stack([player_x, player_y], dim=-1)
    ball_dists = torch.linalg.norm(pred_trace - player_trace, dim=-1)
    nearest_dists = torch.min(ball_dists, dim=-1).values[:, 1:-1]

    # max_poss = pred_encoded[:, :, 0:22].max(dim=-1).values[:, 1:-1]

    # return -(torch.tanh(10*angles) * max_poss.log()).mean()
    return (torch.tanh(angles) * nearest_dists)
    # return (torch.tanh(angles) * nearest_dists).mean()

def phys_loss_fn_2(pred_trace, player_trace, eps=torch.tensor(1e-7)):
    # Compute the angle between two consecutive velocity vectors
    # We skip the division by time difference, which is eventually reduced
    pred_trace = torch.unsqueeze(pred_trace, dim=2)
    player_x = player_trace[:, :, 0:44:2]
    player_y = player_trace[:, :, 1:44:2]
    player_trace = torch.stack([player_x, player_y], dim=-1)

    ball_dists = torch.linalg.norm(pred_trace - player_trace, dim=-1)
    nearest_dists = torch.min(ball_dists, dim=-1).values[:, 1:-1]

    # max_poss = pred_encoded[:, :, 0:22].max(dim=-1).values[:, 1:-1]

    # return -(torch.tanh(10*angles) * max_poss.log()).mean()
    return nearest_dists.mean()


# def ball_close_function(output, input_data, threshold=3e-2):
#     output = output.repeat(1, 1, 22).view(input_data.shape[0], input_data.shape[1], -1, 2)
#     input_data = input_data.view(input_data.shape[0], input_data.shape[1], -1, 2)
#     min_dist_to_ball = torch.min(
#         torch.sqrt((output[:, :, :, 0] - input_data[:, :, :, 0]) ** 2 +
#                    (output[:, :, :, 1] - input_data[:, :, :, 1]) ** 2), dim=2
#     )[0][:, :-2]
#     return (min_dist_to_ball < threshold).float()
    # return torch.minimum(threshold / torch.min(torch.sqrt((output[:,:,:,0] - input_data[:,:,:,0]) ** 2 +
    # (output[:,:,:,1] - input_data[:,:,:,1]) ** 2), axis=2)[0][:,:-2], torch.tensor(1).to(device))
#
#
# def physical_loss_function(output, eps=1e-8):
#     lambda1 = 0.1
#
#     output = rolling_average(output)    # + torch.normal(mean=torch.zeros(output.shape), std=1e-5).to(device)
#
#     t_diff = output[:, 1:, :] - output[:, :-1, :]   # velocity vectors. Size: batch_siz * (time steps - 1) * 2
#     speed = torch.sqrt(t_diff[:, :, 0] ** 2 + t_diff[:, :, 1] ** 2 + eps)
#
#     inner_prod = (t_diff[:, 1:, :] * t_diff[:, :-1, :]).sum(2)
#     # inner product of two velocity vectors. Size: batch_size * (time steps - 2)
#     speed_square = speed[:, 1:] * speed[:, :-1]
#
#     term = torch.maximum(((speed[:, 1:] + eps) / (speed[:, :-1] + eps)),
#                          torch.ones(speed[:, 1:].shape).to(device)) - 1
#     physical_loss = 10 * speed[:, :-1] * torch.abs(1 - (inner_prod + eps) / speed_square) + lambda1 * term
#
#     return physical_loss * (((output > 0) & (output < 1))[:, :, 0] & ((output > 0) & (output < 1))[:, :, 1])[:, :-2]
#
#
# def rolling_average(coord_tensor, size=2):
#     new_tensor_list = []
#     for i in range(coord_tensor.shape[1]):
#         if i - size < 0:
#             start_idx = 0
#         else:
#             start_idx = i - size
#
#         if i + size > coord_tensor.shape[1]:
#             end_idx = coord_tensor.shape[1]
#         else:
#             end_idx = i + size
#
#         new_tensor_list.append(coord_tensor[:, start_idx:end_idx].mean(1))
#
#     return torch.stack(new_tensor_list).permute(1, 0, 2)

def _gradient_penalty(input_data, real_data, generated_data, critic, gp_weight=10, device='cpu'):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand_as(real_data).to(device)

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True).to(device)
    #if self.use_cuda:
    #    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    #with torch.backends.cudnn.flags(enabled=False):
    prob_interpolated = critic(input_data, interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.reshape(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - 1) ** 2).mean()