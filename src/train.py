import time
import pynvml
import torch
import wandb
import torch.nn.functional as F
from tqdm import *


def get_memory_used_GB(device):
    gpu_index = str(device).split(":")[1]
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = (mem_info.used // 1024 ** 2)/1000
    return used
    # total = mem_info.total // 1024 ** 2
    # return round(used/total, 5)*100

def train(logging, config, train_loader, model, criterion, optimizer, scheduler, writer, epoch, device):
    logging.info(
        "-------------------------------------- Training epoch {} --------------------------------------".format(epoch))
    train_capsnet(logging, config, train_loader, model, criterion, optimizer, scheduler, writer, epoch,
                             device)


def train_capsnet(logging, config, train_loader, model, criterion, optimizer, scheduler, writer, epoch, device):
    tot_samples = len(train_loader.sampler)
    loss = 0
    correct = 0

    model.train()

    start_time = time.time()

    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (data, target) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            # Store the indices for calculating accuracy
            label = target.unsqueeze(0).type(torch.LongTensor)

            # Transform to one-hot indices: [batch_size, 10]
            target = F.one_hot(target, config.num_classes)
            # Use GPU if available
            data, target = data.to(device), target.to(device)

            class_caps_poses, class_caps_activations = model(data)

            c_loss = criterion(class_caps_activations, target)

            loss += c_loss.item()

            c_loss.backward()
            if batch_idx == 0:
                used_gpu = get_memory_used_GB(device)
                wandb.log({'used_gpu': used_gpu}, step=epoch)
            #Train step
            if (batch_idx + 1) % config.iter_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Count correct numbers
            # pred: [batch_size,]
            k=1
            pred = torch.topk(class_caps_activations, k=k, dim=1)[1].type(torch.LongTensor)
            correct += pred.eq(label.view_as(pred)).cpu().sum().item()

            formatted_epoch = str(epoch).zfill(len(str(config.epochs - 1)))

            # Print losses
            if batch_idx % config.print_every == 0:
                tepoch.set_postfix(loss=c_loss.item())
    # Print time elapsed for every epoch
    end_time = time.time()
    logging.info('\nEpoch {} takes {:.0f} seconds for training.'.format(formatted_epoch, end_time - start_time))

    # Log train losses
    loss /= len(train_loader)

    acc = correct / tot_samples

    # Log losses
    if writer: 
        writer.add_scalar('train/loss', c_loss.item(), epoch)
        writer.add_scalar('train/accuracy', acc, epoch)

    wandb.log({'train/loss': c_loss.item()}, step=epoch)
    wandb.log({'train/accuracy': acc}, step=epoch)

    # Print losses
    logging.info("Training loss: {:.4f} ".format(loss))

    logging.info("Training accuracy: {}/{} ({:.2f}%)".format(correct, len(train_loader.sampler),
                                                             100. * correct / tot_samples))
    logging.info(
        "Training error: {}/{} ({:.2f}%)".format(tot_samples - correct, tot_samples,
                                                 100. * (1 - correct / tot_samples)))

    if scheduler is not None:
        wandb.log({'lr': scheduler.get_last_lr()[0]}, step=epoch)
    if (config.decay_steps > 0 and epoch % config.decay_steps == 0):
        # Update learning rate
        if (scheduler is not None and scheduler.get_last_lr()[0] > config.min_lr):
            scheduler.step()
            logging.info('New learning rate: {}'.format(scheduler.get_last_lr()[0]))

    return loss, acc