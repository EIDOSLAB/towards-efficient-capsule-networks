import torch
import time
import wandb
import torch.nn.functional as F
from tqdm import *

def test(logging, config, loader, model, criterion, writer, epoch, device, imgdir, split="test"):
    if split=="test":
        logging.info("-------------------------------------- Testing epoch {} --------------------------------------".format(epoch))
    else:
        logging.info("-------------------------------------- Validation epoch {} --------------------------------------".format(epoch))
    return test_capsnet(logging, config, loader, model, criterion, writer, epoch, device, imgdir, split)


def test_capsnet(logging, config, loader, model, criterion, writer, epoch, device, imgdir, split):
    loss = 0
    tot_samples = len(loader.sampler)

    correct = 0

    all_labels = []
    all_pred = []

    model.eval()

    start_time = time.time()
    batch_index = 0
    for data, target in tqdm(loader):
        # Store the indices for calculating accuracy
        label = target.unsqueeze(0).type(torch.LongTensor)

        batch_size = data.size(0)
        # Transform to one-hot indices: [batch_size, 10]
        target_encoded = F.one_hot(target, config.num_classes)
        #assert target.size() == torch.Size([batch_size, 10])

        # Use GPU if available
        data, target_encoded = data.to(device), target_encoded.to(device)

        # Output predictions
        class_caps_poses, class_caps_activations = model(data)
        c_loss = criterion(class_caps_activations, target_encoded)

        loss += c_loss.item()

        # Count correct numbers
        # pred: [batch_size,]
        k=1
        pred = torch.topk(class_caps_activations, k=k, dim=1)[1].type(torch.LongTensor)
        correct += pred.eq(label.view_as(pred)).cpu().sum().item()

        # Classification report
        label_flat = label.view(-1)
        pred_flat = pred.view(-1)
        all_labels.append(label_flat)
        all_pred.append(pred_flat)

        batch_index += 1
    # Print time elapsed for every epoch
    end_time = time.time()
    formatted_epoch = str(epoch).zfill(len(str(config.epochs - 1)))
    logging.info('\nEpoch {} takes {:.0f} seconds for {}.'.format(formatted_epoch, end_time - start_time, split))

    # Log test losses
    loss /= len(loader)
    acc = correct / tot_samples

    if writer: 
        writer.add_scalar('{}/loss'.format(split), loss, epoch)
        writer.add_scalar('{}/accuracy'.format(split), acc, epoch)

    wandb.log({'{}/loss'.format(split): loss}, step=epoch)
    wandb.log({'{}/accuracy'.format(split): acc}, step=epoch)

    # Print test losses
    logging.info("{} loss: {:.4f}".format(split, loss))

    logging.info("{} accuracy: {}/{} ({:.2f}%)".format(split, correct, tot_samples,
                                                    100. * correct / tot_samples))
    logging.info("{} error: {}/{} ({:.2f}%)\n".format(split, tot_samples - correct,  tot_samples,
                                                 100. * (1 - correct / tot_samples)))
    return loss, acc