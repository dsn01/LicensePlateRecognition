import torch
from torch.nn.functional import log_softmax
from ctc_decoder import ctc_decode


def evaluate(net, val_loader, criterion, device, decode_method='beam_search', beam_size=10):
    net.eval()
    total_count = 0
    total_loss = 0
    total_correct = 0
    wrong_cases = []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, targets, target_lengths = [d.to(device) for d in data]
            outputs = net(images)
            log_probs = log_softmax(outputs, dim=2)
            batch_size = images.size(0)
            input_lengths = torch.LongTensor([outputs.size(0)]*batch_size)
            # 计算CTCLoss
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            # CTC解码预测编号
            preds = ctc_decode(
                log_probs, method=decode_method, beam_size=beam_size)
            # 真实编号
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            total_count += batch_size
            total_loss += loss.item()
            # 计算正确率和错误样例
            target_length_counter = 0
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter+target_length]
                target_length_counter += target_length
                if pred == real:
                    total_correct += 1
                else:
                    wrong_cases.append((real, pred))
    metrics = {
        'loss': total_loss / total_count,
        'acc': total_correct / total_count,
        'wrong_cases': wrong_cases
    }
    return metrics
