import torch

def ensemble(models, model_num, test_iter, batch_size, device):
    for i in range(model_num):
        models[i].eval()

    with torch.no_grad():
        test_loss = 0
        total = 0
        correct = 0

        for batch_img, batch_lab in test_iter:
            X = batch_img.view(-1, 28 * 28 * 3).to(device)
            Y = batch_lab.to(device)
            for i in range(model_num):
                if i == 0:
                    y_pred = models[i](X)
                else:
                    y_pred += models[i](X)

            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == Y).sum().item()
            total += batch_img.size(0)
        
        val_acc = 100 * correct / total
    
    return val_acc