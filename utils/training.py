import torch
import torch.nn as nn
import torch.nn.functional as F


from tqdm import tqdm












def contrast_training_period(model, dataloader, contrast_trainig_epochs, optimizer, writer=None):
    data_nums = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    
    
    for epoch in range(contrast_trainig_epochs):
        with tqdm(total=(data_nums-1)//batch_size + 1, bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}',) as t:
            for data, target in dataloader:
                loss = contrast_feature(model, data, target, optimizer)
                if writer:
                    writer.add_scalar('contrast_loss', loss.item(), epoch)
                    
                    
            t.update(1)
            t.set_description_str(f"\33[36m【Pretrained Train Epoch {epoch + 1:03d}】") # 设置迭代数描述字符串
            t.set_postfix_str(f"epoch_train_loss={epoch_loss_list[-1] / batch_size:.4f}, epoch_time:{epoch_time}, total_time:{total_time} ") 



def contrast_feature(model, data, target, optimizer):
    loss = 0
    
    pre_middle_feature = model.get_middle_feature(data)
    target_middle_feature = model.get_middle_feature(target)
    loss = F.mse_loss(pre_middle_feature, target_middle_feature)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss