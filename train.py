from segnet import *
from CamVid import *
import torch
import time
from torch.utils.data import Dataset, DataLoader
import pdb

batch_size = 1   # Number of samples in each batch
epoch_num = 4      # Number of epochs to train the network
lr = 0.0005        # Learning rate

if __name__ == "__main__":
    img_dir = "701_StillsRaw_full"
    mask_dir = "LabeledApproved_full"


    dataset = CamVid(img_dir=img_dir, mask_dir=mask_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    segnet = Segnet(32)
    optimizer = torch.optim.Adam(segnet.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    segnet.train()
    for epoch in range(epoch_num):
        loss_f = 0
        t_start = time.time()

        for batch in dataloader:
            input_tensor = torch.autograd.Variable(batch['image'])
            target_tensor = torch.autograd.Variable(batch['mask'])

            #  if CUDA:
                #  input_tensor = input_tensor.cuda(GPU_ID)
            #      target_tensor = target_tensor.cuda(GPU_ID)

            softmaxed_tensor = segnet(input_tensor)


            optimizer.zero_grad()
            pdb.set_trace()
            loss = criterion(softmaxed_tensor, target_tensor)
            loss.backward()
            optimizer.step()


            loss_f += loss.float()
            prediction_f = softmaxed_tensor.float()

        delta = time.time() - t_start
        is_better = loss_f < prev_loss

        if is_better:
            prev_loss = loss_f
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pth"))

        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s".format(epoch+1, loss_f, delta))
