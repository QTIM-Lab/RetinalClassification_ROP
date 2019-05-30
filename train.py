import sys
import re
import time
from os.path import isfile
from tqdm import tqdm
from sklearn.metrics import classification_report

from model import *
from utils import *

def train(model_path, dataloader, data_size):
    num_epochs = 50 # TODO: add as arg when running
    num_batches = len(dataloader)
    num_classes = 3
    model = inceptionnet(num_classes)

    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-4)
    # optimizer = torch.optim.SGD(model.fc.parameters(), lr = 1e-4) # param.require_grad = False
    weights = torch.cuda.FloatTensor([1./ (4648./192), 1./ (825./192), 1. / (192./192)])
    criterion = nn.CrossEntropyLoss(weight=weights)
    epoch = load_checkpoint(model_path, model) if isfile(model_path) else 0
    filename = re.sub("\.epoch[0-9]+$", "", model_path)
    print("Training model on {} images".format(data_size))
    start_time = time.time()

    for ei in range(epoch + 1, epoch + num_epochs + 1):
        print("Epoch {}/{}".format(ei, num_epochs))
        model.train()
        loss_sum = 0.
        acc_train = 0.
        y_true = []
        y_pred = []
        timer = time.time()
        for i, data in enumerate(dataloader):
            print("\rTraining batch {}/{}".format(i+1, num_batches), end='', flush=True)
            x, y = data['image'], data['plus'] # x = input images, y = labels
            if CUDA:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                model.cuda()
            else:
                x = Variable(x), Variable(y)
            model.zero_grad()
            optimizer.zero_grad()
            outputs = model(x)
            out, aux_out = outputs[0], outputs[1]
            _, preds = torch.max(out.data, 1)
            loss = sum((criterion(o, y) for o in outputs))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            acc_train += torch.sum(preds == y.data).double()
            y_true.extend(y.data.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            del x, y, outputs, preds
            torch.cuda.empty_cache()

        timer = time.time() - timer
        loss_sum /= data_size
        acc_train /= data_size

        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, model, ei, loss_sum, timer)

        print("Epoch {} result: ".format(ei))
        print("Avg loss (train): {:.4f}".format(loss_sum))
        print("Avg acc (train): {:.4f}".format(acc_train))
        print(classification_report(y_true, y_pred))

    elapsed_time = time.time() - start_time
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

if __name__ == "__main__":
    print("cuda: %s" % CUDA)
    model_path = sys.argv[1] # model_path
    dataloader = sys.argv[2] # dataloader file
    # data_size = TODO
    train(model_path, dataloader, data_size)
