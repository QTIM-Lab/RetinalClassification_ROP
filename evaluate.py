import sys
import re
import time
import csv
from sklearn.metrics import classification_report, confusion_matrix

from model import *
from utils import *

def evaluate(model_path, dataloader, data_size):
    num_batches = len(dataloader)
    num_classes = 3
    weights = torch.cuda.FloatTensor([2./ (4648./192), 1.5/ (825./192), 1. / (192./192)])
    criterion = nn.CrossEntropyLoss(weight=weights)
    model = inceptionnet(num_classes)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.train(False)
    model.eval()
    print("Evaluating model on {} images".format(data_size))
    start_time = time.time()

    loss_test = 0.
    acc_test = 0.
    y_true = []
    y_pred = []

    csv_file_name = model_path.split('/')[-1] + '_' + str(data_size) + '_predictions.csv'
    with open(csv_file_name, mode='w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['image', 'true plus', 'predicted plus'])
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                print("\rEvaluating batch {}/{}".format(i+1, num_batches), end='', flush=True)
                x, y, img_name = data['image'], data['plus'], data['img_name']
                if CUDA:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    model.cuda()
                else:
                    x = Variable(x), Variable(y)
                outputs = model(x)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, y)
                loss_test += loss.item()
                acc_test += torch.sum(preds == y.data).double()
                y_true.extend(y.data.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                for i in range(len(img_name)):
                    writer.writerow([img_name[i], y.data[i].item(), preds[i].item()])
                del x, y, outputs, preds
                torch.cuda.empty_cache()

    loss_test /= data_size
    acc_test /= data_size
    elapsed_time = time.time() - start_time
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(loss_test))
    print("Avg acc (test): {:.4f}".format(acc_test))
    print(classification_report(y_true, y_pred))
    print("Confusion matrix: ", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    print("cuda: %s" % CUDA)
    model_path = sys.argv[1] # model_path
    dataloader = sys.argv[2] # dataloader file
    # data_size = TODO
    evaluate(model_path, dataloader, data_size)