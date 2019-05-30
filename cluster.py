import sys
import re
import time
import csv
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AffinityPropagation

from model import *
from utils import *

def cluster(model_path, dataloader, data_size):
    num_batches = len(dataloader)
    num_classes = 3
    model = inceptionnet(num_classes)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    # get features for clustering
    feature_extractor = nn.Sequential(*list(model.fc.children())[:-1]) # 1024 features
    model.fc = feature_extractor
    model.train(False)
    model.eval()
    print("Evaluating model on {} images".format(data_size))
    start_time = time.time()

    img_names = []
    plus_labels = []
    features_extracted = []

    csv_file_name = model_path.split('/')[-1] + '_clusters.csv'
    with open(csv_file_name, mode='w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['image', 'plus', 'cluster'])

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                print("\rEvaluating batch {}/{}".format(i+1, num_batches), end='', flush=True)
                x, y, imgs = data['image'], data['plus'], data['img_name']
                if CUDA:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    model.cuda()
                else:
                    x = Variable(x), Variable(y)
                features = model(x)
                img_names.extend(imgs)
                plus_labels.extend(y.data.cpu().numpy())
                features_extracted.extend(features.cpu().numpy())
                del x, y, features
                torch.cuda.empty_cache()

        predicted = KMeans(n_clusters=3, random_state=0).fit_predict(features_extracted)
        # predicted = DBSCAN().fit(features_extracted).labels_
        # predicted = MeanShift(bandwidth=3).fit_predict(features_extracted)
        # predicted = AffinityPropagation().fit_predict(features_extracted)

        for i in range(len(plus_labels)):
            writer.writerow([img_names[i], plus_labels[i].item(), predicted[i].item()])

    elapsed_time = time.time() - start_time
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    

if __name__ == "__main__":
    print("cuda: %s" % CUDA)
    model_path = sys.argv[1] # model_path
    dataloader = sys.argv[2] # dataloader file
    # data_size = TODO
    cluster(model_path, dataloader, data_size)