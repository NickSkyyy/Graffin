from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from evaluate import *
from model import *
from util import *
from sklearn.manifold import TSNE
from train import *

import logging
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 24
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.dpi"] = 300
import time

# train_data = load("DBLP")
# data = get_dblp_seq(train_data[0], "degree", False)
# # train_data = get_sequences(train_data, "degree", False)
# print(data)
# input()
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
parser = ArgumentParser("PRO2", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve")

parser.add_argument("--dataset", default="Amazon_Photo", help="Dataset")
parser.add_argument("--epoch", default=200, type=int, help="Epoch")
parser.add_argument("--mpnn_type", default="gcn", help="Batch size")
parser.add_argument("--repeat", default=5, type=int, help="Repeat times")
parser.add_argument("--sort", default="degree", help="Sort rules")

args = parser.parse_args()
tsne = TSNE(n_components=2, random_state=0)
colors = [
  [37/255, 43/255, 128/255],
  [55/255, 82/255, 164/255],
  [60/255, 109/255, 180/255],
  [72/255, 198/255, 235/255],
  [129/255, 201/255, 152/255],
  [189/255, 214/255, 56/255],
  [251/255, 205/255, 17/255],
  [239/255, 94/255, 33/255],
  [235/255, 29/255, 34/255],
  [125/255, 19/255, 21/255],
]

# logs
logs_dir = os.path.abspath("./logs")
if not os.path.exists(logs_dir):
  os.makedirs(logs_dir)

logging.basicConfig(filemode='a')
formats = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler = logging.FileHandler(logs_dir + "/base.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formats)

logs = logging.getLogger()
logs.setLevel(logging.INFO)
logs.addHandler(file_handler)
logs.info(device)
logs.info(args)

if __name__ == "__main__":
  train_data = load(args.dataset)
  data = get_sequences(train_data, args.sort, False).to(device)
  logs.info(data)
  logs.info(data.bin)

  FEATURE_SIZE = data.x.shape[1]
  HIDDEN_SIZE = 1024
  NUM_CLASSES = len(torch.unique(data.y))
  MPNN_TYPE = args.mpnn_type

  class_acc = [[] for _ in range(NUM_CLASSES)]
  class_our_acc = [[] for _ in range(NUM_CLASSES)]
  tt = []
  tt_our = []
  
  for rep in range(args.repeat):
    logs.info("Repeat # %d" % rep)

    baseline = TestModel(FEATURE_SIZE, HIDDEN_SIZE, NUM_CLASSES, MPNN_TYPE, 0.5, device).to(device)
    # baseline = ImbGNN(FEATURE_SIZE, HIDDEN_SIZE, NUM_CLASSES, False).to(device)
    logs.info("Baseline")
    optimizer = torch.optim.Adam(baseline.parameters(), lr=0.01, weight_decay=5e-4)
    baseline.train()
    start = time.time()
    for epoch in range(args.epoch):
      optimizer.zero_grad()
      output = baseline(data)
      if args.dataset == "DBLP":
        output = output[:4057]
      loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
      # logs.info("Epoch {}: {}".format(epoch, loss))
      loss.backward()
      optimizer.step()
    end = time.time()
    tt.append(end - start)

    baseline.eval()
    output = baseline(data)
    if args.dataset == "DBLP":
      output = output[:4057]
    pred = output.argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(acc) / int(data.test_mask.sum())
    logs.info("Model Acc: %.4f" % acc)
    # acc = (pred[data.high_mask] == data.y[data.high_mask]).sum()
    # acc = int(acc) / int(data.high_mask.sum())
    # logs.info("High Acc: %.4f" % acc)
    # acc = (pred[data.low_mask] == data.y[data.low_mask]).sum()
    # acc = int(acc) / int(data.low_mask.sum())
    # logs.info("Low Acc: %.4f" % acc)
    for i, label in enumerate(data.label_mask):
      acc = (pred[label] == data.y[label]).sum()
      acc = int(acc) / int(label.sum())
      logs.info("Class %d Acc: %.4f" % (i, acc))
      class_acc[i].append(acc)
    logs.info("AUC_ROC: %.4f" % AUC_ROC(output.cpu(), data.y.cpu()))
    logs.info("F1: %.4f" % F1(pred.cpu(), data.y.cpu()))

    # t-SNE
    X = tsne.fit_transform(output.detach().cpu().numpy())
    Y = np.unique(data.y.cpu())
    plt.figure()
    for label in Y:
      mask = (data.y.cpu() == label)
      plt.scatter(X[mask, 0], X[mask, 1], c=colors[label], label="Label", s=50)
    # plt.legend()
    # plt.show()
    plt.savefig(".\\pic\\tsne\\base_{}_{}.png".format(args.dataset, args.mpnn_type))

    model = CoreModel(FEATURE_SIZE, HIDDEN_SIZE, NUM_CLASSES, MPNN_TYPE, 0.5, device).to(device)
    # model = ImbGNNplus(FEATURE_SIZE, HIDDEN_SIZE, NUM_CLASSES, 0.5).to(device)
    logs.info("Ours")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    start = time.time()
    for epoch in range(args.epoch):
      optimizer.zero_grad()
      output = model(data)
      if args.dataset == "DBLP":
        output = output[:4057]
      loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
      # logs.info("Epoch {}: {}".format(epoch, loss))
      loss.backward()
      optimizer.step()
    end = time.time()
    tt_our.append(end - start)

    model.eval()
    output = model(data)
    if args.dataset == "DBLP":
      output = output[:4057]
    pred = output.argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(acc) / int(data.test_mask.sum())
    logs.info("Model Acc: %.4f" % acc)
    # acc = (pred[data.high_mask] == data.y[data.high_mask]).sum()
    # acc = int(acc) / int(data.high_mask.sum())
    # logs.info("High Acc: %.4f" % acc)
    # acc = (pred[data.low_mask] == data.y[data.low_mask]).sum()
    # acc = int(acc) / int(data.low_mask.sum())
    # logs.info("Low Acc: %.4f" % acc)
    for i, label in enumerate(data.label_mask):
      acc = (pred[label] == data.y[label]).sum()
      acc = int(acc) / int(label.sum())
      logs.info("Class %d Acc: %.4f" % (i, acc))
      class_our_acc[i].append(acc)
    logs.info("AUC_ROC: %.4f" % AUC_ROC(output.cpu(), data.y.cpu()))
    logs.info("F1: %.4f" % F1(pred.cpu(), data.y.cpu()))

    # t-SNE
    X = tsne.fit_transform(output.detach().cpu().numpy())
    Y = np.unique(data.y.cpu())
    plt.figure()
    for label in Y:
      mask = (data.y.cpu() == label)
      plt.scatter(X[mask, 0], X[mask, 1], c=colors[label], label="Label", s=50)
    # plt.legend()
    # plt.show()
    plt.savefig(".\\pic\\tsne\\our_{}_{}.png".format(args.dataset, args.mpnn_type))

  # time
  tt_all = np.mean(tt)
  tt_per = tt_all / 200
  our_all = np.mean(tt_our)
  our_per = our_all / 200
  logs.info("Baseline: %.4lf, %.4lf" % (tt_all, tt_per))
  logs.info("Ours: %.4lf, %.4lf" % (our_all, our_per))

  # for i in range(NUM_CLASSES):
  #   acc = np.mean(class_acc[i])
  #   acc_our = np.mean(class_our_acc[i])
  #   logs.info("Class %d Acc: %.4lf\t%.4lf" % (i, acc, acc_our))