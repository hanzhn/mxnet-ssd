import matplotlib.pyplot as plt  
import numpy as np  
import re  
import argparse  
  
parser = argparse.ArgumentParser(description='Parses log file and generates train/val curves')  
parser.add_argument('--log-file', type=str,default="/home/smiles/hz/mxnet-ssd/train-ssd.log",  
                    help='the path of log file')  
args = parser.parse_args()  
  
  
TR1_RE = re.compile('.*?]\sTrain-CrossEntropy=([\d\.]+)')  
TR2_RE = re.compile('.*?]\sTrain-SmoothL1=([\d\.]+)') 
VA_RE = re.compile('.*?]\sValidation-mAP=([\d\.]+)')  
  
log = open(args.log_file).read()  
  
log_tr1 = [float(x) for x in TR1_RE.findall(log)]  
log_tr2 = [float(x) for x in TR2_RE.findall(log)] 
log_va = [float(x) for x in VA_RE.findall(log)]  
idx = np.arange(len(log_tr1))  
  
plt.figure(figsize=(8, 6))  
plt.xlabel("Epoch")  
plt.ylabel("Accuracy")  
log_tr = [log_tr1[i]+log_tr2[i] for i in range(len(log_tr1))]
plt.plot(idx, log_tr1, 'o', linestyle='-', color="r",  
         label="Train classify loss")  
plt.plot(idx, log_tr2, 'o', linestyle='-', color="g",  
         label="Train localization loss")
plt.plot(idx, log_va, 'o', linestyle='-', color="b",  
         label="Validation accuracy")  
  
plt.legend(loc="best")  
plt.xticks(np.arange(min(idx), max(idx)+1, 5))  
plt.yticks(np.arange(0, 1, 0.2))  
plt.ylim([0,1])  
plt.show()  
