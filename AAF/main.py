from train_bart import Fusion_augmentaton_Training
# from train_bart_SAMSUM import Fusion_augmentaton_Training
import os

# initialization
adjustable_para = 0
stop_threshold = 0
if os.path.exists("./metrics.txt"):
    os.remove("./metrics.txt")
for num in range(1, 15):
    # augmrntation
    adjustable_para, Stop_symbol, stop_threshold = Fusion_augmentaton_Training(num, adjustable_para, stop_threshold)
    if Stop_symbol == 1 and stop_threshold == 3:
        break
    print(adjustable_para)
