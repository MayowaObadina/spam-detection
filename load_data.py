from datasets import load_dataset

ds_train = load_dataset("SetFit/enron_spam", split='train')
ds_test = load_dataset("SetFit/enron_spam", split='test')

ds_train.to_csv("train.csv")
ds_test.to_csv("test.csv")