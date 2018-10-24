import Dataset.port as port


if __name__ == '__main__':
    l1 = port.create_dataset_icdar2015('/home/wzw/ICDAR2015/train_img', None, None)
    print(l1)
