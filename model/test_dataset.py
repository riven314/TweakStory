from trainer import IGCaptionDataset

if __name__ == "__main__":
    data = IGCaptionDataset()

    for i in range(10):
        print(i)
        print(data[i])
