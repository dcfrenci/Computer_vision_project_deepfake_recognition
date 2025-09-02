import dataset_handler
from dc import resnet_18
from matte import clip_fc
from simo import frequency


def main():
    # Dataset
    train_loader, test_loader = dataset_handler.dataset_handler()

    # Resnet18
    resnet_18.resnet_handler(train_loader, test_loader)

    # Clip
    clip_fc.clip_handler(train_loader, test_loader)

    # Frequency
    frequency.frequency_handler()


if __name__ == "__main__":
    main()
