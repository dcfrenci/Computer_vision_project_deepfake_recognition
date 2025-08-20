import dataset_handler
from dc import resnet_18


def main():
    # Dataset
    train_loader, test_loader = dataset_handler.dataset_handler()

    # Resnet18
    resnet_18.resnet_handler(train_loader, test_loader)

    # Clip

    # Frequency


if __name__ == "__main__":
    main()
