import dataset_handler
import ensemble_handler
import helpers
from dc import resnet_18
from matte import clip_fc
from simo import frequency


def main():
    # Dataset
    train_loader, test_loader = dataset_handler.dataset_handler(num_train_examples=256,
                                                                num_test_examples=64,
                                                                batch_size=32)
    training(train_loader, test_loader)


def training(train_loader, test_loader):
    helpers.print_section("TRAINING")

    # Resnet18
    out_resnet = resnet_18.resnet_handler(train_loader, test_loader)
    # Clip
    out_clip = clip_fc.clip_handler(train_loader, test_loader)
    # Frequency
    out_frequency = frequency.frequency_handler(train_loader, test_loader)
    # Ensemble
    ensemble_handler.ensemble_handler([out_resnet, out_clip], test_loader)

    helpers.print_section("END TRAINING")


def results(test_loader):
    helpers.print_section("RESULTS")

    # Resnet18
    out_resnet = resnet_18.resnet_results(test_loader)
    # Clip
    out_clip = clip_fc.clip_results(test_loader)
    # Frequency
    # frequency.frequency_results(test_loader)
    # Ensemble
    ensemble_handler.ensemble_results([out_resnet, out_clip], test_loader)

    helpers.print_section("END RESULTS")


if __name__ == "__main__":
    main()
