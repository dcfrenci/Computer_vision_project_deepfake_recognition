import dataset_handler
import ensemble_handler
import helpers
import trasformations
from dc import resnet_18
from matte import clip_fc
from simo import frequency


def main():
    # training(num_epochs=15)

    results()


def training(num_epochs):
    train_loader, test_loader = dataset_handler.dataset_handler(num_train_examples=256,
                                                                num_test_examples=64,
                                                                batch_size=32)

    helpers.print_section("TRAINING")
    # Resnet18
    out_resnet = resnet_18.resnet_handler(train_loader, test_loader, num_epochs)
    # Clip
    out_clip = clip_fc.clip_handler(train_loader, test_loader, num_epochs)
    # Frequency
    out_frequency = frequency.frequency_handler(train_loader, test_loader, num_epochs)
    # Ensemble
    ensemble_handler.ensemble_handler([out_resnet, out_clip, out_frequency], test_loader)

    helpers.print_section("END TRAINING")


def results():
    test_loader = dataset_handler.dataset_results(num_test_examples=256, batch_size=32)
    helpers.print_section("RESULTS")

    # apply random transformations on the training set
    # dis-comment if you want to test transformations effect on the results
    test_loader_modified = trasformations.apply_transformation(test_loader)

    # Resnet18
    out_resnet = resnet_18.resnet_results(test_loader)
    # Clip
    out_clip = clip_fc.clip_fc_results(test_loader)
    # Frequency
    out_frequency = frequency.frequency_results(test_loader)
    # Ensemble
    ensemble_handler.ensemble_results([out_resnet, out_clip, out_frequency], test_loader)

    out_resnet = resnet_18.resnet_results(test_loader_modified)
    # Clip
    out_clip = clip_fc.clip_fc_results(test_loader_modified)
    # Frequency
    out_frequency = frequency.frequency_results(test_loader_modified)
    # Ensemble
    ensemble_handler.ensemble_results([out_resnet, out_clip, out_frequency], test_loader_modified)

    helpers.print_section("END RESULTS")


if __name__ == "__main__":
    main()
