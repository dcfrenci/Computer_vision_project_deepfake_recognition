import dataset_handler
import ensemble_handler
import helpers
import trasformations
from dc import resnet_18
from matte import clip_fc
from simo import frequency


def main():
    # training(num_epochs=15)
    test_loader = dataset_handler.dataset_results(num_test_examples=1, batch_size=32)
    ret = resnet_18.resnet_get_features(test_loader)
    print(ret)
    # results()


def training(num_epochs):
    train_loader, test_loader = dataset_handler.dataset_handler(num_train_examples=256,
                                                                num_test_examples=64,
                                                                batch_size=32)

    helpers.print_title("TRAINING")
    # Resnet18
    out_resnet = resnet_18.resnet_handler(train_loader, test_loader, num_epochs)
    # Clip
    out_clip = clip_fc.clip_handler(train_loader, test_loader, num_epochs)
    # Frequency
    out_frequency = frequency.frequency_handler(train_loader, test_loader, num_epochs)
    # Ensemble
    ensemble_handler.ensemble_handler([out_resnet, out_clip, out_frequency], test_loader)

    helpers.print_title("FEATURE EXTRACTION")
    # Resnet18
    train_feature_resnet = resnet_18.resnet_get_features(train_loader)
    test_feature_resnet = resnet_18.resnet_get_features(test_loader)
    # Clip
    train_feature_clip = clip_fc.clip_fc_results(train_loader)
    test_feature_clip = clip_fc.clip_fc_results(test_loader)
    # Frequency
    train_feature_frequency = frequency.frequency_results(train_loader)
    test_feature_frequency = frequency.frequency_results(test_loader)
    # Ensemble
    train_feature = [train_feature_resnet, train_feature_clip, train_feature_frequency]
    test_feature = [test_feature_resnet, test_feature_clip, test_feature_frequency]
    ensemble_handler.ensemble_meta_model(train_feature, train_loader, test_feature, test_loader, num_epochs=15, batch_size=32)

    helpers.print_title("END TRAINING")


def results():
    test_loader = dataset_handler.dataset_results(num_test_examples=256, batch_size=32)
    helpers.print_title("RESULTS")

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
    ensemble_handler.ensemble_majority_voting([out_resnet, out_clip, out_frequency], test_loader)

    helpers.print_title("WITH TRANSFORMATION")
    # Resnet18
    out_resnet = resnet_18.resnet_results(test_loader_modified)
    # Clip
    out_clip = clip_fc.clip_fc_results(test_loader_modified)
    # Frequency
    out_frequency = frequency.frequency_results(test_loader_modified)
    # Ensemble
    ensemble_handler.ensemble_results([out_resnet, out_clip, out_frequency], test_loader_modified)
    ensemble_handler.ensemble_majority_voting([out_resnet, out_clip, out_frequency], test_loader_modified)



    helpers.print_title("END RESULTS")


if __name__ == "__main__":
    main()
