import dataset_handler
from ensemble_model import ensemble_handler
import helpers
import trasformations
from resnet_model import resnet_18
from clip_model import clip_fc
from xception_model import frequency


def main():
    # training(num_epochs=15)
    results()
    #importance_map_show()


def training(num_epochs):
    train_loader, test_loader = dataset_handler.dataset_handler(num_train_examples=256,
                                                                num_test_examples=64,
                                                                batch_size=32)

    helpers.print_title("TRAINING")
    # Resnet18
    out_resnet = resnet_18.resnet_handler(train_loader, test_loader, num_epochs, "resnet_model/resnet_18_weight.pth")
    # Clip
    out_clip = clip_fc.clip_handler(train_loader, test_loader, num_epochs, "clip_model/fc_layer_weight.pth")
    # Frequency
    out_frequency = frequency.frequency_handler(train_loader, test_loader, num_epochs,
                                                "xception_model/frequency_Xception_weight.pth")
    # Ensemble
    ensemble_handler.ensemble_handler([out_resnet, out_clip, out_frequency], test_loader)

    helpers.print_title("FEATURE EXTRACTION")
    # Resnet18
    train_feature_resnet = resnet_18.resnet_get_features(train_loader, "resnet_model/resnet_18_weight.pth")
    test_feature_resnet = resnet_18.resnet_get_features(test_loader, "resnet_model/resnet_18_weight.pth")
    # Clip
    train_feature_clip = clip_fc.clip_fc_get_features(train_loader)
    test_feature_clip = clip_fc.clip_fc_get_features(test_loader)
    # Frequency
    train_feature_frequency = frequency.xception_feature_extractor(train_loader,
                                                                   "xception_model/frequency_Xception_weight.pth")
    test_feature_frequency = frequency.xception_feature_extractor(test_loader,
                                                                  "xception_model/frequency_Xception_weight.pth")
    # Ensemble
    train_feature = [train_feature_resnet, train_feature_clip, train_feature_frequency]
    test_feature = [test_feature_resnet, test_feature_clip, test_feature_frequency]
    ensemble_handler.ensemble_meta_model(train_feature, train_loader, test_feature, test_loader, num_epochs=15, batch_size=32)

    # helpers.print_title("TRAINING TRANSFORMATION")
    train_loader_modified = trasformations.apply_transformation(train_loader)
    test_loader_modified = trasformations.apply_transformation(test_loader)
    # Resnet18
    out_resnet = resnet_18.resnet_handler(train_loader_modified, test_loader_modified, num_epochs,
                                          "resnet_model/resnet_18_weight_tr.pth")
    # Clip
    out_clip = clip_fc.clip_handler(train_loader_modified, test_loader_modified, num_epochs,
                                    "clip_model/fc_layer_weight_tr.pth")
    # Frequency
    out_frequency = frequency.frequency_handler(train_loader_modified, test_loader_modified, num_epochs,
                                                "xception_model/frequency_Xception_weight_tr.pth")
    # Ensemble
    ensemble_handler.ensemble_handler([out_resnet, out_clip, out_frequency], test_loader_modified)

    helpers.print_title("FEATURE EXTRACTION TRANSFORMATION")
    # Resnet18
    train_feature_resnet = resnet_18.resnet_get_features(train_loader_modified, "resnet_model/resnet_18_weight_tr.pth")
    test_feature_resnet = resnet_18.resnet_get_features(test_loader_modified, "resnet_model/resnet_18_weight_tr.pth")
    # Clip
    train_feature_clip = clip_fc.clip_fc_get_features(train_loader_modified)
    test_feature_clip = clip_fc.clip_fc_get_features(test_loader_modified)
    # Frequency
    train_feature_frequency = frequency.xception_feature_extractor(train_loader_modified,
                                                                   "xception_model/frequency_Xception_weight_tr.pth")
    test_feature_frequency = frequency.xception_feature_extractor(test_loader_modified,
                                                                  "xception_model/frequency_Xception_weight_tr.pth")
    # Ensemble
    train_feature = [train_feature_resnet, train_feature_clip, train_feature_frequency]
    test_feature = [test_feature_resnet, test_feature_clip, test_feature_frequency]
    ensemble_handler.ensemble_meta_model(train_feature, train_loader_modified, test_feature, test_loader_modified, num_epochs=num_epochs, batch_size=32)

    helpers.print_title("END TRAINING")


def results():
    test_loader = dataset_handler.dataset_results(num_test_examples=256, batch_size=32)
    helpers.print_title("RESULTS")

    # Resnet18
    out_resnet = resnet_18.resnet_results(test_loader, "resnet_model/resnet_18_weight.pth")
    test_feature_resnet = resnet_18.resnet_get_features(test_loader, "resnet_model/resnet_18_weight_tr.pth")
    # Clip
    out_clip = clip_fc.clip_fc_results(test_loader, "clip_model/fc_layer_weight.pth")
    test_feature_clip = clip_fc.clip_fc_get_features(test_loader)
    # Frequency
    out_frequency = frequency.frequency_results(test_loader, "xception_model/frequency_Xception_weight.pth")
    test_feature_frequency = frequency.xception_feature_extractor(test_loader,
                                                                  "xception_model/frequency_Xception_weight_tr.pth")
    # Ensemble
    ensemble_handler.ensemble_majority_voting([out_resnet, out_clip, out_frequency], test_loader)
    ensemble_handler.ensemble_results([out_resnet, out_clip, out_frequency], test_loader)
    test_feature = [test_feature_resnet, test_feature_clip, test_feature_frequency]
    ensemble_handler.ensemble_meta_results(test_feature, test_loader, batch_size=32)

    helpers.print_title("WITH TRANSFORMATION")
    test_loader_modified = trasformations.apply_transformation(test_loader)
    # Resnet18
    out_resnet = resnet_18.resnet_results(test_loader_modified, "resnet_model/resnet_18_weight_tr.pth")
    test_feature_resnet = resnet_18.resnet_get_features(test_loader_modified, "resnet_model/resnet_18_weight_tr.pth")
    # Clip
    out_clip = clip_fc.clip_fc_results(test_loader_modified, "clip_model/fc_layer_weight_tr.pth")
    test_feature_clip = clip_fc.clip_fc_get_features(test_loader_modified)
    # Frequency
    out_frequency = frequency.frequency_results(test_loader_modified, "xception_model/frequency_Xception_weight_tr.pth")
    test_feature_frequency = frequency.xception_feature_extractor(test_loader_modified,
                                                                  "xception_model/frequency_Xception_weight_tr.pth")
    # Ensemble
    ensemble_handler.ensemble_majority_voting([out_resnet, out_clip, out_frequency], test_loader_modified)
    ensemble_handler.ensemble_results([out_resnet, out_clip, out_frequency], test_loader_modified)
    test_feature_tr = [test_feature_resnet, test_feature_clip, test_feature_frequency]
    ensemble_handler.ensemble_meta_results(test_feature_tr, test_loader_modified, batch_size=32)

    helpers.print_title("END RESULTS")

def importance_map_show():
    frequency.xception_heatmap_handler("xception_model/frequency_Xception_weight.pth")
    clip_fc.clip_heatmap_handler("clip_model/fc_layer_weight.pth")
    resnet_18.resnet_heatmap_handler("resnet_model/resnet_18_weight.pth")



if __name__ == "__main__":
    main()
