from train_func import get_train_args_T, get_train_args_TA, get_train_args_TAH, \
    setup_and_train_T, setup_and_train_TA, setup_and_train_TAH
from test_func import get_test_args_T, get_test_args_TA, get_test_args_TAH, get_test_args_real, \
    setup_and_test_T, setup_and_test_TA, setup_and_test_TAH, setup_and_test_real

if __name__ == "__main__":
    dataset_names = ["Rain100H", "Rain100L", "SPA", "Rain-Haze"]
    for dataset_name in dataset_names:
        # training and testing TLNet
        setup_and_train_T(get_train_args_T(dataset_name))
        setup_and_test_T(get_test_args_T(dataset_name))

        # training and testing TLNet and ALNet
        setup_and_train_TA(get_train_args_TA(dataset_name))
        setup_and_test_TA(get_test_args_TA(dataset_name))

        # training and testing DSDNet
        setup_and_train_TAH(get_train_args_TAH(dataset_name))
        setup_and_test_TAH(get_test_args_TAH(dataset_name))

    # testing on the real samples
    # setup_and_test_real(get_test_args_real())
