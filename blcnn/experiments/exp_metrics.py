import os
import sys
from pathlib import Path

from tqdm import tqdm


def main() -> None:

    # EXPERIMENT 2.1 & 2.2
    # Specify which experiment to compute metrics for
    folder = Path('models/experiment_fixed_lr_trained_models')
    # folder = Path('models/experiment_opt_lr_trained_models')

    # Dicts to globally collect metrics
    all_metrics = {}
    all_metrics_folded = {}
    folder = Path('models/experiment_fixed_lr_trained_models')
    for config_folder in tqdm(folder.iterdir(), desc='Evaluating fixed LR models', total=45):
        if config_folder.is_dir():
            if 'logs' in config_folder.name:
                continue
            metrics, metric_folded = compute_metrics(config_folder/ 'predictions.csv')
            key = config_folder.name if '34_' not in config_folder.name else f'{config_folder.name[3:]}_dense'
            all_metrics[key] = metrics
            all_metrics_folded[key] = metric_folded


    # EXPERIMENT 1
    # folder = Path('data/output_experiment_alpha')
    # for path_to_predictions in folder.glob('*.csv'):
    #     metrics, metric_folded = compute_metrics(path_to_predictions)
    #     key = path_to_predictions.stem
    #     all_metrics[key] = metrics
    #     all_metrics_folded[key] = metric_folded

    def print_metrics_table(all_metrics_dict, title: str, sorted_flag=True):
        print(f'\n{title}')
        headers = ['Configuration', 'Sparse Cat. Acc.', 'Elevation Gain', 'Mean Angular Error']
        print(f'{headers[0]:<20} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10}')

        # Sort by nr of layers in configuration and then alphabetically
        if sorted_flag:
            for config_name, metrics in sorted(all_metrics_dict.items(), key=lambda x: (
            x[0].split('_')[-1] == 'dense', len(x[0].split('_')), int(x[0].split('_')[0]))):
                print(f'{config_name:<20} {metrics["sparse_categorical_accuracy"]:<10.4f} '
                      f'{metrics["elevation_gain"]:<10.4f} {metrics["mean_angular_error"]:<10.2f}')
        else:
            for config_name, metrics in all_metrics_dict.items():
                print(f'{config_name:<20} {metrics["sparse_categorical_accuracy"]:<10.4f} '
                      f'{metrics["elevation_gain"]:<10.4f} {metrics["mean_angular_error"]:<10.2f}')

    print_metrics_table(all_metrics, title='Metrics (Unfolded)', sorted_flag=True)
    print_metrics_table(all_metrics_folded, title='Metrics (Folded)', sorted_flag=True)


def load_model_and_predict(config_folder):
    import csv
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    from tqdm import tqdm

    only_cpu = True
    if only_cpu:
        # Force CPU inference for reproducibility.
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass

    def single_example_parser(example):
        feature_description = {
            'train/image': tf.io.FixedLenFeature([], tf.string),
            'train/target': tf.io.FixedLenFeature([], tf.int64)
        }
        example = tf.io.parse_single_example(example, feature_description)
        example['train/image'] = tf.reshape(tf.io.decode_raw(example['train/image'], tf.float32), (39, 8000, 2))
        return example['train/image'], example['train/target']

    def predict_with_ground_truth(model, dataset):
        for batch in dataset:
            inputs, labels = batch
            predictions = model(inputs, training=False)
            yield predictions, labels

    print(f'\nEvaluating experiment for configuration: {config_folder.parent.name}')
    model: keras.Model = keras.models.load_model(config_folder.as_posix(), compile=False)

    model.compile(optimizer=keras.optimizers.legacy.Adam(1e-3), loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    path_to_cochleagrams = Path('data/cochleagrams/naturalsounds165_hrtf_nh2_20')

    dataset = (
        tf.data.TFRecordDataset(path_to_cochleagrams / 'test_cochleagrams.tfrecord', compression_type="GZIP")
        .map(lambda serialized_example: single_example_parser(serialized_example))
        .batch(16, drop_remainder=True)
        .prefetch(1)
    )

    # Predict
    true_classes = []
    pred_classes = []
    for predictions, labels in tqdm(predict_with_ground_truth(model, dataset), unit='batches', total=206):
        true_classes.append(labels.numpy())
        pred_classes.append(predictions.numpy())

    true_classes = np.concatenate(true_classes, axis=0)
    pred_classes = np.concatenate(pred_classes, axis=0).argmax(axis=1)

    # write to CSV
    with open(f'{config_folder.parent}/predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true_class', 'pred_class'])
        writer.writerows(zip(true_classes, pred_classes))


def compute_metrics(config_folder):
    import csv
    import numpy as np
    import scipy

    def read_single_cnn_result(path: Path, data_selection: str, folded: bool):
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            l = []
            for row in reader:
                true_class_loc = CNNpos_to_loc_extra(int(row['true_class']), data_selection=data_selection,
                                                     folded=folded)
                if true_class_loc is None:  # If the true class is not in the desired area, skip this row
                    continue
                # For the predictions we don't want to filter out any values
                pred_class_loc = CNNpos_to_loc_extra(int(row['pred_class']), data_selection='all', folded=folded)
                l.append([*true_class_loc, *pred_class_loc])
            return np.array(l)

    def CNNpos_to_loc_extra(CNN_pos, data_selection='all', folded=False, bin_size=5):
        """
        convert bin label in the CNN from Francl 2022 into [azim, elev] positions
        :param CNN_pos: int, [0, 503]
        :param bin_size: int, degree. note that elevation bin size is 2*bin_size
        :return: tuple, (azi, ele)
        """
        n_azim = int(360 / bin_size)  # bin_size=5 -> 72
        div, mod = divmod(CNN_pos, n_azim)
        azim = bin_size * mod
        if azim >= 180:
            azim -= 360

        # If we only want front or back data, return None if the true class is not in the desired area
        if data_selection == 'front' and not -90 < azim < 90:
            return None
        elif data_selection == 'back' and -90 < azim < 90:
            return None

        # Fold back to front
        if folded:
            if azim > 90:
                azim = 180 - azim
            elif azim < -90:
                azim = -180 - azim

        elev = bin_size * div * 2
        if elev > 60:
            print('Elev > 60!', CNN_pos, azim, elev)
        # return elev, azim # wrong, test
        return azim, elev

    data = read_single_cnn_result(config_folder, data_selection='all', folded=False)
    data_folded = read_single_cnn_result(config_folder, data_selection='all', folded=True)

    def compute_metrics_for_data(data: np.ndarray):
        targets = data[:, :2]  # [az, ele], first two columns
        responses = data[:, 2:]  # [az, ele], last two columns

        # sparse categorical accuracy
        target_ids = np.array([int((elev // 10) * 72 + ((azim % 360) // 5)) for azim, elev in targets])
        response_ids = np.array(
            [int((elev_pred // 10) * 72 + ((azim_pred % 360) // 5)) for azim_pred, elev_pred in responses])
        sparse_categorical_accuracy = np.mean(np.equal(target_ids, response_ids))

        # Elevation gain
        elevation_gain, n = scipy.stats.linregress(targets[:, 1], responses[:, 1])[:2]

        # Mean angular error
        # Convert degrees -> radians
        az_t = np.deg2rad(targets[:, 0])
        el_t = np.deg2rad(targets[:, 1])
        az_r = np.deg2rad(responses[:, 0])
        el_r = np.deg2rad(responses[:, 1])

        # Convert spherical (azimuth, elevation) to 3D unit vectors
        # Using elevation measured from the horizontal plane:
        # x = cos(el)*cos(az), y = cos(el)*sin(az), z = sin(el)
        xt = np.cos(el_t) * np.cos(az_t)
        yt = np.cos(el_t) * np.sin(az_t)
        zt = np.sin(el_t)

        xr = np.cos(el_r) * np.cos(az_r)
        yr = np.cos(el_r) * np.sin(az_r)
        zr = np.sin(el_r)

        # Dot product between corresponding unit vectors gives cos(angle)
        dots = xt * xr + yt * yr + zt * zr

        # Numerical safety: clamp to valid arccos range
        dots = np.clip(dots, -1.0, 1.0)

        # Angular error per sample (radians), then mean (degrees)
        ang_rad = np.arccos(dots)
        mean_angular_error = float(np.rad2deg(np.mean(ang_rad)))

        return {
            'sparse_categorical_accuracy': sparse_categorical_accuracy,
            'elevation_gain': elevation_gain,
            'mean_angular_error': mean_angular_error
        }

    metrics = compute_metrics_for_data(data)
    metrics_folded = compute_metrics_for_data(data_folded)
    return metrics, metrics_folded


if __name__ == "__main__":
    main()
