import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Add, Input, LeakyReLU, Conv3D, ConvLSTM2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import scipy.io
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skimage.metrics import structural_similarity as ssim


class WeightedAdd(Layer):
    def __init__(self, **kwargs):
        super(WeightedAdd, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=(1,), initializer="ones", trainable=True)
        super(WeightedAdd, self).build(input_shape)

    def call(self, inputs):
        return inputs[0] + self.kernel[0] * inputs[1]


def load_mat_data(file_path):
    data = scipy.io.loadmat(file_path)
    return data


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    ssim_value = ssim(y_true[-1], y_pred[-1], channel_axis=-1)
    return mse, rmse, mae, r2, ssim_value


def create_deep_fusion_model(input_shape):
    input1 = Input(shape=(None, *input_shape, 1), name='input_1')
    input2 = Input(shape=(None, *input_shape, 1), name='input_2')

    x1 = Conv3D(filters=10, kernel_size=(3, 3, 3), padding='same')(input1)
    x1 = LeakyReLU(alpha=0.1)(x1)
    x1 = Conv3D(filters=10, kernel_size=(3, 3, 3), padding='same')(x1)
    x1 = LeakyReLU(alpha=0.1)(x1)

    x2 = Conv3D(filters=10, kernel_size=(3, 3, 3), padding='same')(input2)
    x2 = LeakyReLU(alpha=0.1)(x2)
    x2 = Conv3D(filters=10, kernel_size=(3, 3, 3), padding='same')(x2)
    x2 = LeakyReLU(alpha=0.1)(x2)

    fusion = WeightedAdd()([x1, x2])

    x = ConvLSTM2D(filters=5, kernel_size=(3, 3), padding='same', return_sequences=True)(fusion)
    x = ConvLSTM2D(filters=5, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    output_conv3d = Conv3D(filters=1, kernel_size=(3, 3, 3), padding='same', activation='swish')(x)

    model = Model(inputs=[input1, input2], outputs=output_conv3d)
    return model


# Load data
x1_train_path = '/home/wangdong/trustworthy_AI/P1X1_train_norm.mat'
x2_train_path = '/home/wangdong/trustworthy_AI/P1X2_train_norm.mat'
y_train_path = '/home/wangdong/trustworthy_AI/P1Y_train_norm.mat'

X1 = load_mat_data(x1_train_path)['P1X1_train_norm']
X2 = load_mat_data(x2_train_path)['P1X2_train_norm']
Y = load_mat_data(y_train_path)['P1Y_train_norm']

print(f"Original X1 shape: {X1.shape}")
print(f"Original X2 shape: {X2.shape}")
print(f"Original Y shape: {Y.shape}")

# Adding channels and time sequence dimension to the data
X1 = np.expand_dims(X1, axis=-1)
X2 = np.expand_dims(X2, axis=-1)
Y = np.expand_dims(Y, axis=-1)
X1 = np.expand_dims(X1, axis=0)
X2 = np.expand_dims(X2, axis=0)
Y = np.expand_dims(Y, axis=0)

print(f"New X1 shape: {X1.shape}")
print(f"New X2 shape: {X2.shape}")
print(f"New Y shape: {Y.shape}")

input_shape = (250, 200)
time_seq = 120
batch_size = 1  # 调整批量大小
patch_size = (125, 100)  # 调整块大小


# Function to split data into patches
def split_into_patches(X, patch_size):
    patches = []
    print(f"Splitting data with shape: {X.shape}")
    for i in range(0, X.shape[2], patch_size[0]):
        for j in range(0, X.shape[3], patch_size[1]):
            patch = X[:, :, i:i+patch_size[0], j:j+patch_size[1], :]
            print(f"Patch shape: {patch.shape}")
            if patch.shape[2] == patch_size[0] and patch.shape[3] == patch_size[1]:
                patches.append(patch)
    return np.concatenate(patches, axis=0)


# Split the data into patches
X1_patches = split_into_patches(X1, patch_size)
X2_patches = split_into_patches(X2, patch_size)
Y_patches = split_into_patches(Y, patch_size)

print(f"X1_patches shape: {X1_patches.shape}")
print(f"X2_patches shape: {X2_patches.shape}")
print(f"Y_patches shape: {Y_patches.shape}")


# Function to run cross-validation
def run_k_fold_cv(X1, X2, Y, k):
    kf = KFold(n_splits=k)
    metrics = {'mse': [], 'rmse': [], 'mae': [], 'r2': [], 'ssim': []}

    for train_index, val_index in kf.split(X1):
        X1_train, X1_val = X1[train_index], X1[val_index]
        X2_train, X2_val = X2[train_index], X2[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]

        model = create_deep_fusion_model(patch_size)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.fit([X1_train, X2_train], Y_train, batch_size=batch_size, epochs=600, verbose=1)

        predictions = model.predict([X1_val, X2_val])
        mse, rmse, mae, r2, ssim_value = calculate_metrics(Y_val, predictions)

        metrics['mse'].append(mse)
        metrics['rmse'].append(rmse)
        metrics['mae'].append(mae)
        metrics['r2'].append(r2)
        metrics['ssim'].append(ssim_value)

    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return avg_metrics


# Running K-fold cross-validation for K=2, 3
k_values = [2,3,4]
all_metrics = {}

for k in k_values:
    metrics = run_k_fold_cv(X1_patches, X2_patches, Y_patches, k)
    all_metrics[k] = metrics

# Print the results
for k, metrics in all_metrics.items():
    print(f"Metrics for K={k}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print("\n")
