from generate_features import generate_filters
from scipy.signal import convolve
from scipy.signal import hilbert
from scipy.signal import correlate2d
from scipy.signal import correlate
from scipy.special import softmax
import numpy as np
import torch


class DynamicLayer(torch.nn.Module):
    '''
    Pytorch cnn network
    '''
    def __init__(self, kernel1_size, out_groups, kernel2_size, pool_size):
        super(DynamicLayer, self).__init__()

        # -----------------------------------------
        # Input size computation
        dataset = MyDataset('signals-2D')
        x, y = dataset[0]
        print('input: ', x.shape)

        # -----------------------------------------
        # CNN 1
        cnn1 = torch.nn.Conv2d(
            in_channels=x.shape[0],
            out_channels=out_groups,
            kernel_size=(x.shape[1], kernel1_size)
        )
        y1_exp = x.shape[2] - kernel1_size + 1
        print('cnn1 out: ', y1_exp)

        # -----------------------------------------
        # Flatten 1
        f1 = torch.nn.Flatten(2)

        # -----------------------------------------
        # CNN 2
        cnn2 = torch.nn.Conv1d(
            in_channels=out_groups,
            out_channels=1,
            kernel_size=kernel2_size
        )
        y2a_exp = y1_exp - kernel2_size + 1
        print('cnn2 out: ', y2a_exp)

        # -----------------------------------------
        # Flatten 2
        f2 = torch.nn.Flatten(1)


        # -----------------------------------------
        # Average Pooling
        p1 = torch.nn.AvgPool1d(kernel_size=pool_size)
        y3_exp = y2a_exp // pool_size

        print('nn input / pool out: ', y3_exp)
        self.output_size = y3_exp


        # -----------------------------------------
        # Rectifier 1
        r1 = torch.nn.LeakyReLU()

        self.stack = torch.nn.Sequential(
            cnn1,
            f1,
            cnn2,
            f2,
            p1,
            r1,
        )

    def forward(self, x):
        return self.stack(x)


class StaticLayer(torch.nn.Module):
    '''
    Pytorch nn feed forward
    '''
    def __init__(self, input_size, hidden_layers):
        super(StaticLayer, self).__init__()

        n1 = torch.nn.Linear(
            input_size,
            hidden_layers,
        )
        r2 = torch.nn.LeakyReLU()

        n2 = torch.nn.Linear(
            hidden_layers,
            len(label2num),
        )

        self.stack = torch.nn.Sequential(
            n1,
            r2,
            n2,
        )


    def forward(self, x):
        return self.stack(x)


class MyCNN(torch.nn.Module):
    '''
    Combine static and dynamic layers
    '''
    def __init__(self, kernel1_size, out_groups, kernel2_size, pool_size, hidden_layers):
        super(MyCNN, self).__init__()

        self.dynamic = DynamicLayer(
            kernel1_size,
            out_groups,
            kernel2_size,
            pool_size,
        )

        self.static = StaticLayer(
            self.dynamic.output_size,
            hidden_layers,
        )

        self.stack = torch.nn.Sequential(
            self.dynamic,
            self.static,
        )

    def forward(self, x):
        return self.stack(x)


class RealTimeConv2D(object):
    '''
    Realtime 2D CNN implementation
    '''
    def __init__(self, torch_cnn2d, omit_warmup=False):
        with torch.no_grad():
            self.kernel = torch_cnn2d.weight.squeeze(0).numpy()
            self.bias   = torch_cnn2d.bias.squeeze(0).item()

            self.kernel_len = self.kernel.shape[2]
            self.kernel_bin = self.kernel.shape[1]
            self.kernel_chn = self.kernel.shape[0]

            self.cache_len = self.kernel.shape[2] - 1
            self.cache = np.zeros(
                (self.kernel.shape[1], self.cache_len)
            )
            self.omit_warmup = omit_warmup

    def __prepend_cache(self, data):
        new_data = np.zeros(
            (data.shape[0], data.shape[1], data.shape[2] + self.cache_len)
        )

        new_data[:, :, :self.cache_len] = self.cache
        new_data[:, :, self.cache_len:] = data
        return new_data

    def convolve(self, data):
        if not self.omit_warmup:
            data = self.__prepend_cache(data)

        self.omit_warmup = False

        data_chan = data.shape[0]
        data_freq = data.shape[1]
        data_time = data.shape[2]

        exp_len = data_time - self.kernel_len + 1

        out = np.zeros((1, exp_len))

        for chan in range(data_chan):
            out += correlate2d(
                data[chan],
                self.kernel[chan],
                mode='valid'
            )

        out += self.bias

        self.cache = data[:, :, -self.cache_len:]

        return out.squeeze(0)


class RealTimeConv1D(object):
    '''
    Realtime 1D CNN implementation
    '''
    def __init__(self, torch_cnn1d, omit_warmup=False):
        with torch.no_grad():
            self.kernel = torch_cnn1d.weight.flatten(0).numpy()
            self.bias   = torch_cnn1d.bias.item()
            self.kernel_len = self.kernel.shape[0]
            self.cache  = np.zeros(self.kernel_len - 1)
            self.cache_len = self.kernel_len - 1
            self.omit_warmup = omit_warmup

    def __prepend_cache(self, data):
        new_data = np.zeros(len(data) + self.cache_len)
        new_data[:self.cache_len] = self.cache
        new_data[self.cache_len:] = data
        return new_data

    def __update_cache(self, data):
        self.cache = data[-self.cache_len:]

    def convolve(self, data):

        if not self.omit_warmup:
            data = self.__prepend_cache(data)

        self.omit_warmup = False

        out = correlate(data, self.kernel, mode='valid') + self.bias

        self.__update_cache(data)

        return out


class RealTimeAvgPool(object):
    '''
    Realtime avg pooling
    '''
    def __init__(self, num):
        self.num = num
        self.cache = None
        self.kernel = np.matrix(np.ones((1, num))/num)

    def __format_and_cache(self, data):
        if self.cache is not None:
            data = np.concatenate((self.cache, data))

        remainder = len(data) % self.num
        if 0 < remainder:
            self.cache = data[-remainder:]
            return data[:-remainder]
        else:
            self.cache = None
            return data

    def downsample(self, data):
        data = self.__format_and_cache(data)
        return (self.kernel @ np.matrix(data.reshape(-1, self.num).T)).A1


class RealTimeReLU(object):
    '''
    Realtime relu
    '''
    def __init__(self, alpha=0.01):
        self.alpha = np.abs(alpha)
        self.relu = np.vectorize(lambda x: x if x >= 0 else self.alpha * x)

    def rectify(self, x):
        print('relu in shape: ', x.shape)
        return self.relu(x)


class RealTimeNN(object):
    '''
    Realtime neural network
    '''
    def __init__(self, torch_nn):
        self.nn = torch_nn
        self.kernel_size = torch_nn.stack[0].in_features
        self.cache_len = self.kernel_size - 1
        self.cache = np.zeros(self.cache_len)
        self.relu = np.vectorize(lambda x: x if x >= 0 else -0.01 * x)

        with torch.no_grad():
            self.n1_weig = np.matrix(torch_nn.stack[0].weight.numpy())
            self.n1_bias = torch_nn.stack[0].bias.numpy().reshape(-1, 1)
            self.n2_weig = np.matrix(torch_nn.stack[2].weight.numpy())
            self.n2_bias = torch_nn.stack[2].bias.numpy().reshape(-1, 1)


    def evaluate(self, data):
        data = np.matrix(data.reshape(self.kernel_size, 1))

        n1a_out = (self.n1_weig @ data) + self.n1_bias

        n1b_out = self.relu(n1a_out)

        n2_out = ((self.n2_weig @ n1b_out) + self.n2_bias).A1

        return n2_out


    def __prepend_cache(self, data):
        new_data = np.zeros(len(data) + self.cache_len)
        new_data[:self.cache_len] = self.cache
        new_data[self.cache_len:] = data
        return new_data


    def __update_cache(self, data):
        self.cache = data[-self.cache_len:]


    def predict(self, data):
        print('nn input shape: ', data.shape)

        data = self.__prepend_cache(data)

        print('nn input shape after cache: ', data.shape)

        star_ind = 0
        stop_ind = star_ind + self.kernel_size

        predictions = []

        while stop_ind <= len(data):
            p = self.evaluate(data[star_ind:stop_ind])
            predictions.append(softmax(p))

            star_ind +=1
            stop_ind +=1


        self.__update_cache(data)

        res = np.array(predictions)

        print('nn output shape: ', res.shape)

        return res


class RealTimeCNN(object):
    '''
    Combine all the realtime objects for the full detector
    '''
    def __init__(self, filename='cnn11'):
        print('loading: ', filename)
        load_cnn = torch.load(filename)

        self.cnn2d = RealTimeConv2D(load_cnn.dynamic.stack[0])
        self.cnn1d = RealTimeConv1D(load_cnn.dynamic.stack[2])
        self.avgpl = RealTimeAvgPool(8)
        self.reclu = RealTimeReLU()
        self.nn = RealTimeNN(load_cnn.static)

    def predict(self, data):
        return self.nn.predict(
            self.reclu.rectify(
                self.avgpl.downsample(
                    self.cnn1d.convolve(
                        self.cnn2d.convolve(
                            data
                        )
                    )
                )
            )
        )



class FIRFilter(object):
    def __init__(self, taps):
        self.kernel = taps
        self.kernel_len = self.kernel.shape[0]
        self.cache_len = self.kernel_len - 1
        self.cache  = np.zeros(self.cache_len)


    def __prepend_cache(self, data):
        new_data = np.zeros(len(data) + self.cache_len)
        new_data[:self.cache_len] = self.cache
        new_data[self.cache_len:] = data
        return new_data


    def __update_cache(self, data):
        self.cache = data[-self.cache_len:]


    def convolve(self, data):
        data = self.__prepend_cache(data)

        out = convolve(
            data,
            self.kernel,
            mode='valid'
        )

        self.__update_cache(data)

        return out


class FeatureGenerator(object):
    def __init__(self, fs, downsample_factor, bw=5, num_banks=12):
        self.fs_prime = fs / downsample_factor
        self.kernel_size = 128
        self.num_banks = num_banks

        filters, fcenters = generate_filters(
            bw=bw,
            num=self.num_banks,
            fs=self.fs_prime,
            numtaps=self.kernel_size
        )

        self.filter_banks = [
            FIRFilter(f)
            for f in filters
        ]


    def convolve(self, signal):
        exp_len = len(signal)

        data = np.zeros((self.num_banks, exp_len))

        for k, f in enumerate(self.filter_banks):
            ret = f.convolve(signal)

            ret = np.abs(hilbert(ret))

            data[k,:] = ret

        return data
