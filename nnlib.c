#include "nnlib.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

static void im2col (const float *restrict input, float *restrict col,
                   int in_h, int in_w, int in_c,
                   int fh, int fw, int stride) {
    int out_h = (in_h - fh) / stride + 1;
    int out_w = (in_w - fw) / stride + 1;
    int col_cols = out_h * out_w;

    for (int c = 0; c < in_c; c++) {
        for (int fih = 0; fih < fh; fih++) {
            for (int fiw = 0; fiw < fw; fiw++) {
                int row = (c * fh + fih) * fw + fiw;
                int col_offset = row * col_cols;
                int in_offset = c * in_h * in_w;
                for (int oh = 0; oh < out_h; oh++) {
                    int ih = oh * stride + fih;
                    int base = col_offset + oh * out_w;
                    int in_base = in_offset + ih * in_w;
                    for (int ow = 0; ow < out_w; ow++) {
                        col[base + ow] = input[in_base + ow * stride + fiw];
                    }
                }
            }
        }
    }
}

static void col2im (const float *restrict col, float *restrict output,
                   int in_h, int in_w, int in_c,
                   int fh, int fw, int stride) {
    int out_h = (in_h - fh) / stride + 1;
    int out_w = (in_w - fw) / stride + 1;
    int col_cols = out_h * out_w;

    for (int c = 0; c < in_c; c++) {
        for (int fih = 0; fih < fh; fih++) {
            for (int fiw = 0; fiw < fw; fiw++) {
                int row = (c * fh + fih) * fw + fiw;
                int col_offset = row * col_cols;
                int out_offset = c * in_h * in_w;
                for (int oh = 0; oh < out_h; oh++) {
                    int ih = oh * stride + fih;
                    int base = col_offset + oh * out_w;
                    int out_base = out_offset + ih * in_w;
                    for (int ow = 0; ow < out_w; ow++) {
                        output[out_base + ow * stride + fiw] += col[base + ow];
                    }
                }
            }
        }
    }
}

neural_network_t* alloc_neural_network (void) {
    neural_network_t *nn = calloc(1, sizeof(neural_network_t));
    nn->layers = NULL;
    return nn;
}

void add_fc_layer (neural_network_t *nn, int in_size, int out_size) {
    nn->n_layers++;

    nn->layers = realloc(nn->layers, nn->n_layers * sizeof(layer_t));
    nn->layers[nn->n_layers - 1].type = LAYER_FC;
    nn->layers[nn->n_layers - 1].output_size = out_size;
    nn->layers[nn->n_layers - 1].delta = calloc(in_size, sizeof(float));
    nn->layers[nn->n_layers - 1].output = calloc(out_size, sizeof(float));

    nn->layers[nn->n_layers - 1].data.fc.in_size = in_size;
    nn->layers[nn->n_layers - 1].data.fc.out_size = out_size;
    nn->layers[nn->n_layers - 1].data.fc.weight = calloc(in_size * out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.bias = calloc(out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.m_weight = calloc(in_size * out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.v_weight = calloc(in_size * out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.m_bias = calloc(out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.v_bias = calloc(out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.grad_weight = calloc(in_size * out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.total_grad_weight = calloc(in_size * out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.grad_bias = calloc(out_size, sizeof(float));
    nn->layers[nn->n_layers - 1].data.fc.total_grad_bias = calloc(out_size, sizeof(float));
}

void add_conv_layer (neural_network_t *nn, int in_height, int in_width, int in_channel, int filter_height, int filter_width, int n_filters, int filter_stride, int n_padding) {
    nn->n_layers++;
    int out_height = (in_height - filter_height + 2*n_padding) / filter_stride + 1;
    int out_width  = (in_width  - filter_width  + 2*n_padding) / filter_stride + 1;

    nn->layers = realloc(nn->layers, nn->n_layers * sizeof(layer_t));
    nn->layers[nn->n_layers - 1].type = LAYER_CONV;
    nn->layers[nn->n_layers - 1].output_size = n_filters * out_height * out_width;
    nn->layers[nn->n_layers - 1].delta = calloc(in_channel * in_height * in_width, sizeof(float));
    nn->layers[nn->n_layers - 1].output = calloc(n_filters * out_height * out_width, sizeof(float));

    nn->layers[nn->n_layers - 1].data.conv.in_height = in_height;
    nn->layers[nn->n_layers - 1].data.conv.in_width = in_width;
    nn->layers[nn->n_layers - 1].data.conv.in_channel = in_channel;
    nn->layers[nn->n_layers - 1].data.conv.filter_height = filter_height;
    nn->layers[nn->n_layers - 1].data.conv.filter_width = filter_width;
    nn->layers[nn->n_layers - 1].data.conv.n_filters = n_filters;
    nn->layers[nn->n_layers - 1].data.conv.filter_stride = filter_stride;
    nn->layers[nn->n_layers - 1].data.conv.n_padding = n_padding;
    nn->layers[nn->n_layers - 1].data.conv.filter = calloc(in_channel * n_filters * filter_height * filter_width, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.bias = calloc(n_filters, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.m_filter = calloc(in_channel * n_filters * filter_height * filter_width, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.v_filter = calloc(in_channel * n_filters * filter_height * filter_width, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.grad_filter = calloc(in_channel * n_filters * filter_height * filter_width, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.total_grad_filter = calloc(in_channel * n_filters * filter_height * filter_width, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.m_bias = calloc(n_filters, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.v_bias = calloc(n_filters, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.grad_bias = calloc(n_filters, sizeof(float));
    nn->layers[nn->n_layers - 1].data.conv.total_grad_bias = calloc(n_filters, sizeof(float));
}

void add_pool_layer (neural_network_t *nn, int in_height, int in_width, int in_channel, int kernel_height, int kernel_width) {
    nn->n_layers++;
    int out_height = in_height / kernel_height;
    int out_width = in_width / kernel_width;

    nn->layers = realloc(nn->layers, nn->n_layers * sizeof(layer_t));
    nn->layers[nn->n_layers - 1].type = LAYER_POOL;
    nn->layers[nn->n_layers - 1].output_size = in_channel * out_height * out_width;
    nn->layers[nn->n_layers - 1].delta = calloc(in_channel * in_height * in_width, sizeof(float));
    nn->layers[nn->n_layers - 1].output = calloc(in_channel * out_height * out_width, sizeof(float));

    nn->layers[nn->n_layers - 1].data.pool.in_height = in_height;
    nn->layers[nn->n_layers - 1].data.pool.in_width = in_width;
    nn->layers[nn->n_layers - 1].data.pool.in_channel = in_channel;
    nn->layers[nn->n_layers - 1].data.pool.kernel_height = kernel_height;
    nn->layers[nn->n_layers - 1].data.pool.kernel_width = kernel_width;
    nn->layers[nn->n_layers - 1].data.pool.mask = calloc(in_channel * in_height * in_width, sizeof(uint8_t));
}

void add_activation_layer (neural_network_t *nn, layer_type_t activation) {
    nn->n_layers++;

    nn->layers = realloc(nn->layers, nn->n_layers * sizeof(layer_t));
    nn->layers[nn->n_layers - 1].type = activation;
    nn->layers[nn->n_layers - 1].output_size = nn->layers[nn->n_layers - 2].output_size;
    nn->layers[nn->n_layers - 1].delta = calloc(nn->layers[nn->n_layers - 2].output_size, sizeof(float));
    nn->layers[nn->n_layers - 1].output = calloc(nn->layers[nn->n_layers - 2].output_size, sizeof(float));
}

void add_flatten_layer (neural_network_t *nn) {
    nn->n_layers++;

    nn->layers = realloc(nn->layers, nn->n_layers * sizeof(layer_t));
    nn->layers[nn->n_layers - 1].type = LAYER_FLATTEN;
    nn->layers[nn->n_layers - 1].output_size = nn->layers[nn->n_layers - 2].output_size;
    nn->layers[nn->n_layers - 1].delta = calloc(nn->layers[nn->n_layers - 2].output_size, sizeof(float));
    nn->layers[nn->n_layers - 1].output = calloc(nn->layers[nn->n_layers - 2].output_size, sizeof(float));
}

void free_neural_network (neural_network_t *nn) {
    for (int i = (nn->n_layers - 1); i >= 0 ; i--)
    {
        switch (nn->layers[i].type)
        {
        case LAYER_FC:
            free(nn->layers[i].output);
            free(nn->layers[i].delta);
            free(nn->layers[i].data.fc.bias);
            free(nn->layers[i].data.fc.grad_bias);
            free(nn->layers[i].data.fc.grad_weight);
            free(nn->layers[i].data.fc.m_bias);
            free(nn->layers[i].data.fc.m_weight);
            free(nn->layers[i].data.fc.total_grad_bias);
            free(nn->layers[i].data.fc.total_grad_weight);
            free(nn->layers[i].data.fc.v_bias);
            free(nn->layers[i].data.fc.v_weight);
            free(nn->layers[i].data.fc.weight);
            break;

        case LAYER_CONV:
            free(nn->layers[i].output);
            free(nn->layers[i].delta);
            free(nn->layers[i].data.conv.filter);
            free(nn->layers[i].data.conv.bias);
            free(nn->layers[i].data.conv.m_filter);
            free(nn->layers[i].data.conv.v_filter);
            free(nn->layers[i].data.conv.grad_filter);
            free(nn->layers[i].data.conv.total_grad_filter);
            free(nn->layers[i].data.conv.m_bias);
            free(nn->layers[i].data.conv.v_bias);
            free(nn->layers[i].data.conv.grad_bias);
            free(nn->layers[i].data.conv.total_grad_bias);
            break;

        case LAYER_POOL:
            free(nn->layers[i].output);
            free(nn->layers[i].delta);
            free(nn->layers[i].data.pool.mask);
            break;

        case LAYER_RELU:
            free(nn->layers[i].output);
            free(nn->layers[i].delta);
            break;

        case LAYER_LEAKY_RELU:
            free(nn->layers[i].output);
            free(nn->layers[i].delta);
            break;

        case LAYER_SOFTMAX:
            free(nn->layers[i].output);
            free(nn->layers[i].delta);
            break;

        case LAYER_FLATTEN:
            free(nn->layers[i].output);
            free(nn->layers[i].delta);
            break;
        }
    }
    free(nn->layers);
    free(nn);
}

void matrix_arr_mul (float *restrict output_arr, const float *restrict input_arr, const float *restrict matrix, int n_of_output_arr, int n_of_input_arr) {
    memset(output_arr, 0, n_of_output_arr * sizeof(float));

    for (int i = 0; i < n_of_output_arr; i++)
    {
        const float *mat_row = &matrix[n_of_input_arr * i];
        __m256 vsum = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= n_of_input_arr; j += 8) {
            __m256 vm = _mm256_loadu_ps(&mat_row[j]);
            __m256 vx = _mm256_loadu_ps(&input_arr[j]);
            vsum = _mm256_fmadd_ps(vm, vx, vsum);
        }
        float sum = 0.0f;
        for (; j < n_of_input_arr; j++) {
            sum += mat_row[j] * input_arr[j];
        }
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        output_arr[i] = _mm_cvtss_f32(s) + sum;
    }

}

void add_array (float *restrict operated_arr, const float *restrict input_arr, int n_of_arr) {
    int i = 0;
    for (; i + 8 <= n_of_arr; i += 8) {
        __m256 va = _mm256_loadu_ps(&operated_arr[i]);
        __m256 vb = _mm256_loadu_ps(&input_arr[i]);
        _mm256_storeu_ps(&operated_arr[i], _mm256_add_ps(va, vb));
    }
    for (; i < n_of_arr; i++) {
        operated_arr[i] += input_arr[i];
    }
}

void relu (const float *restrict input_arr, float *restrict output_arr, int n_of_arr) {
    __m256 vzero = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n_of_arr; i += 8) {
        __m256 vx = _mm256_loadu_ps(&input_arr[i]);
        __m256 vmask = _mm256_cmp_ps(vx, vzero, _CMP_GT_OS);
        _mm256_storeu_ps(&output_arr[i], _mm256_and_ps(vx, vmask));
    }
    for (; i < n_of_arr; i++) {
        output_arr[i] = input_arr[i] * (input_arr[i] > 0 ? 1.0f : 0.0f);
    }
}

void leaky_relu (const float *restrict input_arr, float *restrict output_arr, int n_of_arr) {
    __m256 vzero = _mm256_setzero_ps();
    __m256 vscale = _mm256_set1_ps(0.01f);
    int i = 0;
    for (; i + 8 <= n_of_arr; i += 8) {
        __m256 vx = _mm256_loadu_ps(&input_arr[i]);
        __m256 vmask = _mm256_cmp_ps(vx, vzero, _CMP_GT_OS);
        __m256 vpos = _mm256_and_ps(vx, vmask);
        __m256 vneg = _mm256_mul_ps(_mm256_andnot_ps(vmask, vx), vscale);
        _mm256_storeu_ps(&output_arr[i], _mm256_add_ps(vpos, vneg));
    }
    for (; i < n_of_arr; i++) {
        output_arr[i] = input_arr[i] * (input_arr[i] > 0 ? 1.0f : 0.01f);
    }
}

float extract_max (const float *restrict input_array, int n_of_input_arr) {
    __m256 vmax = _mm256_loadu_ps(input_array);
    int i = 8;
    for (; i + 8 <= n_of_input_arr; i += 8) {
        __m256 vx = _mm256_loadu_ps(&input_array[i]);
        vmax = _mm256_max_ps(vmax, vx);
    }
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 m = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
    float max = _mm_cvtss_f32(m);
    for (; i < n_of_input_arr; i++) {
        if (max < input_array[i]) max = input_array[i];
    }
    return max;
}

void softmax (const float *restrict input_arr, float *restrict output_arr, int n_of_arr) {
    float max = extract_max(input_arr, n_of_arr);
    float sum = 0.0f;
    int i = 0;
    for (; i + 8 <= n_of_arr; i += 8) {
        float xf[8], ef[8];
        _mm256_storeu_ps(xf, _mm256_sub_ps(_mm256_loadu_ps(&input_arr[i]), _mm256_set1_ps(max)));
        for (int k = 0; k < 8; k++) {
            ef[k] = expf(xf[k]);
            sum += ef[k];
        }
        _mm256_storeu_ps(&output_arr[i], _mm256_loadu_ps(ef));
    }
    for (; i < n_of_arr; i++) {
        output_arr[i] = expf(input_arr[i] - max);
        sum += output_arr[i];
    }

    __m256 vsum_inv = _mm256_set1_ps(1.0f / sum);
    int j = 0;
    for (; j + 8 <= n_of_arr; j += 8) {
        __m256 vt = _mm256_loadu_ps(&output_arr[j]);
        _mm256_storeu_ps(&output_arr[j], _mm256_mul_ps(vt, vsum_inv));
    }
    for (; j < n_of_arr; j++) {
        output_arr[j] /= sum;
    }
}

void forward_convolution (const float *restrict input, const float *restrict filter, float *restrict output, int n_input_height, int n_input_width, int n_input_channel, int filter_height, int filter_width, int n_filters, int stride, const float *restrict bias) {
    int out_h = (n_input_height - filter_height) / stride + 1;
    int out_w = (n_input_width - filter_width) / stride  + 1;
    int col_rows = n_input_channel * filter_height * filter_width;
    int col_cols = out_h * out_w;
    int out_size = n_filters * out_h * out_w;

    memset(output, 0, out_size * sizeof(float));

    float *col_buf = (float *)malloc(col_rows * col_cols * sizeof(float));
    im2col(input, col_buf, n_input_height, n_input_width, n_input_channel,
           filter_height, filter_width, stride);

    #pragma omp parallel for
    for (int n = 0; n < n_filters; n++) {
        const float *filter_row = &filter[n * col_rows];
        float *out_row = &output[n * col_cols];
        for (int k = 0; k < col_rows; k++) {
            __m256 vw = _mm256_set1_ps(filter_row[k]);
            const float *col_row = &col_buf[k * col_cols];
            int ij = 0;
            for (; ij + 8 <= col_cols; ij += 8) {
                __m256 vo = _mm256_loadu_ps(&out_row[ij]);
                __m256 vc = _mm256_loadu_ps(&col_row[ij]);
                _mm256_storeu_ps(&out_row[ij], _mm256_fmadd_ps(vw, vc, vo));
            }
            for (; ij < col_cols; ij++) {
                out_row[ij] += filter_row[k] * col_row[ij];
            }
        }
        __m256 vb = _mm256_set1_ps(bias[n]);
        int ij = 0;
        for (; ij + 8 <= col_cols; ij += 8) {
            __m256 vo = _mm256_loadu_ps(&out_row[ij]);
            _mm256_storeu_ps(&out_row[ij], _mm256_add_ps(vo, vb));
        }
        for (; ij < col_cols; ij++) {
            out_row[ij] += bias[n];
        }
    }

    free(col_buf);
}

void forward_maxpool(const float *restrict input, float *restrict output, int n_channels, int in_height, int in_width, int kernel_height, int kernel_width, uint8_t *restrict mask) {
    //standby
    int out_h = in_height / kernel_height;
    int out_w = in_width / kernel_width;
    memset(mask, 0, n_channels * in_height * in_width * sizeof(uint8_t));

    for (size_t c = 0; c < n_channels; c++)
    {
        for (size_t oh = 0; oh < out_h; oh++)
        {
            for (size_t ow = 0; ow < out_w; ow++)
            {
                float max = -FLT_MAX;
                int max_indics = c * in_height * in_width + (oh*kernel_height)*in_width + (ow*kernel_width);
                for (size_t kh = 0; kh < kernel_height; kh++)
                {
                    for (size_t kw = 0; kw < kernel_width; kw++)
                    {
                        float value = input[c * in_height * in_width + (oh*kernel_height+kh)*in_width + (ow*kernel_width+kw)];
                        if (value > max)
                        {
                            max = value;
                            max_indics = c * in_height * in_width + (oh*kernel_height+kh)*in_width + (ow*kernel_width+kw);
                        }
                        
                    }
                    
                }
                output[c * out_h * out_w + oh * out_w + ow] = max;
                mask[max_indics] = 1;
            }
            
        }
        
    }
    
}

void forward_pass (neural_network_t *nn, const float *restrict input) {
    const float *current_input = input;
    for (size_t i = 0; i < nn->n_layers; i++)
    {
        switch (nn->layers[i].type)
        {
        case LAYER_FC:
            matrix_arr_mul(nn->layers[i].output, current_input, nn->layers[i].data.fc.weight, nn->layers[i].data.fc.out_size, nn->layers[i].data.fc.in_size);
            add_array(nn->layers[i].output, nn->layers[i].data.fc.bias, nn->layers[i].data.fc.out_size);
            break;

        case LAYER_CONV:
            forward_convolution(current_input, nn->layers[i].data.conv.filter, nn->layers[i].output, nn->layers[i].data.conv.in_height, nn->layers[i].data.conv.in_width, nn->layers[i].data.conv.in_channel, nn->layers[i].data.conv.filter_height, nn->layers[i].data.conv.filter_width, nn->layers[i].data.conv.n_filters, nn->layers[i].data.conv.filter_stride, nn->layers[i].data.conv.bias);
            break;

        case LAYER_POOL:
            forward_maxpool(current_input, nn->layers[i].output, nn->layers[i].data.pool.in_channel, nn->layers[i].data.pool.in_height, nn->layers[i].data.pool.in_width, nn->layers[i].data.pool.kernel_height, nn->layers[i].data.pool.kernel_width, nn->layers[i].data.pool.mask);
            break;

        case LAYER_RELU:
            relu(current_input, nn->layers[i].output, nn->layers[i].output_size);
            break;

        case LAYER_LEAKY_RELU:
            leaky_relu(current_input, nn->layers[i].output, nn->layers[i].output_size);
            break;

        case LAYER_SOFTMAX:
            softmax(current_input, nn->layers[i].output, nn->layers[i].output_size);
            break;

        case LAYER_FLATTEN:
            memcpy(nn->layers[i].output, current_input, nn->layers[i].output_size * sizeof(float));
            break;
        }
        current_input = nn->layers[i].output;
    }
    
}

void compute_output_softmax_delta (float *restrict output_delta, const float *restrict output_layer_activation, const float *restrict answer_arr, int n_of_arr) {
    int i = 0;
    for (; i + 8 <= n_of_arr; i += 8) {
        __m256 va = _mm256_loadu_ps(&output_layer_activation[i]);
        __m256 vb = _mm256_loadu_ps(&answer_arr[i]);
        _mm256_storeu_ps(&output_delta[i], _mm256_sub_ps(va, vb));
    }
    for (; i < n_of_arr; i++) {
        output_delta[i] = output_layer_activation[i] - answer_arr[i];
    }
}

void compute_backward_fc (float *restrict output_delta, const float *restrict current_delta, const float *restrict weight, int n_output_delta, int n_current_delta) {
    memset(output_delta, 0, n_output_delta * sizeof(float));
    for (int j = 0; j < n_current_delta; j++)
    {
        float w = current_delta[j];
        __m256 vw = _mm256_set1_ps(w);
        const float *w_row = &weight[j * n_output_delta];
        int i = 0;
        for (; i + 8 <= n_output_delta; i += 8) {
            __m256 vo = _mm256_loadu_ps(&output_delta[i]);
            __m256 vw_row = _mm256_loadu_ps(&w_row[i]);
            _mm256_storeu_ps(&output_delta[i], _mm256_fmadd_ps(vw, vw_row, vo));
        }
        for (; i < n_output_delta; i++) {
            output_delta[i] += w * w_row[i];
        }
    }
}

void compute_weight_grad (const float *restrict z_delta, const float *restrict previous_activation_arr, float *restrict output_arr, int n_of_output, int n_of_input) {
    for (int i = 0; i < n_of_output; i++)
    {
        __m256 vz = _mm256_set1_ps(z_delta[i]);
        float *out_row = &output_arr[i * n_of_input];
        int j = 0;
        for (; j + 8 <= n_of_input; j += 8) {
            __m256 vact = _mm256_loadu_ps(&previous_activation_arr[j]);
            _mm256_storeu_ps(&out_row[j], _mm256_mul_ps(vz, vact));
        }
        for (; j < n_of_input; j++) {
            out_row[j] = z_delta[i] * previous_activation_arr[j];
        }
    }
}

void compute_bias_grad (float *restrict output_bias_grad, const float *restrict delta, int n_of_arr) {
    memcpy(output_bias_grad, delta, n_of_arr * sizeof(float));
}

void compute_backward_maxpool (float *restrict computed_delta, const float *restrict current_delta, const uint8_t *restrict mask, int n_channels, int in_h, int in_w) {
    memset(computed_delta, 0, n_channels * in_h * in_w * sizeof(float));

    int index = 0;
    for (size_t c = 0; c < n_channels; c++)
    {
        for (size_t h = 0; h < in_h; h++)
        {
            for (size_t w = 0; w < in_w; w++)
            {
                if (mask[in_h * in_w * c + in_w * h + w] == 1)
                {
                    computed_delta[in_h * in_w * c + in_w * h + w] = current_delta[index];
                    index++;
                }
                
            }
            
        }
        
    }
    
}

void compute_backward_conv (float *restrict computed_delta, float *restrict grad_filter, float *restrict grad_bias, const float *restrict activation, const float *restrict current_delta, const float *restrict filter, const float *restrict input, int n_input_height, int n_input_width, int filter_height, int filter_width, int n_filters, int in_channel, int in_h, int in_w, int stride) {
    (void)activation; (void)in_h; (void)in_w;
    int out_h = (n_input_height - filter_height) / stride + 1;
    int out_w = (n_input_width - filter_width) / stride  + 1;
    int col_rows = in_channel * filter_height * filter_width;
    int col_cols = out_h * out_w;

    memset(computed_delta, 0, in_channel * n_input_height * n_input_width * sizeof(float));
    memset(grad_filter, 0, n_filters * col_rows * sizeof(float));

    float *col_buf = (float *)malloc(col_rows * col_cols * sizeof(float));
    im2col(input, col_buf, n_input_height, n_input_width, in_channel,
           filter_height, filter_width, stride);

    // grad_filter[n, k] = sum_{ij} current_delta[n, ij] * col_buf[k, ij]
    #pragma omp parallel for
    for (int n = 0; n < n_filters; n++) {
        const float *delta_row = &current_delta[n * col_cols];
        float *grad_row = &grad_filter[n * col_rows];
        for (int k = 0; k < col_rows; k++) {
            const float *col_row = &col_buf[k * col_cols];
            __m256 vsum = _mm256_setzero_ps();
            int ij = 0;
            for (; ij + 8 <= col_cols; ij += 8) {
                __m256 vd = _mm256_loadu_ps(&delta_row[ij]);
                __m256 vc = _mm256_loadu_ps(&col_row[ij]);
                vsum = _mm256_fmadd_ps(vd, vc, vsum);
            }
            float sum = 0.0f;
            for (; ij < col_cols; ij++) {
                sum += delta_row[ij] * col_row[ij];
            }
            __m128 hi = _mm256_extractf128_ps(vsum, 1);
            __m128 lo = _mm256_castps256_ps128(vsum);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            grad_row[k] = _mm_cvtss_f32(s) + sum;
        }
    }

    // computed_delta[c, ih, iw] via col2im of (filter^T * current_delta)
    // tmp_mat[t, ij] = sum_{n} filter[n, t] * current_delta[n, ij]
    float *tmp_mat = (float *)malloc(col_rows * col_cols * sizeof(float));
    memset(tmp_mat, 0, col_rows * col_cols * sizeof(float));
    #pragma omp parallel for
    for (int t = 0; t < col_rows; t++) {
        float *tmp_row = &tmp_mat[t * col_cols];
        for (int n = 0; n < n_filters; n++) {
            __m256 vw = _mm256_set1_ps(filter[n * col_rows + t]);
            const float *delta_row = &current_delta[n * col_cols];
            int ij = 0;
            for (; ij + 8 <= col_cols; ij += 8) {
                __m256 vt = _mm256_loadu_ps(&tmp_row[ij]);
                __m256 vd = _mm256_loadu_ps(&delta_row[ij]);
                _mm256_storeu_ps(&tmp_row[ij], _mm256_fmadd_ps(vw, vd, vt));
            }
            for (; ij < col_cols; ij++) {
                tmp_row[ij] += filter[n * col_rows + t] * delta_row[ij];
            }
        }
    }
    col2im(tmp_mat, computed_delta, n_input_height, n_input_width, in_channel,
           filter_height, filter_width, stride);
    free(tmp_mat);
    free(col_buf);

    // grad_bias[n] = sum over positions current_delta[n, :]
    for (int n = 0; n < n_filters; n++) {
        const float *delta_row = &current_delta[n * col_cols];
        __m256 vsum = _mm256_setzero_ps();
        int ij = 0;
        for (; ij + 8 <= col_cols; ij += 8) {
            vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(&delta_row[ij]));
        }
        float sum = 0.0f;
        for (; ij < col_cols; ij++) {
            sum += delta_row[ij];
        }
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        grad_bias[n] = _mm_cvtss_f32(s) + sum;
    }
}

void backward_pass (neural_network_t *nn, const float *restrict input, const float *restrict answer) {
    float *current_delta;
    switch (nn->layers[nn->n_layers - 1].type)
    {
    case LAYER_SOFTMAX:
        compute_output_softmax_delta(nn->layers[nn->n_layers - 1].delta, nn->layers[nn->n_layers - 1].output, answer, nn->layers[nn->n_layers - 1].output_size);
        break;
    }
    current_delta = nn->layers[nn->n_layers - 1].delta;

    for (int i = (nn->n_layers - 2); i >= 0; i--)
    {
        switch (nn->layers[i].type)
        {
        case LAYER_FC:
            if (i == 0)
            {
                compute_weight_grad(current_delta, input, nn->layers[i].data.fc.grad_weight, nn->layers[i].output_size, nn->layers[i].data.fc.in_size);
            }
            else
            {
                compute_weight_grad(current_delta, nn->layers[i - 1].output, nn->layers[i].data.fc.grad_weight, nn->layers[i].output_size, nn->layers[i - 1].output_size);
            }
            compute_bias_grad(nn->layers[i].data.fc.grad_bias, current_delta, nn->layers[i].output_size);
            if (i > 0)
            {
                compute_backward_fc(nn->layers[i].delta, current_delta, nn->layers[i].data.fc.weight, nn->layers[i].data.fc.in_size, nn->layers[i].data.fc.out_size);
            }
            for (size_t j = 0; j < nn->layers[i].data.fc.in_size * nn->layers[i].data.fc.out_size; j++)
            {
                nn->layers[i].data.fc.total_grad_weight[j] += nn->layers[i].data.fc.grad_weight[j];
            }
            for (size_t j = 0; j < nn->layers[i].data.fc.out_size; j++)
            {
                nn->layers[i].data.fc.total_grad_bias[j] += nn->layers[i].data.fc.grad_bias[j];
            }
            break;

        case LAYER_CONV:
            if (i == 0)
            {
                compute_backward_conv(nn->layers[i].delta, nn->layers[i].data.conv.grad_filter, nn->layers[i].data.conv.grad_bias, nn->layers[i].output, current_delta, nn->layers[i].data.conv.filter, input, nn->layers[i].data.conv.in_height, nn->layers[i].data.conv.in_width, nn->layers[i].data.conv.filter_height, nn->layers[i].data.conv.filter_width, nn->layers[i].data.conv.n_filters, nn->layers[i].data.conv.in_channel, nn->layers[i].data.conv.in_height, nn->layers[i].data.conv.in_width, nn->layers[i].data.conv.filter_stride);
            }
            else
            {
                compute_backward_conv(nn->layers[i].delta, nn->layers[i].data.conv.grad_filter, nn->layers[i].data.conv.grad_bias, nn->layers[i].output, current_delta, nn->layers[i].data.conv.filter, nn->layers[i - 1].output, nn->layers[i].data.conv.in_height, nn->layers[i].data.conv.in_width, nn->layers[i].data.conv.filter_height, nn->layers[i].data.conv.filter_width, nn->layers[i].data.conv.n_filters, nn->layers[i].data.conv.in_channel, nn->layers[i].data.conv.in_height, nn->layers[i].data.conv.in_width, nn->layers[i].data.conv.filter_stride);
            }
            for (size_t j = 0; j < nn->layers[i].data.conv.filter_height * nn->layers[i].data.conv.filter_width * nn->layers[i].data.conv.n_filters * nn->layers[i].data.conv.in_channel; j++)
            {
                nn->layers[i].data.conv.total_grad_filter[j] += nn->layers[i].data.conv.grad_filter[j];
            }
            for (size_t j = 0; j < nn->layers[i].data.conv.n_filters; j++)
            {
                nn->layers[i].data.conv.total_grad_bias[j] += nn->layers[i].data.conv.grad_bias[j];
            }
            break;

        case LAYER_POOL:
            compute_backward_maxpool(nn->layers[i].delta, current_delta, nn->layers[i].data.pool.mask, nn->layers[i].data.pool.in_channel, nn->layers[i].data.pool.in_height, nn->layers[i].data.pool.in_width);
            break;

        case LAYER_RELU: {
            float *delta_out = nn->layers[i].delta;
            const float *prev_out = nn->layers[i - 1].output;
            __m256 vzero = _mm256_setzero_ps();
            int j = 0;
            for (; j + 8 <= nn->layers[i].output_size; j += 8) {
                __m256 vd = _mm256_loadu_ps(&current_delta[j]);
                __m256 vp = _mm256_loadu_ps(&prev_out[j]);
                __m256 vmask = _mm256_cmp_ps(vp, vzero, _CMP_GT_OS);
                _mm256_storeu_ps(&delta_out[j], _mm256_and_ps(vd, vmask));
            }
            for (; j < nn->layers[i].output_size; j++) {
                delta_out[j] = current_delta[j] * (prev_out[j] > 0);
            }
            break;
        }

        case LAYER_LEAKY_RELU: {
            float *delta_out = nn->layers[i].delta;
            const float *prev_out = nn->layers[i - 1].output;
            __m256 vzero = _mm256_setzero_ps();
            __m256 vscale = _mm256_set1_ps(0.01f);
            int j = 0;
            for (; j + 8 <= nn->layers[i].output_size; j += 8) {
                __m256 vd = _mm256_loadu_ps(&current_delta[j]);
                __m256 vp = _mm256_loadu_ps(&prev_out[j]);
                __m256 vmask = _mm256_cmp_ps(vp, vzero, _CMP_GT_OS);
                __m256 vpos = _mm256_and_ps(vd, vmask);
                __m256 vneg = _mm256_mul_ps(_mm256_andnot_ps(vmask, vd), vscale);
                _mm256_storeu_ps(&delta_out[j], _mm256_add_ps(vpos, vneg));
            }
            for (; j < nn->layers[i].output_size; j++) {
                delta_out[j] = current_delta[j] * (prev_out[j] > 0) + 0.01f * current_delta[j] * (prev_out[j] <= 0);
            }
            break;
        }

        case LAYER_SOFTMAX:
            break;

        case LAYER_FLATTEN:
            memcpy(nn->layers[i].delta, current_delta, nn->layers[i].output_size * sizeof(float));
            break;
        }
        current_delta = nn->layers[i].delta;
    }
    
}

void parameter_initialize (neural_network_t *nn) {
    for (int i = 0; i < nn->n_layers; i++) {
        switch (nn->layers[i].type) {
        case LAYER_FC:{
            float std = sqrtf(2.0f / (float)nn->layers[i].data.fc.in_size);
            for (int j = 0; j < nn->layers[i].data.fc.in_size * nn->layers[i].data.fc.out_size; j++) {
                nn->layers[i].data.fc.weight[j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                nn->layers[i].data.fc.weight[j] *= std * sqrtf(3.0f);
            }
            break;
        }
        case LAYER_CONV:{
            float limit = sqrtf(2.0f / (nn->layers[i].data.conv.in_channel * nn->layers[i].data.conv.filter_height * nn->layers[i].data.conv.filter_width));
            for (int j = 0; j < nn->layers[i].data.conv.in_channel * nn->layers[i].data.conv.n_filters * nn->layers[i].data.conv.filter_height * nn->layers[i].data.conv.filter_width; j++) {
                nn->layers[i].data.conv.filter[j] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
            }
            break;
        }
        default:
            break;
        }
    }
}

void update_param_adam (neural_network_t *nn, float lr, float weight_decay, float beta1, float beta2, float eps, int t, int batch_size) {
    float bc = lr * sqrtf(1.0f - powf(beta2, t)) / (1.0f - powf(beta1, t));
    float inv_batch = 1.0f / batch_size;
    __m256 vbeta1 = _mm256_set1_ps(beta1);
    __m256 vbeta1c = _mm256_set1_ps(1.0f - beta1);
    __m256 vbeta2 = _mm256_set1_ps(beta2);
    __m256 vbeta2c = _mm256_set1_ps(1.0f - beta2);
    __m256 vbc = _mm256_set1_ps(bc);
    __m256 veps = _mm256_set1_ps(eps);
    __m256 vlr = _mm256_set1_ps(lr);
    __m256 vwd = _mm256_set1_ps(weight_decay);
    __m256 vinv_batch = _mm256_set1_ps(inv_batch);

    #pragma omp parallel for
    for (int layer = 0; layer < nn->n_layers - 1; layer++)
    {
        switch (nn->layers[layer].type)
        {
        case LAYER_FC:{
            int n_w = nn->layers[layer].data.fc.in_size * nn->layers[layer].data.fc.out_size;
            float *m = nn->layers[layer].data.fc.m_weight;
            float *v = nn->layers[layer].data.fc.v_weight;
            float *w = nn->layers[layer].data.fc.weight;
            float *g_total = nn->layers[layer].data.fc.total_grad_weight;
            int j = 0;
            for (; j + 8 <= n_w; j += 8) {
                __m256 vg = _mm256_mul_ps(_mm256_loadu_ps(&g_total[j]), vinv_batch);
                __m256 vm = _mm256_loadu_ps(&m[j]);
                __m256 vv = _mm256_loadu_ps(&v[j]);
                vm = _mm256_add_ps(_mm256_mul_ps(vbeta1, vm), _mm256_mul_ps(vbeta1c, vg));
                vv = _mm256_add_ps(_mm256_mul_ps(vbeta2, vv), _mm256_mul_ps(vbeta2c, _mm256_mul_ps(vg, vg)));
                __m256 vw = _mm256_loadu_ps(&w[j]);
                vw = _mm256_sub_ps(vw, _mm256_add_ps(
                    _mm256_div_ps(_mm256_mul_ps(vbc, vm), _mm256_add_ps(_mm256_sqrt_ps(vv), veps)),
                    _mm256_mul_ps(_mm256_mul_ps(vlr, vwd), vw)
                ));
                _mm256_storeu_ps(&m[j], vm);
                _mm256_storeu_ps(&v[j], vv);
                _mm256_storeu_ps(&w[j], vw);
            }
            for (; j < n_w; j++) {
                float g = g_total[j] * inv_batch;
                m[j] = beta1 * m[j] + (1 - beta1) * g;
                v[j] = beta2 * v[j] + (1 - beta2) * g * g;
                w[j] -= bc * m[j] / (sqrtf(v[j]) + eps) + lr * weight_decay * w[j];
            }
            memset(g_total, 0, n_w * sizeof(float));

            int n_b = nn->layers[layer].data.fc.out_size;
            float *mb = nn->layers[layer].data.fc.m_bias;
            float *vb = nn->layers[layer].data.fc.v_bias;
            float *wb = nn->layers[layer].data.fc.bias;
            float *gb_total = nn->layers[layer].data.fc.total_grad_bias;
            j = 0;
            for (; j + 8 <= n_b; j += 8) {
                __m256 vg = _mm256_mul_ps(_mm256_loadu_ps(&gb_total[j]), vinv_batch);
                __m256 vm = _mm256_loadu_ps(&mb[j]);
                __m256 vv = _mm256_loadu_ps(&vb[j]);
                vm = _mm256_add_ps(_mm256_mul_ps(vbeta1, vm), _mm256_mul_ps(vbeta1c, vg));
                vv = _mm256_add_ps(_mm256_mul_ps(vbeta2, vv), _mm256_mul_ps(vbeta2c, _mm256_mul_ps(vg, vg)));
                __m256 vw = _mm256_loadu_ps(&wb[j]);
                vw = _mm256_sub_ps(vw, _mm256_div_ps(_mm256_mul_ps(vbc, vm), _mm256_add_ps(_mm256_sqrt_ps(vv), veps)));
                _mm256_storeu_ps(&mb[j], vm);
                _mm256_storeu_ps(&vb[j], vv);
                _mm256_storeu_ps(&wb[j], vw);
            }
            for (; j < n_b; j++) {
                float g = gb_total[j] * inv_batch;
                mb[j] = beta1 * mb[j] + (1 - beta1) * g;
                vb[j] = beta2 * vb[j] + (1 - beta2) * g * g;
                wb[j] -= bc * mb[j] / (sqrtf(vb[j]) + eps);
            }
            memset(gb_total, 0, n_b * sizeof(float));
            break;
        }

        case LAYER_CONV:{
            int n_f = nn->layers[layer].data.conv.n_filters * nn->layers[layer].data.conv.in_channel * nn->layers[layer].data.conv.filter_height * nn->layers[layer].data.conv.filter_width;
            float *m = nn->layers[layer].data.conv.m_filter;
            float *v = nn->layers[layer].data.conv.v_filter;
            float *w = nn->layers[layer].data.conv.filter;
            float *g_total = nn->layers[layer].data.conv.total_grad_filter;
            int i = 0;
            for (; i + 8 <= n_f; i += 8) {
                __m256 vg = _mm256_mul_ps(_mm256_loadu_ps(&g_total[i]), vinv_batch);
                __m256 vm = _mm256_loadu_ps(&m[i]);
                __m256 vv = _mm256_loadu_ps(&v[i]);
                vm = _mm256_add_ps(_mm256_mul_ps(vbeta1, vm), _mm256_mul_ps(vbeta1c, vg));
                vv = _mm256_add_ps(_mm256_mul_ps(vbeta2, vv), _mm256_mul_ps(vbeta2c, _mm256_mul_ps(vg, vg)));
                __m256 vw = _mm256_loadu_ps(&w[i]);
                vw = _mm256_sub_ps(vw, _mm256_add_ps(
                    _mm256_div_ps(_mm256_mul_ps(vbc, vm), _mm256_add_ps(_mm256_sqrt_ps(vv), veps)),
                    _mm256_mul_ps(_mm256_mul_ps(vlr, vwd), vw)
                ));
                _mm256_storeu_ps(&m[i], vm);
                _mm256_storeu_ps(&v[i], vv);
                _mm256_storeu_ps(&w[i], vw);
            }
            for (; i < n_f; i++) {
                float g = g_total[i] * inv_batch;
                m[i] = beta1 * m[i] + (1 - beta1) * g;
                v[i] = beta2 * v[i] + (1 - beta2) * g * g;
                w[i] -= bc * m[i] / (sqrtf(v[i]) + eps) + lr * weight_decay * w[i];
            }
            memset(g_total, 0, n_f * sizeof(float));

            int n_b = nn->layers[layer].data.conv.n_filters;
            float *mb = nn->layers[layer].data.conv.m_bias;
            float *vb = nn->layers[layer].data.conv.v_bias;
            float *wb = nn->layers[layer].data.conv.bias;
            float *gb_total = nn->layers[layer].data.conv.total_grad_bias;
            i = 0;
            for (; i + 8 <= n_b; i += 8) {
                __m256 vg = _mm256_mul_ps(_mm256_loadu_ps(&gb_total[i]), vinv_batch);
                __m256 vm = _mm256_loadu_ps(&mb[i]);
                __m256 vv = _mm256_loadu_ps(&vb[i]);
                vm = _mm256_add_ps(_mm256_mul_ps(vbeta1, vm), _mm256_mul_ps(vbeta1c, vg));
                vv = _mm256_add_ps(_mm256_mul_ps(vbeta2, vv), _mm256_mul_ps(vbeta2c, _mm256_mul_ps(vg, vg)));
                __m256 vw = _mm256_loadu_ps(&wb[i]);
                vw = _mm256_sub_ps(vw, _mm256_div_ps(_mm256_mul_ps(vbc, vm), _mm256_add_ps(_mm256_sqrt_ps(vv), veps)));
                _mm256_storeu_ps(&mb[i], vm);
                _mm256_storeu_ps(&vb[i], vv);
                _mm256_storeu_ps(&wb[i], vw);
            }
            for (; i < n_b; i++) {
                float g = gb_total[i] * inv_batch;
                mb[i] = beta1 * mb[i] + (1 - beta1) * g;
                vb[i] = beta2 * vb[i] + (1 - beta2) * g * g;
                wb[i] -= bc * mb[i] / (sqrtf(vb[i]) + eps);
            }
            memset(gb_total, 0, n_b * sizeof(float));
            break;
        }

        default:
            break;
    }
    }

}