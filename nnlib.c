#include "nnlib.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

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
    nn->layers[nn->n_layers - 1].delta = calloc(out_size, sizeof(float));
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
    nn->layers[nn->n_layers - 1].delta = calloc(n_filters * out_height * out_width, sizeof(float));
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
    nn->layers[nn->n_layers - 1].delta = calloc(in_channel * out_height * out_width, sizeof(float));
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

void matrix_arr_mul (float *output_arr, float *input_arr, float *matrix, int n_of_output_arr, int n_of_input_arr) {
    memset(output_arr, 0, n_of_output_arr * sizeof(float));
    
    for (int i = 0; i < n_of_output_arr; i++)
    {
        for (int j = 0; j < n_of_input_arr; j++)
        {
            output_arr[i] += matrix[n_of_input_arr * i + j] * input_arr[j];
        }
        
    }

}

void add_array (float *operated_arr, float *input_arr, int n_of_arr) {
    for (size_t i = 0; i < n_of_arr; i++)
    {
        operated_arr[i] += input_arr[i];
    }
    
}

void relu (float *input_arr, float *output_arr, int n_of_arr) {
    for (int i = 0; i < n_of_arr; i++)
    {
        output_arr[i] = fmaxf(0, input_arr[i]);
    }

}

void leaky_relu (float *input_arr, float *output_arr, int n_of_arr) {
    for (int i = 0; i < n_of_arr; i++)
    {
        if (input_arr[i] > 0) {
            output_arr[i] = input_arr[i];
        }
        else {
            output_arr[i] = 0.01f * input_arr[i];
        }
        
    }

}

float extract_max (float *input_array, int n_of_input_arr) {
    float max = input_array[0];
    for (int i = 0; i < n_of_input_arr; i++)
    {
        if (max < input_array[i])
        {
            max = input_array[i];
        }
        
    }
    
    return max;
}

void softmax (float *input_arr, float *output_arr, int n_of_arr) {
    float max = 0.0, sum = 0.0;
    float *tmp = malloc(n_of_arr * sizeof(float));
    max = extract_max(input_arr, n_of_arr);
    for (int i = 0; i < n_of_arr; i++)
    {
        tmp[i] = expf(input_arr[i] - max);
        sum += tmp[i];
    }
    for (int j = 0; j < n_of_arr; j++)
    {
        output_arr[j] = tmp[j] / sum;
    }
    free(tmp);

}

void forward_convolution (float *input, float *filter, float *output, int n_input_height, int n_input_width, int n_input_channel, int filter_height, int filter_width, int n_filters, int stride, float *bias) {
    //standby
    int n_output_height = (n_input_height - filter_height) / stride + 1;
    int n_output_width = (n_input_width - filter_width) / stride  + 1;

    //zero fill
    memset(output, 0, n_filters * n_output_height * n_output_width * sizeof(float));

    for (size_t n = 0; n < n_filters; n++)
    {
        for (size_t oh = 0; oh < n_output_height; oh++)
        {
            for (size_t ow = 0; ow < n_output_width; ow++)
            {
                for (size_t c = 0; c < n_input_channel; c++)
                {
                    for (size_t fh = 0; fh < filter_height; fh++)
                    {
                        for (size_t fw = 0; fw < filter_width; fw++)
                        {
                            output[n * n_output_height * n_output_width + oh * n_output_width + ow] += input[c * n_input_height * n_input_width + (oh*stride+fh) * n_input_width + (ow*stride+fw)] * filter[n * n_input_channel * filter_width * filter_height + c * filter_width * filter_height  + fh * filter_width + fw];
                        }
                        
                    }
                    
                }
                
            }
            
        }
        
    }
    for (int n = 0; n < n_filters; n++)
        for (int p = 0; p < n_output_height * n_output_width; p++)
            output[n * n_output_height * n_output_width + p] += bias[n];
    
}

void forward_maxpool(float *input, float *output, int n_channels, int in_height, int in_width, int kernel_height, int kernel_width, uint8_t *mask) {
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

void forward_pass (neural_network_t *nn, float *input) {
    float *current_input = input;
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

void compute_output_softmax_delta (float *output_delta, float *output_layer_activation, float *answer_arr, int n_of_arr) {
    for (size_t i = 0; i < n_of_arr; i++)
    {
        output_delta[i] = output_layer_activation[i] - answer_arr[i];
    }
    
}

void compute_backward_fc (float *output_delta, float *current_delta, float *weight, float *backward_pre_activation, int n_of_activation, int n_of_z_delta) {
    memset(output_delta, 0, n_of_activation * sizeof(float));
    for (size_t i = 0; i < n_of_activation; i++)
    {
        for (size_t j = 0; j < n_of_z_delta; j++)
        {
            output_delta[i] += current_delta[j] * current_weight[j * n_of_activation+ i];
        }
        
    }
}

void backward_pass (neural_network_t *nn, float *answer) {
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
            break;

        case LAYER_CONV:
            break;

        case LAYER_POOL:
            break;

        case LAYER_RELU:
            for (size_t j = 0; j < nn->layers[i].output_size; j++)
            {
                nn->layers[i].delta[j] = current_delta[j] * (nn->layers[i - 1].output[j] > 0);
            }
            break;

        case LAYER_LEAKY_RELU:
            for (size_t j = 0; j < nn->layers[i].output_size; j++)
            {
                nn->layers[i].delta[j] = current_delta[j] * (nn->layers[i - 1].output[j] > 0) + 0.01 * current_delta[j] * (nn->layers[i - 1].output[j] <= 0);
            }
            break;

        case LAYER_SOFTMAX:
            break;

        case LAYER_FLATTEN:
            memcpy(nn->layers[i].delta, current_delta, nn->layers[i].output_size * sizeof(float));
            break;
        }
        current_delta = nn->layers[i].delta;
    }
    
}