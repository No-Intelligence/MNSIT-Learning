#include "fcnnlib.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

layer_t* alloc_layer (int n_layers, int *layer_size) {
    layer_t *p = calloc(n_layers, sizeof(layer_t));
    for (size_t i = 0; i < n_layers; i++)
    {
        p[i].pre_activation = calloc(layer_size[i], sizeof(float));
        p[i].activation = calloc(layer_size[i], sizeof(float));
    }
    return p;
}

void free_layer (layer_t *layer, int n_layers, int *layer_size) {
    for (size_t i = 0; i < n_layers; i++)
    {
        free(layer[i].pre_activation);
        free(layer[i].activation);
    }
    free(layer);
}

parameter_t* alloc_parameter (int n_layers, int *layer_size) {
    parameter_t *p = calloc(n_layers - 1, sizeof(parameter_t));
    for (size_t i = 0; i < n_layers - 1; i++)
    {
        p[i].weight   = calloc(layer_size[i] * layer_size[i + 1], sizeof(float));
        p[i].bias     = calloc(layer_size[i + 1], sizeof(float));
        p[i].m_weight = calloc(layer_size[i] * layer_size[i + 1], sizeof(float));
        p[i].v_weight = calloc(layer_size[i] * layer_size[i + 1], sizeof(float));
        p[i].m_bias   = calloc(layer_size[i + 1], sizeof(float));
        p[i].v_bias   = calloc(layer_size[i + 1], sizeof(float));
    }
    return p;
}

void free_parameter (parameter_t *parameter, int n_layers, int *layer_size) {
    for (size_t i = 0; i < n_layers - 1; i++)
    {
        free(parameter[i].weight);
        free(parameter[i].bias);
        free(parameter[i].m_weight);
        free(parameter[i].v_weight);
        free(parameter[i].m_bias);
        free(parameter[i].v_bias);
    }
    free(parameter);
}

layre_grad_t* alloc_layer_grad (int n_layers, int *layer_size) {
    layre_grad_t *p = calloc(n_layers - 1, sizeof(layre_grad_t));
    for (size_t i = 0; i < n_layers - 1; i++)
    {
        p[i].delta = calloc(layer_size[i + 1], sizeof(float));
    }
    return p;
}

void free_layer_grad (layre_grad_t *layer_grad, int n_layers, int *layer_size) {
    for (size_t i = 0; i < n_layers - 1; i++)
    {
        free(layer_grad[i].delta);
    }
    free(layer_grad);
}

param_grad_t* alloc_param_grad (int n_layers, int *layer_size) {
    param_grad_t *p = calloc(n_layers - 1, sizeof(param_grad_t));
    for (size_t i = 0; i < n_layers - 1; i++)
    {
        p[i].weight_grad = calloc(layer_size[i] * layer_size[i + 1], sizeof(float));
        p[i].bias_grad = calloc(layer_size[i + 1], sizeof(float));
        p[i].total_weight_grad = calloc(layer_size[i] * layer_size[i + 1], sizeof(float));
        p[i].total_bias_grad = calloc(layer_size[i + 1], sizeof(float));
    }
    return p;
}

void free_param_grad (param_grad_t *param_grad, int n_layers, int *layer_size) {
    for (size_t i = 0; i < n_layers - 1; i++)
    {
        free(param_grad[i].weight_grad);
        free(param_grad[i].bias_grad);
        free(param_grad[i].total_weight_grad);
        free(param_grad[i].total_bias_grad);
    }
    free(param_grad);
}

neural_network_t* alloc_neural_network (int n_layers, int *layer_size, activation_t *activations) {
    neural_network_t *p = calloc(1, sizeof(neural_network_t));
    p->layers = alloc_layer(n_layers, layer_size);
    p->parameters = alloc_parameter(n_layers, layer_size);
    p->activations = calloc(n_layers - 1, sizeof(activation_t));
    memcpy(p->activations, activations, (n_layers - 1) * sizeof(activation_t));
    p->layer_grad = alloc_layer_grad(n_layers, layer_size);
    p->param_grad = alloc_param_grad(n_layers, layer_size);
    p->n_layers = n_layers;
    p->layer_size = calloc(n_layers, sizeof(int));
    memcpy(p->layer_size, layer_size, n_layers * sizeof(int));
    return p;
}

void free_neural_network (neural_network_t *neural_network) {
    free_layer(neural_network->layers, neural_network->n_layers, neural_network->layer_size);
    free_parameter(neural_network->parameters, neural_network->n_layers, neural_network->layer_size);
    free(neural_network->activations);
    free_layer_grad(neural_network->layer_grad, neural_network->n_layers, neural_network->layer_size);
    free_param_grad(neural_network->param_grad, neural_network->n_layers, neural_network->layer_size);
    free(neural_network->layer_size);
    
    free(neural_network);
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

void he_initialize (float *weight, int fan_in, int fan_out) {
    float std = sqrtf(2.0f / (float)fan_in);
    for (int i = 0; i < fan_in * fan_out; i++) {
        weight[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        weight[i] *= std * sqrtf(3.0f);
    }
}

void compute_output_delta (float *output_delta, float *output_layer_activation, float *answer_arr, int n_of_arr) {
    for (size_t i = 0; i < n_of_arr; i++)
    {
        output_delta[i] = output_layer_activation[i] - answer_arr[i];
    }
    
}

void compute_hidden_delta (float *output_delta, float *current_delta, float *current_weight, float *backward_pre_activation, int n_of_activation, int n_of_z_delta, activation_t activation) {
    memset(output_delta, 0, n_of_activation * sizeof(float));
    for (size_t i = 0; i < n_of_activation; i++)
    {
        for (size_t j = 0; j < n_of_z_delta; j++)
        {
            output_delta[i] += current_delta[j] * current_weight[j * n_of_activation+ i];
        }
        
    }

    for (size_t i = 0; i < n_of_activation; i++)
    {
        switch (activation)
        {
        case ACTIVATION_RELU:
            if (backward_pre_activation[i] <= 0)
            {
                output_delta[i] = 0.0f;
            }
            break;
    
        case ACTIVATION_LEAKY_RELU:
            if (backward_pre_activation[i] <= 0)
            {
                output_delta[i] *= 0.01f;
            }
            break;
        }
        
    }
    
}

void compute_weight_grad (float *z_delta, float *previous_activation_arr, float *output_arr, int n_of_output, int n_of_input) {
    for (int i = 0; i < n_of_output; i++)
    {
        for (int j = 0; j < n_of_input; j++)
        {
            output_arr[i * n_of_input + j] = z_delta[i] * previous_activation_arr[j];
        }
        
    }
}

void compute_bias_grad (float *output_bias_grad, float *delta, int n_of_arr) {
    memcpy(output_bias_grad, delta, n_of_arr * sizeof(float));
}

void forward_pass (neural_network_t *neural_network, float *input, float *output) {
    memcpy(neural_network->layers[0].activation, input, neural_network->layer_size[0] * sizeof(float));

    for (size_t i = 0; i < neural_network->n_layers - 1; i++)
    {
        matrix_arr_mul(neural_network->layers[i + 1].pre_activation, neural_network->layers[i].activation, neural_network->parameters[i].weight, neural_network->layer_size[i + 1], neural_network->layer_size[i]);
        add_array(neural_network->layers[i + 1].pre_activation, neural_network->parameters[i].bias, neural_network->layer_size[i + 1]);
        switch (neural_network->activations[i])
        {
        case ACTIVATION_RELU:
            relu(neural_network->layers[i + 1].pre_activation, neural_network->layers[i + 1].activation, neural_network->layer_size[i + 1]);
            break;
        
        case ACTIVATION_LEAKY_RELU:
            leaky_relu(neural_network->layers[i + 1].pre_activation, neural_network->layers[i + 1].activation, neural_network->layer_size[i + 1]);
            break;

        case ACTIVATION_SOFTMAX:
            softmax(neural_network->layers[i + 1].pre_activation, neural_network->layers[i + 1].activation, neural_network->layer_size[i + 1]);
            break;
        }

    }
    memcpy(output, neural_network->layers[neural_network->n_layers - 1].activation, neural_network->layer_size[neural_network->n_layers - 1] * sizeof(float));
    
}

void parameter_initialize (neural_network_t *neural_network) {
    for (size_t i = 0; i < neural_network->n_layers - 1; i++)
    {
        he_initialize(neural_network->parameters[i].weight, neural_network->layer_size[i], neural_network->layer_size[i + 1]);
    }
    
}

void backward_pass (neural_network_t *neural_network, float *answer){
    int L = neural_network->n_layers - 1;   //最後のレイヤーのインデックス

    compute_output_delta(neural_network->layer_grad[L - 1].delta, neural_network->layers[L].activation, answer, neural_network->layer_size[L]);
    compute_weight_grad(neural_network->layer_grad[L - 1].delta, neural_network->layers[L - 1].activation, neural_network->param_grad[L - 1].weight_grad, neural_network->layer_size[L], neural_network->layer_size[L - 1]);
    compute_bias_grad(neural_network->param_grad[L - 1].bias_grad, neural_network->layer_grad[L - 1].delta, neural_network->layer_size[L]);


    for (size_t i = L - 1; i >= 1; i--)
    {
        compute_hidden_delta(neural_network->layer_grad[i - 1].delta, neural_network->layer_grad[i].delta, neural_network->parameters[i].weight, neural_network->layers[i].pre_activation, neural_network->layer_size[i], neural_network->layer_size[i + 1], neural_network->activations[i - 1]);
        compute_weight_grad(neural_network->layer_grad[i - 1].delta, neural_network->layers[i - 1].activation, neural_network->param_grad[i - 1].weight_grad, neural_network->layer_size[i], neural_network->layer_size[i - 1]);
        compute_bias_grad(neural_network->param_grad[i - 1].bias_grad, neural_network->layer_grad[i - 1].delta, neural_network->layer_size[i]);
    }

    for (size_t i = 0; i < L; i++)
    {
        for (size_t j = 0; j < neural_network->layer_size[i] * neural_network->layer_size[i + 1]; j++)
        {
            neural_network->param_grad[i].total_weight_grad[j] += neural_network->param_grad[i].weight_grad[j];
        }
        for (size_t j = 0; j < neural_network->layer_size[i + 1]; j++)
        {
            neural_network->param_grad[i].total_bias_grad[j] += neural_network->param_grad[i].bias_grad[j];
        }
        
    }
    
}

void update_param (neural_network_t *neural_network, float learning_rate, float regularization_rate, int batch_size) {
    int L = neural_network->n_layers - 1;
    for (size_t i = 0; i < L; i++)
    {
        for (size_t j = 0; j < neural_network->layer_size[i] * neural_network->layer_size[i + 1]; j++)
        {
            neural_network->param_grad[i].total_weight_grad[j] /= batch_size;
        }
        for (size_t j = 0; j < neural_network->layer_size[i + 1]; j++)
        {
            neural_network->param_grad[i].total_bias_grad[j] /= batch_size;
        }
        
    }

    for (size_t i = 0; i < neural_network->n_layers - 1; i++)
    {
        for (size_t j = 0; j < neural_network->layer_size[i] * neural_network->layer_size[i + 1]; j++)
        {
            neural_network->parameters[i].weight[j] -= learning_rate * (neural_network->param_grad[i].total_weight_grad[j] + regularization_rate * neural_network->parameters[i].weight[j]);
        }
        for (size_t j = 0; j < neural_network->layer_size[i + 1]; j++)
        {
            neural_network->parameters[i].bias[j] -= learning_rate * neural_network->param_grad[i].total_bias_grad[j];
        }
    }
}

void update_param_adam (neural_network_t *neural_network, float lr, float weight_decay, float beta1, float beta2, float eps, int t, int batch_size) {
    int L = neural_network->n_layers - 1;
    for (size_t i = 0; i < L; i++)
    {
        for (size_t j = 0; j < neural_network->layer_size[i] * neural_network->layer_size[i + 1]; j++)
        {
            neural_network->param_grad[i].total_weight_grad[j] /= batch_size;
        }
        for (size_t j = 0; j < neural_network->layer_size[i + 1]; j++)
        {
            neural_network->param_grad[i].total_bias_grad[j] /= batch_size;
        }
        
    }
    
    float bc = lr * sqrtf(1.0f - powf(beta2, t)) / (1.0f - powf(beta1, t));

    for (size_t i = 0; i < neural_network->n_layers - 1; i++)
    {
        for (size_t j = 0; j < neural_network->layer_size[i] * neural_network->layer_size[i + 1]; j++)
        {
            float g = neural_network->param_grad[i].total_weight_grad[j];
            neural_network->parameters[i].m_weight[j] = beta1 * neural_network->parameters[i].m_weight[j] + (1 - beta1) * g;
            neural_network->parameters[i].v_weight[j] = beta2 * neural_network->parameters[i].v_weight[j] + (1 - beta2) * g * g;
            neural_network->parameters[i].weight[j] -= bc * neural_network->parameters[i].m_weight[j] / (sqrtf(neural_network->parameters[i].v_weight[j]) + eps) + lr * weight_decay * neural_network->parameters[i].weight[j];
        }
        for (size_t j = 0; j < neural_network->layer_size[i + 1]; j++)
        {
            float g = neural_network->param_grad[i].total_bias_grad[j];
            neural_network->parameters[i].m_bias[j] = beta1 * neural_network->parameters[i].m_bias[j] + (1 - beta1) * g;
            neural_network->parameters[i].v_bias[j] = beta2 * neural_network->parameters[i].v_bias[j] + (1 - beta2) * g * g;
            neural_network->parameters[i].bias[j] -= bc * neural_network->parameters[i].m_bias[j] / (sqrtf(neural_network->parameters[i].v_bias[j]) + eps);
        }
    }
}

void flush_grad (neural_network_t *neural_network) {
    for (size_t i = 0; i < neural_network->n_layers - 1; i++)
    {
        memset(neural_network->param_grad[i].total_weight_grad, 0, neural_network->layer_size[i] * neural_network->layer_size[i + 1] * sizeof(float));
        memset(neural_network->param_grad[i].total_bias_grad, 0, neural_network->layer_size[i + 1] * sizeof(float));
        
    }

}

//  ↓Generated by DeepSeek V4 Pro
int save_neural_network(const neural_network_t *neural_network, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) {
        return -1;  // エラー処理（必要なら perror など）
    }

    // ネットワーク構造を書き込み
    fwrite(&neural_network->n_layers, sizeof(int), 1, fp);
    fwrite(neural_network->layer_size, sizeof(int), neural_network->n_layers, fp);

    // 各パラメータを書き込み
    for (int i = 0; i < neural_network->n_layers - 1; i++) {
        int weight_count = neural_network->layer_size[i] * neural_network->layer_size[i + 1];
        int bias_count = neural_network->layer_size[i + 1];

        fwrite(neural_network->parameters[i].weight, sizeof(float), weight_count, fp);
        fwrite(neural_network->parameters[i].bias, sizeof(float), bias_count, fp);
    }

    fclose(fp);
    return 0;
}

int load_neural_network(neural_network_t *neural_network, const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        return -1;  // ファイルが開けない
    }

    // ファイルのネットワーク構造を読み取り
    int file_n_layers;
    if (fread(&file_n_layers, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    int *file_layer_size = malloc(file_n_layers * sizeof(int));
    if (file_layer_size == NULL) {
        fclose(fp);
        return -1;
    }
    if (fread(file_layer_size, sizeof(int), file_n_layers, fp) != (size_t)file_n_layers) {
        free(file_layer_size);
        fclose(fp);
        return -1;
    }

    // 現在のネットワークと比較
    if (file_n_layers != neural_network->n_layers) {
        free(file_layer_size);
        fclose(fp);
        return -1;  // 構造不一致
    }
    for (int i = 0; i < file_n_layers; i++) {
        if (file_layer_size[i] != neural_network->layer_size[i]) {
            free(file_layer_size);
            fclose(fp);
            return -1;  // 構造不一致
        }
    }
    free(file_layer_size);

    // パラメータを読み込んで上書き
    for (int i = 0; i < neural_network->n_layers - 1; i++) {
        int weight_count = neural_network->layer_size[i] * neural_network->layer_size[i + 1];
        int bias_count = neural_network->layer_size[i + 1];

        if (fread(neural_network->parameters[i].weight, sizeof(float), weight_count, fp) != (size_t)weight_count) {
            fclose(fp);
            return -1;  // 読み込みエラー
        }
        if (fread(neural_network->parameters[i].bias, sizeof(float), bias_count, fp) != (size_t)bias_count) {
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    return 0;
}
//  ↑Generated by DeepSeek V4 Pro