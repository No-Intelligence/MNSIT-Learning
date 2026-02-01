#include <math.h>

float extract_max (float *input_array, int n_of_input_arr){
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

/**
 * 行列ベクトル積:
 *   - 行列とベクトルの積を求めます
 * @param output_arr 結果を出力する配列
 * @param input_arr 入力ベクトルの配列
 * @param matrix 入力行列の配列
 * @param n_of_output_arr 出力配列の要素数
 * @param n_of_input_arr 入力配列の要素数
 */
void mmul (float *output_arr, float *input_arr, float *matrix, int n_of_output_arr, int n_of_input_arr){
    for (int i = 0; i < n_of_output_arr; i++)
    {
        output_arr[i] = 0.0;
    }
    
    for (int i = 0; i < n_of_output_arr; i++)
    {
        for (int j = 0; j < n_of_input_arr; j++)
        {
            output_arr[i] += matrix[n_of_input_arr * i + j] * input_arr[j];
        }
        
    }
}

/**
 * バイアス加算関数:
 *   - 配列で入力した配列とバイアスを加算します
 * @param operated_arr バイアスを加算する配列（入出力共有）
 * @param bias バイアスの配列
 * @param n_of_arr 配列の要素数
 */
void add_bias (float *operated_arr, float *bias, int n_of_arr){
    for (int i = 0; i < n_of_arr; i++)
    {
        operated_arr[i] = operated_arr[i] + bias[i];
    }
    
}

/**
 * ReLU活性化関数
 * @param input_arr ReLU関数を適用する配列
 * @param output_arr 結果を出力する配列
 * @param n_of_arr 配列の要素数
 */
void relu(float *input_arr, float *output_arr, int n_of_arr){
    for (int i = 0; i < n_of_arr; i++)
    {
        if (input_arr[i] > 0.0f)
        {
            output_arr[i] = input_arr[i];
        }
        else
        {
            output_arr[i] = 0.0f;
        }
    }
    
}

/**
 * Leaky ReLU活性化関数
 * @param input_arr Leaky ReLU関数を適用する配列
 * @param output_arr 結果を出力する配列
 * @param n_of_arr 配列の要素数
 */
void leaky_relu(float* input_arr, float* output_arr, int n_of_arr){
    for (int i = 0; i < n_of_arr; i++)
    {
        if (input_arr[i] > 0.0f)
        {
            output_arr[i] = input_arr[i];
        }
        else
        {
            output_arr[i] = 0.01 * input_arr[i];
        }
    }
}

/**
 * Softmax活性化関数
 * @param input_arr Softmax関数を適用する配列
 * @param output_arr 結果を出力する配列
 * @param n_of_arr 配列の要素数
 */
void softmax (float *input_arr, float *output_arr, int n_of_arr){
    float max = 0.0, sum = 0.0;
    max = extract_max(input_arr, n_of_arr);
    for (int i = 0; i < n_of_arr; i++)
    {
        input_arr[i] = exp(input_arr[i] - max);
        sum += input_arr[i];
    }
    for (int j = 0; j < n_of_arr; j++)
    {
        output_arr[j] = input_arr[j] / sum;
    }    
}

int find_max_index (float *arr, int size){
    if(size <= 0){
        return -1;
    }
    int max_index = 0;
    float max_value = arr[0];

    for (int i = 0; i < size; i++)
    {
        if (arr[i] > max_value)
        {
            max_value = arr[i];
            max_index = i;
        }
        
    }
    
    return max_index;
}