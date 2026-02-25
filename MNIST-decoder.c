#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <stdint.h>

// 画像のサイズ定義（MNIST 準拠）
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT)
#define n_of_input_layer 784
#define n_of_first_hidden_layer 512
#define n_of_second_hidden_layer 256
#define n_of_third_hidden_layer 128
#define n_of_output_layer 10

/**
 * BMP ファイルを読み込み、784 個の float 配列に格納する
 * 配列は 0.0 〜 1.0 に正規化され、ニューラルネットワークの入力として ready な状態になります
 * 
 * @param filename 読み込む BMP ファイルのパス
 * @param output 結果を格納する float 配列（事前に IMAGE_SIZE 分確保しておくこと）
 * @return 成功すれば 0、失敗すれば -1
 */
int load_bmp_to_array(const char *filename, float *output) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("エラー：ファイル '%s' を開けません。\n", filename);
        return -1;
    }

    // --- BMP ヘッダーの読み取り ---
    // BMP ファイルは 14 バイトのファイルヘッダーを持っています
    uint8_t file_header[14];
    if (fread(file_header, 1, 14, fp) != 14) {
        printf("エラー：ファイルヘッダーの読み込みに失敗しました。\n");
        fclose(fp);
        return -1;
    }

    // データオフセットの取得（10 バイト目からの 4 バイト）
    // 画像データがファイルの何バイト目から始まっているかを取得します
    uint32_t data_offset = file_header[10] | (file_header[11] << 8) | 
                           (file_header[12] << 16) | (file_header[13] << 24);

    // オフセットまでシーク
    fseek(fp, data_offset, SEEK_SET);

    // --- 画像データの読み取り ---
    // BMP は基本的に「下から上」にデータが格納されています
    // また、1 行のバイト数が 4 の倍数になるようにパディング（詰め物）が入ることがあります
    // 24 ビット BMP の場合、1 ピクセル 3 バイト (B, G, R) です
    
    int row_bytes = IMAGE_WIDTH * 3;
    int padding = (4 - (row_bytes % 4)) % 4; // 行のパディングサイズ計算

    // 一時バッファ（1 行分 + パディング）
    uint8_t *row_buffer = (uint8_t *)malloc(row_bytes + padding);
    if (row_buffer == NULL) {
        printf("エラー：メモリの確保に失敗しました。\n");
        fclose(fp);
        return -1;
    }

    // 配列への格納インデックス
    // MNIST 形式は「上から下、左から右」なので、BMP の縦方向を反転して読み込みます
    for (int y = IMAGE_HEIGHT - 1; y >= 0; y--) {
        // 1 行読み込み（パディング含む）
        if (fread(row_buffer, 1, row_bytes + padding, fp) != (size_t)(row_bytes + padding)) {
            printf("エラー：画像データの読み込みに失敗しました。\n");
            free(row_buffer);
            fclose(fp);
            return -1;
        }

        for (int x = 0; x < IMAGE_WIDTH; x++) {
            // 24 ビット BMP は B, G, R の順で格納されています
            uint8_t b = row_buffer[x * 3 + 0];
            uint8_t g = row_buffer[x * 3 + 1];
            uint8_t r = row_buffer[x * 3 + 2];

            // グレースケール変換（単純平均）
            // 厳密には重み付けしますが、MNIST 風であれば平均で十分です
            float gray = (r + g + b) / 3.0f;

            // 0.0 〜 1.0 に正規化
            float normalized = gray / 255.0f;

            // 配列に格納（行優先）
            output[y * IMAGE_WIDTH + x] = normalized;
        }
    }

    free(row_buffer);
    fclose(fp);
    return 0;
}

int argmax(const float *array, int size) {
    // 配列が空の場合のエラー処理
    if (size <= 0) {
        return -1;
    }

    int max_index = 0;
    float max_value = array[0];

    // 2 番目の要素から最後まで比較
    for (int i = 1; i < size; i++) {
        if (array[i] > max_value) {
            max_value = array[i];
            max_index = i;
        }
    }

    return max_index;
}

void add_bias (float *operated_arr, float *input_bias, int n_of_arr){
    for (int i = 0; i < n_of_arr; i++)
    {
        operated_arr[i] = operated_arr[i] + input_bias[i];
    }
    
}

int mmul (float *output_arr, float *input_arr, float *matrix, int n_of_output_arr, int n_of_input_arr){
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

    return 0;
}

float larger (float input_1, float input_2){
    if (input_1 > input_2) return input_1;
    else return input_2;
    
}

void relu(float *input_arr, float *output_arr, int n_of_arr){
    for (int i = 0; i < n_of_arr; i++)
    {
        output_arr[i] = larger(input_arr[i], 0.0f);
    }
    
}

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

void softmax (float *input_arr, float *output_arr, int n_of_arr){
    float max = 0.0, sum = 0.0;
    float *tmp = malloc(n_of_arr * sizeof(float));
    max = extract_max(input_arr, n_of_arr);
    for (int i = 0; i < n_of_arr; i++)
    {
        tmp[i] = exp(input_arr[i] - max);
        sum += tmp[i];
    }
    for (int j = 0; j < n_of_arr; j++)
    {
        output_arr[j] = tmp[j] / sum;
    }
    free(tmp);
}

int main(int argc, char *argv[]) {
    float *input_layer;
    float *first_hidden_layer;
    float *second_hidden_layer;
    float *third_hidden_layer;
    float *output_layer;
    float *weight_to_first_hidden_layer;
    float *bias_of_first_hidden_layer;
    float *weight_to_second_hidden_layer;
    float *bias_of_second_hidden_layer;
    float *weight_to_third_hidden_layer;
    float *bias_of_third_hidden_layer;
    float *weight_to_output_layer;
    float *bias_of_output_layer;
    float *z1;
    float *z2;
    float *z3;
    float *zout;
    float *delta_4, *delta_3, *delta_2, *delta_1;
    float *grad_to_w4, *grad_to_b4, *grad_to_w3, *grad_to_b3, *grad_to_w2, *grad_to_b2, *grad_to_w1, *grad_to_b1;
    float *grad_to_w4t, *grad_to_b4t, *grad_to_w3t, *grad_to_b3t, *grad_to_w2t, *grad_to_b2t, *grad_to_w1t, *grad_to_b1t;
    input_layer = (float*)malloc(n_of_input_layer * sizeof(float));
    first_hidden_layer = (float*)malloc(n_of_first_hidden_layer * sizeof(float));
    second_hidden_layer = (float*)malloc(n_of_second_hidden_layer * sizeof(float));
    third_hidden_layer = (float*)malloc(n_of_third_hidden_layer * sizeof(float));
    output_layer = (float*)malloc(n_of_output_layer * sizeof(float));
    weight_to_first_hidden_layer = (float*)malloc(n_of_first_hidden_layer * n_of_input_layer * sizeof(float));
    bias_of_first_hidden_layer = (float*)malloc(n_of_first_hidden_layer * sizeof(float));
    weight_to_second_hidden_layer = (float*)malloc(n_of_second_hidden_layer * n_of_first_hidden_layer * sizeof(float));
    bias_of_second_hidden_layer = (float*)malloc(n_of_second_hidden_layer * sizeof(float));
    weight_to_third_hidden_layer = (float*)malloc(n_of_third_hidden_layer * n_of_second_hidden_layer * sizeof(float));
    bias_of_third_hidden_layer = (float*)malloc(n_of_third_hidden_layer * sizeof(float));
    weight_to_output_layer = (float*)malloc(n_of_output_layer * n_of_third_hidden_layer * sizeof(float));
    bias_of_output_layer = (float*)malloc(n_of_output_layer * sizeof(float));
    z1 = (float*)malloc(n_of_first_hidden_layer * sizeof(float));
    z2 = (float*)malloc(n_of_second_hidden_layer * sizeof(float));
    z3 = (float*)malloc(n_of_third_hidden_layer * sizeof(float));
    zout = (float*)malloc(n_of_output_layer * sizeof(float));
    
    if (argc < 2) {
        printf("使用方法：./bmp_converter [bmp ファイル名]\n");
        return 1;
    }

    const char *filename = argv[1];
    float *input_data = (float *)malloc(sizeof(float) * IMAGE_SIZE);

    if (input_data == NULL) {
        printf("エラー：メモリの確保に失敗しました。\n");
        return 1;
    }

    printf("ファイル '%s' を読み込み中...\n", filename);
    if (load_bmp_to_array(filename, input_data) == 0) {
        printf("成功！784 個の配列に変換しました。\n");
        
        // デモンストレーション用に最初の 10 個の値を表示
        printf("配列の先頭 10 個の値 (0.0-1.0):\n");
        for (int i = 0; i < 10; i++) {
            printf("%.4f ", input_data[i]);
        }
        printf("\n");

        // ここでニューラルネットワークの推論関数などを呼び出します
        // example: neural_network_predict(input_data);

        FILE *weight;
        weight = fopen("weight", "r");
        if (weight == NULL)
        {
            printf("重みがありません\n");
            return 0;
        }
    for (size_t i = 0; i < n_of_input_layer * n_of_first_hidden_layer; i++)
    {
        fscanf(weight, "%f\n", weight_to_first_hidden_layer[i]);
    }
    for (size_t i = 0; i < n_of_first_hidden_layer * n_of_second_hidden_layer; i++)
    {
        fscanf(weight, "%f\n", weight_to_second_hidden_layer[i]);
    }
    for (size_t i = 0; i < n_of_second_hidden_layer * n_of_third_hidden_layer; i++)
    {
        fscanf(weight, "%f\n", weight_to_third_hidden_layer[i]);
    }
    for (size_t i = 0; i < n_of_third_hidden_layer * n_of_output_layer; i++)
    {
        fscanf(weight, "%f\n", weight_to_output_layer[i]);
    }
    for (size_t i = 0; i < n_of_first_hidden_layer; i++)
    {
        fscanf(weight,"%f\n", bias_of_first_hidden_layer[i]);
    }
    for (size_t i = 0; i < n_of_second_hidden_layer; i++)
    {
        fscanf(weight,"%f\n", bias_of_second_hidden_layer[i]);
    }
    for (size_t i = 0; i < n_of_third_hidden_layer; i++)
    {
        fscanf(weight,"%f\n", bias_of_third_hidden_layer[i]);
    }
    for (size_t i = 0; i < n_of_output_layer; i++)
    {
        fscanf(weight,"%f\n", bias_of_output_layer[i]);
    }
    fclose(weight);
        
        mmul(z1, input_data, weight_to_first_hidden_layer, n_of_first_hidden_layer, n_of_input_layer);
        add_bias(z1, bias_of_first_hidden_layer, n_of_first_hidden_layer);
        relu(z1, first_hidden_layer, n_of_first_hidden_layer);

        mmul(z2, first_hidden_layer, weight_to_second_hidden_layer, n_of_second_hidden_layer, n_of_first_hidden_layer);
        add_bias(z2, bias_of_second_hidden_layer, n_of_second_hidden_layer);
        relu(z2, second_hidden_layer, n_of_second_hidden_layer);

        mmul(z3, second_hidden_layer, weight_to_third_hidden_layer, n_of_third_hidden_layer, n_of_second_hidden_layer);
        add_bias(z3, bias_of_third_hidden_layer, n_of_third_hidden_layer);
        relu(z3, third_hidden_layer, n_of_third_hidden_layer);

        mmul(zout, third_hidden_layer, weight_to_output_layer, n_of_output_layer, n_of_third_hidden_layer);
        add_bias(zout, bias_of_output_layer, n_of_output_layer);
        softmax(zout, output_layer, n_of_output_layer);

        printf("%d\n", argmax(output_layer, n_of_output_layer));

    } else {
        printf("変換に失敗しました。\n");
        free(input_data);
        return 1;
    }

    free(input_data);
    free(input_layer);
    free(first_hidden_layer);
    free(second_hidden_layer);
    free(third_hidden_layer);
    free(output_layer);
    free(weight_to_first_hidden_layer);
    free(bias_of_first_hidden_layer);
    free(weight_to_second_hidden_layer);
    free(bias_of_second_hidden_layer);
    free(weight_to_third_hidden_layer);
    free(bias_of_third_hidden_layer);
    free(weight_to_output_layer);
    free(bias_of_output_layer);
    free(z1);
    free(z2);
    free(z3);
    free(zout);
    return 0;
}