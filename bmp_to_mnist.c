#include "bmp_to_mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// BMPファイルヘッダー構造体
#pragma pack(push, 1)
typedef struct {
    uint16_t bfType;      // ファイルタイプ ('BM')
    uint32_t bfSize;      // ファイルサイズ
    uint16_t bfReserved1; // 予約領域
    uint16_t bfReserved2; // 予約領域
    uint32_t bfOffBits;   // 画像データまでのオフセット
} BITMAPFILEHEADER;

// BMP情報ヘッダー構造体
typedef struct {
    uint32_t biSize;          // この構造体のサイズ
    int32_t  biWidth;         // 画像の幅
    int32_t  biHeight;        // 画像の高さ
    uint16_t biPlanes;        // プレーン数
    uint16_t biBitCount;      // 1ピクセルあたりのビット数
    uint32_t biCompression;   // 圧縮形式
    uint32_t biSizeImage;     // 画像データのサイズ
    int32_t  biXPelsPerMeter; // 水平解像度
    int32_t  biYPelsPerMeter; // 垂直解像度
    uint32_t biClrUsed;       // 使用する色数
    uint32_t biClrImportant;  // 重要な色数
} BITMAPINFOHEADER;
#pragma pack(pop)

int bmp_to_mnist_array(const char* filename, uint8_t* output_array) {
    if (filename == NULL || output_array == NULL) {
        fprintf(stderr, "エラー: NULLポインタが渡されました\n");
        return -1;
    }

    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "エラー: ファイル '%s' を開けません\n", filename);
        return -1;
    }

    // BMPファイルヘッダー読み込み
    BITMAPFILEHEADER fileHeader;
    if (fread(&fileHeader, sizeof(BITMAPFILEHEADER), 1, fp) != 1) {
        fprintf(stderr, "エラー: ファイルヘッダーの読み込みに失敗しました\n");
        fclose(fp);
        return -1;
    }
    
    if (fileHeader.bfType != 0x4D42) { // 'BM'
        fprintf(stderr, "エラー: BMPファイルではありません\n");
        fclose(fp);
        return -1;
    }

    // BMP情報ヘッダー読み込み
    BITMAPINFOHEADER infoHeader;
    if (fread(&infoHeader, sizeof(BITMAPINFOHEADER), 1, fp) != 1) {
        fprintf(stderr, "エラー: 情報ヘッダーの読み込みに失敗しました\n");
        fclose(fp);
        return -1;
    }

    // 画像サイズの確認
    if (infoHeader.biWidth != MNIST_IMAGE_SIZE || 
        abs(infoHeader.biHeight) != MNIST_IMAGE_SIZE) {
        fprintf(stderr, "エラー: 画像サイズは%dx%dである必要があります (実際: %dx%d)\n",
                MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE,
                infoHeader.biWidth, abs(infoHeader.biHeight));
        fclose(fp);
        return -1;
    }

    // 画像データの位置へ移動
    fseek(fp, fileHeader.bfOffBits, SEEK_SET);

    // 1行のバイト数を計算(4バイトアラインメント)
    int bytes_per_pixel = infoHeader.biBitCount / 8;
    int row_size = ((infoHeader.biWidth * bytes_per_pixel + 3) / 4) * 4;
    
    // 画像データの読み込み用バッファ
    uint8_t* row_buffer = (uint8_t*)malloc(row_size);
    if (row_buffer == NULL) {
        fprintf(stderr, "エラー: メモリ確保に失敗しました\n");
        fclose(fp);
        return -1;
    }

    // BMPは下から上に格納されているので、上下反転して読み込む
    int height_direction = (infoHeader.biHeight > 0) ? -1 : 1;
    int start_row = (infoHeader.biHeight > 0) ? (MNIST_IMAGE_SIZE - 1) : 0;

    for (int y = 0; y < MNIST_IMAGE_SIZE; y++) {
        int actual_row = start_row + (y * height_direction);
        
        if (fread(row_buffer, 1, row_size, fp) != (size_t)row_size) {
            fprintf(stderr, "エラー: 画像データの読み込みに失敗しました\n");
            free(row_buffer);
            fclose(fp);
            return -1;
        }

        for (int x = 0; x < MNIST_IMAGE_SIZE; x++) {
            uint8_t brightness;
            
            if (bytes_per_pixel == 1) {
                // グレースケール(8bit)
                brightness = row_buffer[x];
            } else if (bytes_per_pixel == 3) {
                // カラー(24bit BGR)
                int pixel_offset = x * 3;
                uint8_t b = row_buffer[pixel_offset];
                uint8_t g = row_buffer[pixel_offset + 1];
                uint8_t r = row_buffer[pixel_offset + 2];
                // RGB to グレースケール変換 (NTSC係数使用)
                brightness = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
            } else if (bytes_per_pixel == 4) {
                // カラー(32bit BGRA)
                int pixel_offset = x * 4;
                uint8_t b = row_buffer[pixel_offset];
                uint8_t g = row_buffer[pixel_offset + 1];
                uint8_t r = row_buffer[pixel_offset + 2];
                brightness = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
            } else {
                fprintf(stderr, "エラー: 未対応のビット深度です (%dbit)\n", 
                        infoHeader.biBitCount);
                free(row_buffer);
                fclose(fp);
                return -1;
            }
            
            // MNISTスタイル: 左上から右下へ (row-major order)
            output_array[actual_row * MNIST_IMAGE_SIZE + x] = brightness;
        }
    }

    free(row_buffer);
    fclose(fp);
    return 0;
}

int save_mnist_array(const char* filename, const uint8_t* array) {
    if (filename == NULL || array == NULL) {
        fprintf(stderr, "エラー: NULLポインタが渡されました\n");
        return -1;
    }

    FILE* fp = fopen(filename, "wb");
    if (fp == NULL) {
        fprintf(stderr, "エラー: ファイル '%s' を開けません\n", filename);
        return -1;
    }

    size_t written = fwrite(array, sizeof(uint8_t), MNIST_ARRAY_SIZE, fp);
    fclose(fp);

    if (written != MNIST_ARRAY_SIZE) {
        fprintf(stderr, "エラー: データの書き込みに失敗しました\n");
        return -1;
    }

    return 0;
}

int load_mnist_array(const char* filename, uint8_t* array) {
    if (filename == NULL || array == NULL) {
        fprintf(stderr, "エラー: NULLポインタが渡されました\n");
        return -1;
    }

    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "エラー: ファイル '%s' を開けません\n", filename);
        return -1;
    }

    size_t read = fread(array, sizeof(uint8_t), MNIST_ARRAY_SIZE, fp);
    fclose(fp);

    if (read != MNIST_ARRAY_SIZE) {
        fprintf(stderr, "エラー: データの読み込みに失敗しました\n");
        return -1;
    }

    return 0;
}

void print_mnist_array(const uint8_t* array) {
    if (array == NULL) {
        fprintf(stderr, "エラー: NULLポインタが渡されました\n");
        return;
    }

    printf("MNIST形式ピクセル配列 (%dx%d = %d要素):\n", 
           MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, MNIST_ARRAY_SIZE);
    for (int i = 0; i < MNIST_ARRAY_SIZE; i++) {
        printf("%3d", array[i]);
        if ((i + 1) % MNIST_IMAGE_SIZE == 0) {
            printf("\n");
        } else {
            printf(" ");
        }
    }
    printf("\n");
}

void visualize_mnist_array(const uint8_t* array) {
    if (array == NULL) {
        fprintf(stderr, "エラー: NULLポインタが渡されました\n");
        return;
    }

    printf("視覚化 (ASCII):\n");
    const char* chars = " .':-=+*#%@";
    for (int y = 0; y < MNIST_IMAGE_SIZE; y++) {
        for (int x = 0; x < MNIST_IMAGE_SIZE; x++) {
            int brightness = array[y * MNIST_IMAGE_SIZE + x];
            int char_index = (brightness * 10) / 255;
            printf("%c", chars[char_index]);
        }
        printf("\n");
    }
}

void mnist_array_stats(const uint8_t* array, int* min_val, int* max_val, float* avg_val) {
    if (array == NULL) {
        fprintf(stderr, "エラー: NULLポインタが渡されました\n");
        return;
    }

    int sum = 0;
    int min = 255, max = 0;
    
    for (int i = 0; i < MNIST_ARRAY_SIZE; i++) {
        sum += array[i];
        if (array[i] < min) min = array[i];
        if (array[i] > max) max = array[i];
    }
    
    if (min_val != NULL) *min_val = min;
    if (max_val != NULL) *max_val = max;
    if (avg_val != NULL) *avg_val = sum / (float)MNIST_ARRAY_SIZE;
}

uint8_t get_pixel(const uint8_t* array, int x, int y) {
    if (array == NULL || x < 0 || x >= MNIST_IMAGE_SIZE || 
        y < 0 || y >= MNIST_IMAGE_SIZE) {
        return 0;
    }
    return array[y * MNIST_IMAGE_SIZE + x];
}

void set_pixel(uint8_t* array, int x, int y, uint8_t value) {
    if (array == NULL || x < 0 || x >= MNIST_IMAGE_SIZE || 
        y < 0 || y >= MNIST_IMAGE_SIZE) {
        return;
    }
    array[y * MNIST_IMAGE_SIZE + x] = value;
}
