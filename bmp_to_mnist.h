#ifndef BMP_TO_MNIST_H
#define BMP_TO_MNIST_H

#include <stdint.h>

// 定数定義
#define MNIST_IMAGE_SIZE 28
#define MNIST_ARRAY_SIZE (MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE)  // 784

/**
 * BMPファイルを読み込み、MNISTスタイルのピクセル配列に変換
 * 
 * @param filename 入力BMPファイル名
 * @param output_array 出力配列 (MNIST_ARRAY_SIZE = 784要素必要)
 *                     各要素は0-255の明るさ値
 *                     左上から右下へ、行優先順(row-major order)で格納
 * @return 成功時0、失敗時-1
 * 
 * 対応フォーマット:
 *   - 8bit グレースケールBMP
 *   - 24bit カラーBMP (BGR)
 *   - 32bit カラーBMP (BGRA)
 * 
 * 注意:
 *   - 画像サイズは必ず28×28である必要があります
 *   - カラー画像は自動的にグレースケールに変換されます(NTSC係数使用)
 *   - 配列のインデックスは array[y * 28 + x] で計算されます
 */
int bmp_to_mnist_array(const char* filename, uint8_t* output_array);

/**
 * MNIST形式の配列をバイナリファイルに保存
 * 
 * @param filename 出力ファイル名
 * @param array MNIST形式の配列 (784要素)
 * @return 成功時0、失敗時-1
 */
int save_mnist_array(const char* filename, const uint8_t* array);

/**
 * MNIST形式の配列をバイナリファイルから読み込み
 * 
 * @param filename 入力ファイル名
 * @param array 読み込み先の配列 (784要素必要)
 * @return 成功時0、失敗時-1
 */
int load_mnist_array(const char* filename, uint8_t* array);

/**
 * MNIST形式の配列を標準出力に表示
 * 
 * @param array MNIST形式の配列 (784要素)
 */
void print_mnist_array(const uint8_t* array);

/**
 * MNIST形式の配列をASCIIアートとして視覚化
 * 
 * @param array MNIST形式の配列 (784要素)
 */
void visualize_mnist_array(const uint8_t* array);

/**
 * MNIST形式の配列の統計情報を計算
 * 
 * @param array MNIST形式の配列 (784要素)
 * @param min_val 最小値の格納先 (NULLも可)
 * @param max_val 最大値の格納先 (NULLも可)
 * @param avg_val 平均値の格納先 (NULLも可)
 */
void mnist_array_stats(const uint8_t* array, int* min_val, int* max_val, float* avg_val);

/**
 * 特定のピクセルの明るさ値を取得
 * 
 * @param array MNIST形式の配列 (784要素)
 * @param x X座標 (0-27)
 * @param y Y座標 (0-27)
 * @return ピクセルの明るさ値 (0-255)、範囲外の場合は0
 */
uint8_t get_pixel(const uint8_t* array, int x, int y);

/**
 * 特定のピクセルの明るさ値を設定
 * 
 * @param array MNIST形式の配列 (784要素)
 * @param x X座標 (0-27)
 * @param y Y座標 (0-27)
 * @param value 設定する明るさ値 (0-255)
 */
void set_pixel(uint8_t* array, int x, int y, uint8_t value);

#endif // BMP_TO_MNIST_H
