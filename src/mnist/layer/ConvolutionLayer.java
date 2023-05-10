package mnist.layer;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.imageio.ImageIO;
import mnist.filter.ConvolutionFilter;
import mnist.strategy.ActivationFunction;
import mnist.strategy.Optimization;
import mnist.util.Util;

/**
 * 畳み込み層
 *
 * <pre>
 * 活性化もこの中でする。
 * </pre>
 */
public class ConvolutionLayer extends LayerBase {
    
    /** フィルタ */
    private final List<ConvolutionFilter> filters;
    
    /** 活性化関数 */
    private final ActivationFunction activationFunction;
    
    /** フィルタのサイズ */
    private final int filterSize;
    
    /**
     * コンストラクタ
     *
     * <pre>
     * outputChannel分のフィルタが必要。フィルタのチャネルはinputChannel。
     * 入力画像の縦横のサイズを引数で受けてるけど、レイヤごと受けた方がいいのか？
     * </pre>
     *
     * @param output        出力の枚数=チャネル
     * @param input         入力の枚数=チャネル
     * @param inputWidth    入力画像の幅
     * @param inputHeight   入力画像の高さ
     * @param size          フィルタのサイズ（とりあえずフィルタは正方にしとく）
     */
    public ConvolutionLayer(int outputChannel, int inputChannel, 
            int inputWidth, int inputHeight, int size, ActivationFunction activationFunction, Optimization opt) {
        super(inputChannel, outputChannel, false);
        this.activationFunction = activationFunction;
        this.filterSize = size;
        this.filters = Stream.generate(() -> new ConvolutionFilter(inputChannel, inputWidth, inputHeight, size, activationFunction, opt))
                             .limit(outputChannel)
                             .collect(Collectors.toList());
    }
    
    /**
     * 順伝播
     */
    @Override
    public float[] forward() {
        
        List<float[]> images = new ArrayList<>();
        for (int i = 0; i < this.filters.size(); i++) {
            images.add(this.filters.get(i).fiter(super.input));
        }
        
        return this.joinArray(images);
    }
    
    /**
     * 複数チャネルを結合する
     */
    private float[] joinArray(List<float[]> images) {
        
        // 画像1枚の大きさ
        int size = images.get(0).length;
        
        // 全画像結合
        float[] join = new float[images.size() * size];
        
        // 1つの配列に突っ込む
        for (int i = 0; i < images.size(); i++) {
            System.arraycopy(images.get(i), 0, join, size * i, size);
        }
        
        return join;
    }
    
    /**
     * 前の層への誤差
     */
    @Override
    protected float[] toPrevError(float[] error) {
        
//        // 誤差を出力のチャネルごとに分解
//        List<float[]> splitError = this.splitArray(error);
        
        List<float[]> errors = new ArrayList<>();
        for (int i = 0; i < this.filters.size(); i++) {
            errors.add(this.filters.get(i).toPrevError(super.input, super.output, error, i));
        }
        
        return this.sumArray(errors);
    }
    
    private float[] sumArray(List<float[]> errors) {
        
        int size = super.input.length;
        int len = errors.size();
        
        float[] sum = new float[size];
        for (float[] error : errors) {
            for (int i = 0; i < size; i++) {
                sum[i] += error[i] / len;
            }
        }
        
        return sum;
    }
    
    /**
     * この層での誤差
     */
    @Override
    protected float[] getError(float[] nextError) {
        float[] error = new float[nextError.length];
        for (int i = 0; i < nextError.length; i++) {
            error[i] = nextError[i] * this.activationFunction.df(super.output[i]);
        }
        return error;
    }
    
    /**
     * 更新
     */
    @Override
    protected void update(float[] error) {
        
        List<float[]> splitError = this.splitArray(error);
        
        for (int i = 0; i < this.filters.size(); i++) {
            this.filters.get(i).update(this.input, splitError.get(i));
        }
    }
    
    /**
     * 複数チャネルに分解する
     */
    private List<float[]> splitArray(float[] error) {
        
        // 画像1枚の大きさ
        int size = error.length / this.filters.size();
        
        List<float[]> res = new ArrayList<>();
        
        // 1つの配列に突っ込む
        for (int i = 0; i < this.filters.size(); i++) {
            float[] split = new float[size];
            System.arraycopy(error, size * i, split, 0, size);
            res.add(split);
        }
        
        return res;
    }
    
    /**
     * フィルタを画像として出力する
     * 
     * <pre>
     * 真っ黒になりそうだから1000倍スケーリングしてみる。
     * </pre>
     * 
     * @param path
     * @param extension
     * @throws IOException 
     */
    public void outputFilterImage(String path, String extension) throws IOException {
        for (int i = 0; i < this.filters.size(); i++) {
            BufferedImage img = Util.arrayToImageMono(Util.scaling(this.filters.get(i).getWeight(),1 / 1000), filterSize, filterSize);
            ImageIO.write(img, extension, Paths.get(path + "filter" + i + "." + extension).toFile());
        }
    }
    
//    private float[] scaling(float[] data) {
//        float[] res = new float[data.length];
//        for (int i = 0; i < data.length; i++) {
//            res[i] = data[i] * 1000;
//        }
//        return res;
//    }
}
