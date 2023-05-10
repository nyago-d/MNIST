package mnist.filter;

import java.util.stream.Collectors;
import java.util.stream.IntStream;
import mnist.strategy.ActivationFunction;
import mnist.strategy.Optimization;
import mnist.strategy.Optimization.AdaDelta;
import mnist.strategy.Optimization.SGD;

/**
 * 畳み込み層用フィルタ
 */
public class ConvolutionFilter extends Filter {
    
    /** 重みベクトル */
    private final float[] weight;
    
    /** バイアス */
    private float bias = 0.01f;
    
    /** 活性化関数 */
    private final ActivationFunction activationFunc;
    
    /** 最適化 */
    private final Optimization opt;
    
    /**
     * コンストラクタ
     *
     * @param inputChannel
     * @param inputWidth
     * @param inputHeight
     * @param filterSize
     * @param activationFunc
     */
    public ConvolutionFilter(int inputChannel, int inputWidth, int inputHeight, int filterSize, ActivationFunction activationFunc, Optimization opt) {
        super(inputChannel, inputWidth, inputHeight, filterSize, 1, true);  // ストライドはとりあえず1固定
        this.activationFunc = activationFunc;
        this.weight = new float[inputChannel * filterSize * filterSize];
        this.opt = opt.get();
        for (int i = 0; i < this.weight.length; i++) {
            this.weight[i] = (float) (Math.random() * 0.01);
        }
    }
    
    /**
     * フィルタする
     *
     * @param input     入力
     * @param channel   入力のチャネル
     * @param pos       フィルタのクリップ位置
     */
    @Override
    protected float fiter(float[] input, int channel, int[] pos) {

        float sum = 0;

        for (int j = 0; j < super.filterSize; j++) {
            for (int k = 0; k < super.filterSize; k++) {
                sum += this.weight[channel * super.filterSize * super.filterSize + j * super.filterSize + k]
                            * input[super.inputImageSize * channel + super.toIndex(pos, j, k)];
            }
        }

        return this.activationFunc.f(sum + bias);
    }
    
    /**
     * 更新する
     * 
     * @param input
     * @param channel
     * @param pos
     * @param error
     * @param len 
     */
    @Override
    protected void update(float[] input, int channel, int[] pos, float error, int len) {
        
        // フィルタの重みを更新（同じ重みに何度も更新がかかる…？）
        for (int j = 0; j < super.filterSize; j++) {
            for (int k = 0; k < super.filterSize; k++) {
                int idx = j * super.filterSize + k;
                float dw = input[super.inputImageSize * channel + super.toIndex(pos, j, k)] * error;
                this.weight[idx] -= this.opt.getDeltaWeight(dw, idx) / len;
            }
        }
        
        // バイアスを更新
        this.bias -= this.opt.getDeltaWeight(error, super.filterSize * super.filterSize) / len;
    }
    
    /**
     * 前の層への誤差を設定する
     * 
     * @param input
     * @param channel
     * @param pos
     * @param output
     * @param error
     * @param prevError 
     */
    @Override
    protected void setPrevError(float[] input, int inChannel, int[] pos, float output, float error, float[] prevError) {
        
        for (int j = 0; j < super.filterSize; j++) {
            for (int k = 0; k < super.filterSize; k++) {
                
                // 重みのインデックス
                int weightIdx = super.filterSize * j + k;
                
                // 前の層へ伝播する誤差のインデックス
                int errorIdx = super.inputImageSize * inChannel + super.toIndex(pos, j, k);
                
                // 誤差に重みを掛けて逆伝播する
                prevError[errorIdx] += this.weight[weightIdx] * error;
            }
        }
    }
    
    public float[] getWeight() {
        return weight;
    }
    
    @Override
    public String toString() {
        return IntStream.range(0, weight.length).mapToObj(j -> String.valueOf(weight[j])).collect(Collectors.joining(" "));
    }
}
