package mnist.layer;

import mnist.strategy.Optimization;

/**
 * ニューロンのモデル
 */
public class Neuron {
    
    /** 重みベクトル */
    final float[] weight;
    
    /** 最適化（とりあえず確率的勾配法にしとく） */
    private final Optimization opt;
    
    /**
     * コンストラクタ
     *
     * @param inputSize 入力の次元（前のレイヤの出力数）
     */
    Neuron(int inputSize, Optimization opt) {
        this.weight = new float[inputSize];
        for (int i = 0; i < inputSize; i++) {
            this.weight[i] = (float) (Math.random() * 2 - 1);
        }
        this.opt = opt.get();;
    }
    
    /**
     * 出力
     *
     * <pre>
     * o = f(ωx) = f(u)
     * </pre>
     *
     * @param input 入力データ
     * @return      出力値
     */
    public float output(float[] input) {
        return this.dot(input);
    }
    
    /**
     * 重みベクトルを更新
     *
     * <pre>
     * ω = ω - ρ・(∂E/∂ω) = ω - ρ・(o(L) - t)・f'(u)・o(L-1)
     * ∂E/∂uを誤差とか呼んでいいの？
     * </pre>
     *
     * @param input このニューロンへの入力（＝ひとつ前の層の出力）
     * @param error 誤差（の出力における偏微分）
     */
    public void update(float[] input, float error) {
        for (int i = 0; i < this.weight.length; i++) {
            float gw = error * input[i];
            this.weight[i] -= this.opt.getDeltaWeight(gw, i);
        }
    }
    
    /**
     * 内積の計算
     *
     * @param data  入力データ
     * @return      出力値
     */
    private float dot(float[] data) {
        float res = 0;
        for (int i = 0; i < data.length; i++) {
            res += data[i] * this.weight[i];
        }
        return res;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (float w : this.weight) {
            sb.append(w).append(" ");
        }
        return sb.toString();
    }
}
