package mnist.layer;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import mnist.strategy.ActivationFunction;
import mnist.strategy.Optimization;

/**
 * 全結合層
 * 
 * <pre>
 * ただの多層パーセプトロンなら隠し層も出力層もこれでいい。
 * </pre>
 */
public class FullyConnectedLayer extends LayerBase {
    
    /** このレイヤのニューロン */
    protected final List<Neuron> neurons;
    
    /** 活性化関数 */
    protected final ActivationFunction activationFunction;
    
    /**
     * コンストラクタ
     *
     * @param size          このレイヤのニューロンの数
     * @param inputSize     前のレイヤからの入力の数
     * @param activationFunction 活性化関数
     */
    public FullyConnectedLayer(int size, int inputSize, ActivationFunction activationFunction, Optimization opt) {
        
        super(inputSize, size, true);
        
        // ニューロンを作成
        this.neurons = Stream.generate(() -> new Neuron(inputSize + 1, opt))
                             .limit(size)
                             .collect(Collectors.toList());
        
        // 活性化関数を設定
        this.activationFunction = activationFunction;
    }
    
    /**
     * 順伝播
     *
     * @return 次の層への出力
     */
    @Override
    public float[] forward() {
        float[] res = new float[this.neurons.size()];
        for (int i = 0; i < this.neurons.size(); i++) {
            res[i] = this.activationFunction.f(this.neurons.get(i).output(super.input));
        }
        return res;
    }
    
    /**
     * この層の更新
     * 
     * @param error この層での誤差 
     */
    @Override
    protected void update(float[] error) {
        for (int i = 0; i < this.neurons.size(); i++) {
            this.neurons.get(i).update(this.input, error[i]);
        }
    }
    
    /**
     * 前のレイヤ誤差に変換する
     * 
     * @param error この層での誤差 
     * @return 
     */
    @Override
    protected float[] toPrevError(float[] error) {
        
        // バイアスが末尾にある前提で回しているので、変えると死ぬかも
        float[] prevError = new float[this.input.length - 1];
        for (int i = 0; i < this.input.length - 1; i++) {
            float sumError = 0f;
            for (int j = 0; j < error.length; j++) {
                sumError += error[j] * this.neurons.get(j).weight[i];
            }
            prevError[i] = sumError;
        }
        
        return prevError;
    }
    
    /**
     * この層の誤差を取得
     *
     * @param nextError     次の層の誤差
     * @return 出力層の誤差（次元は出力層のユニット数）
     */
    @Override
    protected float[] getError(float[] nextError) {
        float[] error = new float[nextError.length];
        for (int i = 0; i < nextError.length; i++) {
            error[i] = nextError[i] * this.activationFunction.df(super.output[i]);
        }
        
        return error;
    }
    
//    @Override
//    public String toString() {
//        return this.neurons.stream().map(Neuron::toString).collect(Collectors.joining("\n"));
//    }
}
