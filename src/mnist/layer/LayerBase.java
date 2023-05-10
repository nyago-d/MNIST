package mnist.layer;

/**
 * レイヤの基底
 * 
 * <pre>
 * バイアスの考え方は、この層にバイアスユニットが存在するかではなく、
 * 入力時にバイアスを追加するかという発想で考える。
 * </pre>
 */
public abstract class LayerBase implements Layer {
    
    /** この層への入力の次元 */
    private final int inSize;
    
    /** この層からの出力の次元 */
    private final int outSize;
    
    /** この層を計算するときにバイアスを含めるか */
    private final boolean addBias;
    
    /** この層への入力 */
    protected float[] input;
    
    /** この層からの出力 */
    protected float[] output;
    
    /** この層の誤差 */
    private float[] error;
    
    /**
     * コンストラクタ
     * 
     * @param inSize
     * @param outSize 
     * @param addBias 
     */
    LayerBase(int inSize, int outSize, boolean addBias) {
        this.inSize = inSize;
        this.outSize = outSize;
        this.addBias = addBias;
    }
    
    /**
     * 順伝播
     *
     * @param input この層への入力
     * @return この層からの出力
     */
    @Override
    public final float[] forward(float[] input) {
        this.input = this.addBias ? this.addBias(input) : input;
        this.output = this.forward();
        return this.output;
    }
    
    /**
     * 順伝播
     * 
     * @return この層からの出力
     */
    protected abstract float[] forward();
    
    /**
     * 逆伝播
     * 
     * @param nextError 次の層からの誤差
     * @return 前の層への誤差
     */
    @Override
    public final float[] backward(float[] nextError) {
        this.error = this.getError(nextError);
//        this.update(error); // ひょっとしたらこれは全層逆伝播してからじゃないとだめかも
        return this.toPrevError(this.error);
    }
    
    /**
     * この層の誤差を計算
     * 
     * @param nextError 次の層の誤差
     * @return この層の誤差
     */
    protected abstract float[] getError(float[] nextError);
    
    /**
     * この層の更新
     * 
     * @param error この層の誤差 
     */
    protected abstract void update(float[] error);
    
    /**
     * この層の誤差を前の層の誤差に変換する
     * 
     * @param error この層の誤差 
     * @return 前の層の誤差
     */
    protected abstract float[] toPrevError(float[] error);
    
    /**
     * この層を更新する
     */
    @Override
    public final void update() {
        this.update(this.error);
    }
    
    /**
     * この層への入力の次元
     */
    @Override
    public int inSize() {
        return this.inSize;
    }

    /**
     * この層からの出力の次元
     */
    @Override
    public int outSize() {
        return this.outSize;
    }
    
    /**
     * 末尾にバイアスを追加する
     */
    private float[] addBias(float[] data) {
        float[] d = new float[data.length + 1];
        System.arraycopy(data, 0, d, 0, data.length);
        d[d.length - 1] = 1f;
        return d;
    }
}
