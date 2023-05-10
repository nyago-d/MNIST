package mnist.layer;

import mnist.filter.PoolingFilter;

/**
 * プーリング層
 *
 * <pre>
 * プーリングっていうか縮小？
 * とりあえず最大値プーリングで作っておこう。
 * </pre>
 */
public class PoolingLayer extends LayerBase {
    
    /** フィルタ */
    private final PoolingFilter filter;
    
    /**
     * コンストラクタ
     *
     * @param inputChannel  入力の枚数=出力の枚数=チャネル
     * @param inputWidth    入力画像の幅
     * @param inputHeight   入力画像の高さ
     * @param size          フィルタのサイズ（とりあえずフィルタは正方にしとく）
     */
    public PoolingLayer(int inputChannel, int inputWidth, int inputHeight, int size) {
        super(inputChannel, inputChannel, false);
        this.filter = new PoolingFilter(inputChannel, inputWidth, inputHeight, size);
    }
    
    /**
     * 順伝播
     */
    @Override
    public float[] forward() {
        return this.filter.fiter(super.input);
    }

    /**
     * プーリング層は更新しない
     * 
     * <pre>
     * 活性化もしないので次の層から来た誤差そのまま。
     * </pre>
     */
    @Override
    protected float[] getError(float[] nextError) {
        return nextError;
    }

    /**
     * プーリング層は更新しない
     */
    @Override
    protected void update(float[] error) {
    }
    
    /**
     * 前の層への誤差
     */
    @Override
    protected float[] toPrevError(float[] error) {
        return this.filter.toPrevError(super.input, super.output, error, 0);
    }
}