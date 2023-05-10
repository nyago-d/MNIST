package mnist.layer;

/**
 * 入力層
 * 
 * <pre>
 * この層なくてもよくない？
 * </pre>
 */
public class InputLayer extends LayerBase {
    
    /**
     * コンストラクタ
     * 
     * @param inSize
     */
    public InputLayer(int inSize) {
        super(inSize, inSize, false);
    }
    
    /**
     * 順伝播
     */
    @Override
    protected float[] forward() {
        return input;
    }
    
    /**
     * 前の層への誤差
     */
    @Override
    protected float[] toPrevError(float[] error) {
        return null;
    }
    
    /**
     * この層での誤差
     * @param nextError
     * @return
     */
    @Override
    protected float[] getError(float[] nextError) {
        return null;
    }
    
    /**
     * 更新
     * @param error
     */
    @Override
    protected void update(float[] error) {
    }
}
