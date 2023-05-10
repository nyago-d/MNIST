package mnist.filter;

/**
 * プーリング層用フィルタ
 */
public class PoolingFilter extends Filter {
    
    /**
     * コンストラクタ
     *
     * @param inputChannel
     * @param inputWidth
     * @param inputHeight
     * @param filterSize
     */
    public PoolingFilter(int inputChannel, int inputWidth, int inputHeight, int filterSize) {
        super(inputChannel, inputWidth, inputHeight, filterSize, filterSize, false);
    }
    
    /**
     * フィルタする（窓の中の最大値を取得）
     *
     * @param input    入力
     * @param channel   チャネル
     * @param pos       フィルタのクリッピング位置
     */
    @Override
    protected float fiter(float[] input, int channel, int[] pos) {

        float max = Float.NEGATIVE_INFINITY;

        // 最大値更新
        for (int j = 0; j < super.filterSize; j++) {
            for (int k = 0; k < super.filterSize; k++) {
                max = Math.max(max, input[super.inputImageSize * channel + super.toIndex(pos, j, k)]);
            }
        }

        return max;
    }
    
    /**
     * プーリング層に更新はない
     * 
     * <pre>
     * これをオーバーライドしなくても更新はされないが、
     * 処理が無駄なので空にしておく。
     * </pre>
     */
    @Override
    public void update(float[] input, float[] error) {
    }

    /**
     * 更新しない
     */
    @Override
    protected void update(float[] input, int channel, int[] pos, float error, int len) {
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
    protected void setPrevError(float[] input, int channel, int[] pos, float output, float error, float[] prevError) {
        
        boolean isProp = false;
        for (int j = 0; j < super.filterSize; j++) {
            for (int k = 0; k < super.filterSize; k++) {
                int idx = super.inputImageSize * channel + super.toIndex(pos, j, k);
                
                // この位置が最大値としてプーリングされていれば逆伝播する（複数あっても最初だけ）
                if (!isProp && input[idx] == output) {
                    prevError[idx] = error;
                    
                // 採用されていない箇所は0で逆伝播する
                } else {
                    prevError[idx] = 0;
                    isProp = true;
                }
            }
        }
    }
}