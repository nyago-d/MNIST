package mnist.filter;

/**
 * フィルタの基底
 */
public abstract class Filter {
    
    /** 入力のチャネル */
    protected final int inputChannel;
    
    /** フィルタのサイズ（正方） */
    protected final int filterSize;
    
    /** 入力画像の幅 */
    protected final int inputWidth;
    
    /** 入力画像の高さ */
    protected final int inputHeight;
    
    /** 入力画像のサイズ */
    protected final int inputImageSize;
    
    /** 出力画像のサイズ */
    protected final int outputImageSize;
    
    /** フィルタのストライド */
    protected final int stride;
    
    /** 複数チャネルを1チャネルにマージするか */
    private final boolean mergeChannels;
    
    /**
     * コンストラクタ
     * 
     * @param inputChannel
     * @param inputWidth
     * @param inputHeight
     * @param size
     * @param stride 
     */
    Filter(int inputChannel, int inputWidth, int inputHeight, int filterSize, int stride, boolean mergeChannels) {
        this.inputChannel = inputChannel;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.filterSize = filterSize;
        this.stride = stride;
        this.mergeChannels = mergeChannels;
        this.inputImageSize = inputWidth * inputHeight;
        this.outputImageSize = ((this.inputWidth - this.filterSize) / this.stride + 1) * ((this.inputHeight - this.filterSize) / this.stride + 1);
    }
    
    /**
     * フィルタする
     * 
     * @param input 入力
     * @return フィルタの結果
     */
    public float[] fiter(float[] input) {
        
        float[] res = new float[this.outputImageSize * (this.mergeChannels ? 1 : this.inputChannel)];
        for (int channel = 0; channel < this.inputChannel; channel++) {

            // フィルタの左上が指す位置
            int[] pos = {0, 0};

            for (int i = 0; i < this.outputImageSize; i++) {

                // フィルタする（入力が多チャネルの場合、合算になる）
                if (this.mergeChannels) {
                    res[i] += this.fiter(input, channel, pos);
                    
                // フィルタする（チャネルごとに）
                } else {
                    res[this.outputImageSize * channel + i] = this.fiter(input, channel, pos);
                }

                // 横移動
                pos[1] += stride;

                // はみ出たら縦移動
                if (pos[1] + this.filterSize - 1 >= this.inputWidth) {
                    pos[0] += stride;
                    pos[1] = 0;
                }
            }
        }

        return res;
    }
    
    /**
     * フィルタする
     *
     * @param input     入力
     * @param channel   チャネル
     * @param pos       フィルタのクリッピング位置
     * @return フィルタの結果
     */
    protected abstract float fiter(float[] input, int channel, int[] pos);
    
    /**
     * 更新する
     * 
     * @param input     入力
     * @param error     誤差
     */
    public void update(float[] input, float[] error) {
        
        int len = error.length;
        
        for (int channel = 0; channel < this.inputChannel; channel++) {

            // フィルタの左上が指す位置
            int[] pos = {0, 0};

            for (int i = 0; i < len; i++) {
                
                // 更新する
                this.update(input, channel, pos, error[i], len);

                // 横移動
                pos[1] += stride;

                // はみ出たら縦移動
                if (pos[1] + this.filterSize - 1 >= this.inputWidth) {
                    pos[0] += stride;
                    pos[1] = 0;
                }
            }
        }
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
    protected abstract void update(float[] input, int channel, int[] pos, float error, int len);
    
    /**
     * 前の層への誤差を計算する
     * 
     * @param error
     * @return 
     */
    public float[] toPrevError(float[] input, float[] output, float[] error, int outChannel) {
        
        float[] prevError = new float[input.length];

        for (int inChannel = 0; inChannel < this.inputChannel; inChannel++) {

            // フィルタの左上が指す位置
            int[] pos = {0, 0};

            for (int i = 0; i < this.outputImageSize; i++) {

                // 出力側のインデックス
                int idx = this.outputImageSize * (this.mergeChannels ? outChannel : inChannel) + i;
                
                // 誤差を計算
                this.setPrevError(input, inChannel, pos, output[idx], error[idx], prevError);

                // 横移動
                pos[1] += stride;

                // はみ出たら縦移動
                if (pos[1] + this.filterSize - 1 >= this.inputWidth) {
                    pos[0] += stride;
                    pos[1] = 0;
                }
            }
        }

        return prevError;
    }
    
    protected abstract void setPrevError(float[] input, int channel, int[] pos, float output, float error, float[] prevError);
    
    /**
     * 位置をインデックスに変換する
     * 
     * @param pos
     * @param offsetY
     * @param offsetX
     * @return 
     */
    protected int toIndex(int[] pos, int offsetY, int offsetX) {
        return (pos[0] + offsetY) * this.inputWidth + (pos[1] + offsetX);
    }
}
