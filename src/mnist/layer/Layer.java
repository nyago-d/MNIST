package mnist.layer;

/**
 *
 */
public interface Layer {
    
    /**
     * 順伝播
     * 
     * @param input 入力
     * @return 出力
     */
    public float[] forward(float[] input);
    
    /**
     * 逆伝播
     * 
     * @param nextError 次の層からの誤差
     * @return この層から前の層への誤差
     */
    public float[] backward(float[] nextError);
    
    /**
     * このレイヤを更新する（逆伝播終わってからまとめてならこれ使うけど…）
     */
    public void update();
    
    /**
     * 入力のサイズ（畳み込み層とプーリング層はチャネルでいい？）
     */
    public int inSize();
    
    /**
     * 出力のサイズ（畳み込み層とプーリング層はチャネルでいい？）
     */
    public int outSize();
}
