package mnist.layer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * レイヤの管理者
 */
public class LayerManger {
    
    /** レイヤ（順伝播用） */
    private final List<Layer> layers;
    
    /** レイヤ（逆伝播用） */
    private final List<Layer> reverseLayers;
    
    /**
     * コンストラクタ
     * 
     * @param layers レイヤ
     */
    public LayerManger(List<Layer> layers) {
        this.layers = layers;
        List<Layer> r = new ArrayList<>(layers);
        Collections.reverse(r);
        this.reverseLayers = r;
    }
    
    /**
     * 順伝播
     * 
     * @param feature 特徴量
     * @return 結果
     */
    public float[] forward(float[] feature) {
        
        // 順伝播
        float[] inOut = feature;
        for (Layer layer : this.layers) {
            inOut = layer.forward(inOut);
        }
        
        return inOut;
    }
    
    /**
     * 逆伝播
     * 
     * @param answer
     * @param label
     */
    public void backward(float[] answer, float[] label) {
        
        // 逆伝播
        float[] delta = this.getError(answer, label);
        for (Layer layer : this.reverseLayers) {
            delta = layer.backward(delta);
        }
        
        for (Layer layer : this.reverseLayers) {
            layer.update();
        }
        
//        this.reverseLayers.parallelStream().forEach(Layer::update);
    }
    
    /**
     * 誤差を計算
     * 
     * <pre>
     * 出力層の活性化関数が何であれ、誤差はy - tになる。
     * 実際には誤差の種類が違うが、あまり気にしないことにする。
     * （恒等関数⇒二乗誤差、ソフトマックス⇒クロスエントロピーなど）
     * </pre>
     * 
     * @param answer
     * @param label
     * @return 
     */
    private float[] getError(float[] answer, float[] label) {
        float[] error = new float[answer.length];
        for (int i = 0; i < answer.length; i++) {
            error[i] = (answer[i] - label[i]);
        }
        return error;
    }
    
    /**
     * 分類ラベル数を取得
     */
    public int getLabelKind() {
        return this.reverseLayers.get(0).outSize();
    }
}
