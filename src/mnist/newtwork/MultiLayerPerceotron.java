package mnist.newtwork;

import mnist.model.LearningData;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import mnist.layer.Layer;
import mnist.layer.LayerManger;
import mnist.util.Util;

/**
 * 多層パーセプトロン
 */
public class MultiLayerPerceotron {
    
    /** 認識したパターン */
    private final List<LearningData> learning = new ArrayList<>();
    
    /** レイヤ管理クラス */
    private final LayerManger layerManager;
    
    /** 分類ラベル */
    private final int labelKind;
    
    /** 確率的勾配降下の1回分のイテレーション */
    private final int maxIteration;
    
    /**
     * コンストラクタ
     * 
     * @param layers        レイヤ
     * @param maxIteration  1回分のイテレーション
     */
    private MultiLayerPerceotron(List<Layer> layers, int maxIteration) {
        this.layerManager = new LayerManger(layers);
        this.labelKind = this.layerManager.getLabelKind();
        this.maxIteration = maxIteration;
    }
    
    /**
     * 教師データを追加
     * 
     * <pre>
     * 分類ラベルのユニットだけが発火する教師信号を作る。
     * </pre>
     * 
     * @param cls   ラベル
     * @param data  入力値
     */
    public void add(int cls, float[] data) {
        float[] label = new float[this.labelKind];
        Arrays.fill(label, 0);
        label[cls] = 1;
        this.learning.add(new LearningData(label, data));
    }
    
    /**
     * 学習
     */
    public void learn() {
        
        // 特徴量スケーリングする
        this.learning.forEach(ld -> ld.scalingFeature = Util.scaling(ld.feature, 256));
        
        int size = this.learning.size();
        for (int i = 0; i < maxIteration; i ++) {
            
            LearningData ld = this.learning.get((int) (Math.random() * size));
            
            // 順伝播
            float[] result = this.layerManager.forward(ld.scalingFeature);
            
            // 逆伝播
            this.layerManager.backward(result, ld.lavel);
        }
    }
    
    /**
     * 判定する
     * 
     * @param data
     * @return 
     */
    public int predict(float[] data) {
        
        float[] result = this.layerManager.forward(Util.scaling(data, 256));
        
        float max = Float.MIN_VALUE;
        int ans = 0;
        for (int i = 0; i < this.labelKind; i++) {
            if (result[i] > max) {
                max = result[i];
                ans = i;
            }
        }
        
        return ans;
    }
    
//    /**
//     * 入力データをスケーリングする
//     * 
//     * TODO スケールの幅をどう決めるか
//     * 
//     * @param data
//     * @return 
//     */
//    private float[] scaling(float[] data) {
//        float[] res = new float[data.length];
//        for (int i = 0; i < data.length; i++) {
//            res[i] = data[i] / 256;
//        }
//        return res;
//    }
    
    public static class Builder {
        
        private final List<Layer> layers = new ArrayList<>();
        
        private int maxIteration = 10000;
        
        private Builder() {
        }
        
        public Builder addLayer(Layer layer) {
            this.layers.add(layer);
            return this;
        }
        
        /**
         * 1回分のループ回数を設定
         *
         * @param maxIteration
         * @return
         */
        public Builder setMaxIteration(int maxIteration) {
            this.maxIteration = maxIteration;
            return this;
        }
        
        public MultiLayerPerceotron build() {
            return new MultiLayerPerceotron(this.layers, this.maxIteration);
        }
    }
    
    public static Builder builder() {
        return new Builder();
    }
}