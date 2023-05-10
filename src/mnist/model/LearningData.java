package mnist.model;

/**
 * 学習データ
 */
public class LearningData {
    
    /** 分類ラベル */
    public final float[] lavel;
    
    /** 特徴量 */
    public final float[] feature;
    
    /** スケーリングした特徴量 */
    public float[] scalingFeature;
    
    /**
     * コンストラクタ
     * 
     * @param lavel
     * @param feature 
     */
    public LearningData(float[] lavel, float[] feature) {
        this.lavel = lavel;
        this.feature = feature;
    }
}
