package mnist.strategy;

/**
 * 最適化
 */
public interface Optimization {
    
    float getDeltaWeight(float gw, int i);
    
    Optimization get();
    
    /**
     * 勾配法
     */
    public static class SGD implements Optimization {
        
        private final float learningRate;
        
        public SGD(float learningRate) {
            this.learningRate = learningRate;
        }

        @Override
        public float getDeltaWeight(float gw, int i) {
            return learningRate * gw;
        }
        
        @Override
        public SGD get() {
            return this;
        }
    }
    
    /**
     * AdaDelta
     */
    public static class AdaDelta implements Optimization {
        
        private static final float ALPHA = 1f;
        
        private static final float BETA = 0.95f;
        
        private final float EPS = (float) Math.pow(10, -8);
        
        private final float[] v;
        
        private final float[] r;
        
        private final float[] s;
        
        private final int weightSize;
        
        public AdaDelta(int weightSize) {
            this.v = new float[weightSize];
            this.r = new float[weightSize];
            this.s = new float[weightSize];
            this.weightSize = weightSize;
        }
        
        @Override
        public float getDeltaWeight(float gw, int i) {
            this.r[i] = BETA * this.r[i] + (1 - BETA) * gw * gw;
            this.v[i] = (float) (Math.sqrt(this.s[i] + EPS) / Math.sqrt(this.r[i] + EPS) * gw);
            this.s[i] = BETA * this.s[i] + (1 - BETA) * this.v[i] * this.v[i];
            return  ALPHA * this.v[i];
        }
        
        public AdaDelta get() {
            return new AdaDelta(this.weightSize);
        } 
    }
    
    //                this.v[i] = BETA * this.v[i] + (1 - BETA) * gw;
//                this.r[i] = GAMMA * this.r[i] + (1 - GAMMA) * gw * gw;
//                float mt = (float) (this.v[i] / (1 - Math.pow(BETA, cnt)));
//                float vt = (float) (this.r[i] / (1 - Math.pow(GAMMA, cnt)));
//                this.weight[i] -= ALPHA / (Math.sqrt(mt) + EPS) * vt; // Adam

//                this.r[i] += gw * gw;
//                this.weight[i] -= learningRate / (Math.sqrt(this.r[i]) + EPS) * gw;   // AdaGrad

//                this.r[i] = BETA * this.r[i] + (1 - BETA) * gw * gw;
//                this.v[i] = (float) (Math.sqrt(this.s[i] + EPS) / Math.sqrt(this.r[i] + EPS) * gw);
//                this.weight[i] -= ALPHA * this.v[i];
//                this.s[i] = BETA * this.s[i] + (1 - BETA) * this.v[i] * this.v[i];   // AdaDelta
}
