package mnist.strategy;

import java.util.function.UnaryOperator;

public enum ActivationFunction {
    
    /** 恒等関数 */
    Identity(x -> x, y -> 1f)
            
    /** シグモイド */
    , Sigmoid(x -> (float) (1. / (1. + Math.exp(-x))), y -> (float) (y * (1f - y)))

    /** tanh */
    , Tanh(x -> (float) Math.tanh(x), y -> (float) (1f - y * y))

    /** ReLu */
    , ReLU(x -> Math.max(0, x), y -> y > 0 ? 1f : 0)

    ;

    private UnaryOperator<Float> f;
    private UnaryOperator<Float> df;

    private ActivationFunction(UnaryOperator<Float> f, UnaryOperator<Float> df) {
        this.f = f;
        this.df = df;
    }

    public float f(float x) {
        return f.apply(x);
    }

    public float df(float y) {
        return df.apply(y);
    }
}
