package mnist;

import mnist.util.MnistLoader;
import mnist.util.Util;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import javax.imageio.ImageIO;
import mnist.layer.ConvolutionLayer;
import mnist.layer.LayerManger;
import mnist.layer.PoolingLayer;
import mnist.strategy.ActivationFunction;
import mnist.strategy.Optimization.AdaDelta;
import mnist.strategy.Optimization.SGD;
import org.junit.Test;

/**
 *
 * @author daiki
 */
public class LayerTest {
    
    public LayerTest() {
    }
    
//    @Test
    public void testPoolingLayer() {
        
        float[] input = {0,5,10,10,10,10,10,5,0,0,0,10,150,255,255,255,150,10,0,0,0,20,255,10,10,10,255,20,0,0,0,20,255,10,10,10,255,20,0,0,0,10,150,255,255,255,150,10,0,0,0,20,255,10,10,10,255,20,0,0,0,20,255,10,0,10,255,20,0,0,0,20,255,10,10,10,255,20,0,0,0,10,150,255,255,255,150,10,0,0,0,5,10,10,10,10,10,5,0,0};
        
        PoolingLayer pooling = new PoolingLayer(6, 4, 4, 2);
        
        float[] output = pooling.forward(input);

        for (int i = 0; i < output.length; i++) {
//            for (int j = 0; j < 2; j++) {
//                System.out.print(output[i * 2 + j] + " ");
//            }
//            System.out.println();
            System.out.println(output[i]);
        }
    }
    
    private static final String PATH = "C:\\Users\\daiki\\Desktop\\MLP\\";
    
//    @Test
    public void testImg() throws IOException {
        
//        BufferedImage img = ImageIO.read(Paths.get("C:\\Users\\daiki\\Desktop\\test2\\in.png").toFile());
//        
//        float[] data = Util.imageToArrayRGB(img);
//        
//        ConvolutionLayer convolution = new ConvolutionLayer(5, 3, 200, 200, 5);
//        data = convolution.forward(data);
//        
//        PoolingLayer pooling = new PoolingLayer(5, 196, 196, 2);
//        data = pooling.forward(data);
//        
//        for (int channel = 0; channel < 5; channel++) {
//            BufferedImage img2 = Util.arrayToImageMono(data, 98, 98, channel);
//            ImageIO.write(img2, "png", Paths.get("C:\\Users\\daiki\\Desktop\\test2\\out" + channel + ".png").toFile());
//        }

        List<MnistLoader.LearningData> testData = MnistLoader.loadMnist(PATH + "t10k-images.idx3-ubyte", PATH + "t10k-labels.idx1-ubyte");
        float[] data = testData.get(4).feature;
//        for (int i = 0; i < 28; i++) {
//            for (int j = 0; j < 28; j++) {
//                System.out.print(ld.feature[i * 28 + j] > 0 ? "■" : "　");
//            }
//            System.out.println();
//        }
//        for (int i = 0; i < 100; i++) {
//            BufferedImage img2 = Util.arrayToImageMono(testData.get(i).feature, 28, 28);
//            ImageIO.write(img2, "png", Paths.get("C:\\Users\\daiki\\Desktop\\test2\\out-" + i + "-" + testData.get(i).lavel + ".png").toFile());
//        }

        ConvolutionLayer convolution = new ConvolutionLayer(10, 1, 28, 28, 3, ActivationFunction.ReLU, new SGD(0.01f));
        data = convolution.forward(data);
        
//        PoolingLayer pooling = new PoolingLayer(10, 24, 24, 2);
//        data = pooling.forward(data);
        
        for (int channel = 0; channel < 10; channel++) {
            BufferedImage img2 = Util.arrayToImageMono(data, 26, 26, channel);
            ImageIO.write(img2, "png", Paths.get("C:\\Users\\daiki\\Desktop\\test2\\out" + channel + ".png").toFile());
        }
    }
    
    private static final String ROOT = "C:\\Users\\daiki\\Desktop\\test2\\";
    
    @Test
    public void testBackProp() throws IOException {
        
        BufferedImage img = ImageIO.read(Paths.get(ROOT + "in.png").toFile());
        float[] data = Util.imageToArrayRGB(img);
        
        BufferedImage ans = ImageIO.read(Paths.get(ROOT + "ans3.png").toFile());
        float[] ansData = Util.imageToArrayRGB(ans);

        ConvolutionLayer convLayer = new ConvolutionLayer(3, 3, 200, 200, 3, ActivationFunction.ReLU, new AdaDelta(10));
        ConvolutionLayer convLayer2 = new ConvolutionLayer(3, 3, 198, 198, 3, ActivationFunction.ReLU, new AdaDelta(10));
//        PoolingLayer poolLayer = new PoolingLayer(3, 196, 196, 2);
//        PoolingLayer poolLayer2 = new PoolingLayer(3, 29, 29, 2);
        LayerManger lm = new LayerManger(Arrays.asList(convLayer, convLayer2));
        
        data = this.scaling(data);
        ansData = this.scaling(ansData);
        
        float[] res = null;
//        res = lm.forward(data);
        for (int i = 0; i < 1000; i++) {
        
            // 順伝播
            res = lm.forward(data);

            if (i % 100 == 0) {

                BufferedImage img2 = Util.arrayToImageRGB(this.rscaling(res), 196, 196);
                ImageIO.write(img2, "png", Paths.get(ROOT + i + ".png").toFile());

                float[] error = this.getError(ansData, res);

                System.out.println(IntStream.range(0, error.length).mapToDouble(j -> error[j]).sum());
            }
            
            // 逆伝播
            lm.backward(res, ansData);
        }
        
        BufferedImage img2 = Util.arrayToImageRGB(this.rscaling(res), 196, 196);
        ImageIO.write(img2, "png", Paths.get(ROOT + "after.png").toFile());
    }
    
    private float[] getError(float[] label, float[] result) {
        float[] error = new float[result.length];
        for (int i = 0; i < result.length; i++) {
            error[i] = Math.abs(result[i] - label[i]);
        }
        return error;
    }
    
    private float[] scaling(float[] data) {
        float[] res = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            res[i] = data[i] / 256;
        }
        return res;
    }
    
    private float[] rscaling(float[] data) {
        float[] res = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            res[i] = data[i] * 256;
        }
        return res;
    }
}
