package mnist;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Paths;
import mnist.newtwork.MultiLayerPerceotron;
import mnist.strategy.ActivationFunction;
import java.util.List;
import javax.imageio.ImageIO;
import mnist.layer.ConvolutionLayer;
import mnist.layer.FullyConnectedLayer;
import mnist.layer.ImageOutputLayer;
import mnist.layer.PoolingLayer;
import mnist.strategy.Optimization.AdaDelta;
import mnist.strategy.Optimization.SGD;
import mnist.util.MnistLoader.LearningData;
import static mnist.util.MnistLoader.loadMnist;
import mnist.util.Util;

public class MNIST {
    
    private static final String PATH = "C:\\Users\\daiki\\Desktop\\MLP\\";
    
    public static void main(String[] args) throws IOException {
        
        List<LearningData> trainingData = loadMnist(PATH + "train-images.idx3-ubyte", PATH + "train-labels.idx1-ubyte");
        
        List<LearningData> testData = loadMnist(PATH + "t10k-images.idx3-ubyte", PATH + "t10k-labels.idx1-ubyte");
        
//        System.out.println("データ件数：" + trainingData.size());
//        System.out.println();
        
//        MultiLayerPerceotron mlp = new MultiLayerPerceotron(28 * 28, 128, 10, ActivationFunction.Sigmoid, ActivationFunction.Sigmoid);

        ConvolutionLayer conv1 = new ConvolutionLayer(10, 1, 28, 28, 3, ActivationFunction.ReLU, new SGD(0.01f));
        PoolingLayer pool1 = new PoolingLayer(10, 26, 26, 2);
        ImageOutputLayer imgOut1 = new ImageOutputLayer(10, 13, 13, 2);
        
        ConvolutionLayer conv2 = new ConvolutionLayer(30, 10, 13, 13, 3, ActivationFunction.ReLU, new SGD(0.01f));
//        PoolingLayer pool2 = new PoolingLayer(20, 10, 10, 2);
        ImageOutputLayer imgOut2 = new ImageOutputLayer(30, 11, 11, 4);
        
        FullyConnectedLayer hidden = new FullyConnectedLayer(256, 28 * 28, ActivationFunction.Sigmoid, new SGD(0.1f));
        FullyConnectedLayer hidden2 = new FullyConnectedLayer(128, 256, ActivationFunction.Sigmoid, new SGD(0.1f));
        FullyConnectedLayer out = new FullyConnectedLayer(10, 128, ActivationFunction.Sigmoid, new SGD(0.1f));
        
        MultiLayerPerceotron mlp = MultiLayerPerceotron.builder()
//                                                       .addLayer(conv1)
//                                                       .addLayer(pool1)
//                                                       .addLayer(imgOut1)
//                                                       .addLayer(conv2)
//                                                       .addLayer(pool2)
//                                                       .addLayer(imgOut2)
                                                       .addLayer(hidden)
                                                       .addLayer(hidden2)
                                                       .addLayer(out)
                                                       .setMaxIteration(10000)
                                                       .build();
        
        trainingData.forEach(ld -> mlp.add(ld.lavel, ld.feature));
        
        BufferedImage img2 = Util.arrayToImageMono(testData.get(testData.size() - 1).feature, 28, 28);
        ImageIO.write(img2, "png", Paths.get("C:\\Users\\daiki\\Desktop\\test2\\convImg\\org.png").toFile());
        
        for (int i = 0; i < 100; i++) {
            mlp.learn();
            int cnt = 0;
            for (LearningData ld : testData) {
                int predict = mlp.predict(ld.feature);
//                System.out.println(ld.lavel + " " + predict);
                if (ld.lavel == predict) {
                    cnt++;
                }
            }
            System.out.println((i + 1) + ". 正答率:" + ((float)cnt / (float)testData.size() * 100) + "%");
//            imgOut1.outputImage("C:\\Users\\daiki\\Desktop\\test2\\convImg\\1_", "png");
//            imgOut2.outputImage("C:\\Users\\daiki\\Desktop\\test2\\convImg\\2_", "png");
//            conv.outputFilterImage("C:\\Users\\daiki\\Desktop\\test2\\filter\\", "jpg");
        }
    }
}
