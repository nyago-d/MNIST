package mnist.layer;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.stream.IntStream;
import javax.imageio.ImageIO;
import mnist.util.Util;

/**
 * 畳み込んだりプーリングした画像を表示してみるためのレイヤ
 * 最後に突っ込んだ画像の末路が見える
 */
public class ImageOutputLayer extends LayerBase {
    
    private final int channel;
    
    private final int inputWidth;
    
    private final int inputHeight;
    
    private final int zoom;

    public ImageOutputLayer(int channel, int inputWidth, int inputHeight, int zoom) {
        super(0, 0, false);
        this.channel = channel;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.zoom = zoom;
    }

    @Override
    protected float[] forward() {
        return super.input;
    }

    @Override
    protected float[] getError(float[] nextError) {
        return nextError;
    }

    @Override
    protected void update(float[] error) {
    }

    @Override
    protected float[] toPrevError(float[] error) {
        return error;
    }
    
    public void outputImage(String path, String extension) throws IOException {
        
        double max = IntStream.range(0, input.length).mapToDouble(i -> input[i]).max().getAsDouble();
        
        for (int i = 0; i < channel; i++) {
            float[] split = new float[inputWidth * inputHeight];
            System.arraycopy(input, inputWidth * inputHeight * i, split, 0, inputWidth * inputHeight);
            BufferedImage img = Util.arrayToImageMono(Util.scaling(split, (float) (max / 256)), inputWidth, inputHeight);
            img = this.changSize(img, inputWidth * zoom, inputHeight * zoom);
            ImageIO.write(img, extension, Paths.get(path + "conv" + i + "." + extension).toFile());
        }
    }
    
    private BufferedImage changSize(BufferedImage image, int width, int height) {
        BufferedImage shrinkImage = new BufferedImage(width,height, image.getType());
        Graphics2D g2d = shrinkImage.createGraphics();
        g2d.drawImage(image, 0, 0, width, height, null);
        return shrinkImage;
    }
}
