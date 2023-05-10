package mnist.util;

import java.awt.image.BufferedImage;
import java.util.stream.IntStream;

public class Util {
    
    /** 
     * 画像から配列へ変換
     */
    public static float[] imageToArrayRGB(BufferedImage img) {
        
        float[] imageData = new float[3 * img.getWidth() * img.getHeight()];
        
        int width = img.getWidth();
        int height = img.getHeight();
        
        for(int x = 0; x < width; ++x){
            for(int y = 0; y < height; ++y){
                int rgb = img.getRGB(x, y);
                int pos = y * height + x;
                imageData[pos] = (rgb >> 16 & 0xff);
                imageData[pos + width * height] = (rgb >> 8 & 0xff);
                imageData[pos + 2 * width * height] = (rgb & 0xff);
            }
        }
        
        return imageData;
    }
    
    public static float[] imageToArrayMono(BufferedImage img) {
        
        float[] imageData = new float[img.getWidth() * img.getHeight()];
        
        int width = img.getWidth();
        int height = img.getHeight();
        
        for(int x = 0; x < width; ++x){
            for(int y = 0; y < height; ++y){
                int rgb = img.getRGB(x, y);
                int pos = y * height + x;
                
                int red   = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue  = rgb & 0xFF;

                imageData[pos] = (red + green + blue) / 3;
            }
        }
        
        return imageData;
    }
    
    public static BufferedImage arrayToImageRGB(float[] data, int width, int height) {
        return arrayToImageRGB(data, width, height, 0);
    }
    
    /**
     * 配列から画像に変換
     * 
     * @param filteredData
     * @param width
     * @param height
     * @return 
     */
    public static BufferedImage arrayToImageRGB(float[] data, int width, int height, int channel) {
        
        int offset = width * height * channel;
        BufferedImage filtered = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for(int x = 0; x < width; ++x){
            for(int y = 0; y < height; ++y){
                filtered.setRGB(y, x,
                        ((int)(data[offset + x * height + y]) << 16) +
                        ((int)(data[offset + x * height + y + width * height]) << 8) +
                         (int)(data[offset + x * height + y + 2 * width * height]));
            }
        }
        
        return filtered;
    }
    
    public static BufferedImage arrayToImageMono(float[] data, int width, int height) {
        return arrayToImageMono(data, width, height, 0);
    }
    
    public static BufferedImage arrayToImageMono(float[] data, int width, int height, int channel) {
        
        float[] colorData = new float[width * height * 3];
        
        int offset = width * height * channel;
        IntStream.range(0, width).parallel().forEach(x -> {
            for(int y = 0; y < height; ++y){
                int i = x * height + y;
                float c = data[offset + i];
                colorData[i] = limit(c);
                colorData[i + width * height] = limit(c);
                colorData[i + width * height * 2] = limit(c);
            }
        });
        
        return arrayToImageRGB(colorData, width, height);
    }
    
    /**
     * サチったのを丸める
     */
    private static float limit(float val) {
        if (val > 255) {
            return 255;
        } else if (val < 0) {
            return 0;
        } else {
            return val;
        }
    }
    
    /**
     * 引数の値で割ってスケーリングする
     * 
     * @param data
     * @param divideBy
     * @return 
     */
    public static float[] scaling(float[] data, float divideBy) {
        float[] res = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            res[i] = data[i] / divideBy;
        }
        return res;
    }
}
