package mnist.util;

import mnist.util.StreamHelper;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MnistLoader {
    
    public static void saveAsText(List<LearningData> data, String fileName) {
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(fileName), Charset.defaultCharset())) {
            data.forEach(StreamHelper.throwingConsumer(line -> {
                writer.write(line.toString());
                writer.newLine();
            }));
        } catch (IOException e) {
            System.err.println("むり");
        }
    }
    
    public static List<LearningData> loadMnist(String imageFileName, String labelFileName) {

        float[][] images = loadImages(imageFileName);
        int[] labels = loadLabels(labelFileName);
        
        List<LearningData> list = new ArrayList<>();
        for (int i = 0; i < images.length; i++) {
            list.add(new LearningData(labels[i], images[i]));
        }
        
        return list;
    }

    private static float[][] loadImages(String fileName) {
        
        try (FileChannel fc = FileChannel.open(Paths.get(fileName))) {

            // ヘッダー部分の読み込み
            ByteBuffer headerBuf = ByteBuffer.allocateDirect(16);
            fc.read(headerBuf);
            headerBuf.rewind();

            int magicNum = headerBuf.getInt();  // マジックナンバー（いらない）
            int num = headerBuf.getInt();       // 画像の数
            int rowNum = headerBuf.getInt();    // 画像の縦px
            int colNum = headerBuf.getInt();    // 画像の横px

            // 画像データ部分の読み込み
            ByteBuffer buf = ByteBuffer.allocateDirect(num * rowNum * colNum);
            fc.read(buf);
            buf.rewind();

            // 1画像分のデータサイズ
            int dataSize = rowNum * colNum;

            float[][] features = new float[num][];            
            for (int i = 0; i < num; i++) {
                float[] feature = new float[dataSize];

                 // & 0xff する事で unsigned の値へ変換
                for (int j = 0; j < dataSize; j++) {
                    feature[j] = buf.get() & 0xff;
                }
                
                features[i] = feature;
            }
            return features;

        } catch(IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    private static int[] loadLabels(String fileName) {
        
        try (FileChannel fc = FileChannel.open(Paths.get(fileName))) {

            // ヘッダー部分の読み込み
            ByteBuffer headerBuf = ByteBuffer.allocateDirect(8);
            fc.read(headerBuf);
            headerBuf.rewind();

            int magicNum = headerBuf.getInt();
            int num = headerBuf.getInt();

            // ラベルデータ部分の読み込み
            ByteBuffer buf = ByteBuffer.allocateDirect(num);
            fc.read(buf);
            buf.rewind();

            int[] labels = new int[num];
            for (int i = 0; i < num; i++) {
                labels[i] = buf.get() & 0xff;
            }

            return labels;

        } catch(IOException ex) {
            throw new RuntimeException(ex);
        }
    }
    
    /**
     * 学習データ
     */
    public static class LearningData {
        
        /** 分類ラベル（式中の記号だとtと書かれることが多い） */
        public final int lavel;
        
        /** 特徴量（式中の記号だとxと書かれることが多い） */
        public final float[] feature;
        
        /**
         * コンストラクタ
         * 
         * @param lavel     分類ラベル
         * @param feature   特徴量
         */
        LearningData(int lavel, float[] feature) {
            this.lavel = lavel;
            this.feature = feature;
        }
        
        @Override
        public String toString() {
            return IntStream.range(0, feature.length).mapToObj(idx -> String.valueOf(feature[idx])).collect(Collectors.joining(" "));
        }
    }
}
